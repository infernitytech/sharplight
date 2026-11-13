using System.Diagnostics;
using System.Threading.Channels;
using Sharplight.Tokenizers;

namespace Sharplight.Llm.Serving;

/// <summary>
/// Continuous-batching scheduler over an <see cref="IBatchDecoder"/>. A single scheduler thread owns the decoder; every step it
/// mixes one decode token of each running sequence with prompt chunks of newly admitted ones (decode first, then prefill, within
/// the batch-token budget), samples, streams text deltas and frees finished slots. With a one-slot decoder requests simply run one
/// after another; with a multi-slot decoder they are batched together.
/// </summary>
public sealed class InferenceEngine : IDisposable
{
    private sealed class Slot { public int Index; public List<int> Tokens = []; public long LastUsed; public Seq? Active; }

    private sealed class Seq
    {
        public long Id;
        public required GenerationRequest Req;
        public required ChannelWriter<GenerationChunk> Writer;
        public required int[] Prompt;
        public required int MaxTokens;
        public required long SubmitTicks;
        public long DeadlineTicks = long.MaxValue, QueueDeadlineTicks = long.MaxValue;
        public CancellationTokenRegistration Reg;
        public volatile bool CancelRequested;
        public int Finished;
        public Slot? Slot;
        public int PromptPos, Cached, Generated, Pending;
        public long LastTokenTicks;
        public RequestSampler Sampler = null!;
        public StreamingDecoder Decoder = null!;
        public StopStringFilter Filter = null!;
        public HashSet<int> StopIds = null!;
        public bool Prefilling => PromptPos < Prompt.Length;
    }

    private readonly IBatchDecoder _decoder;
    private readonly ITokenizer _tokenizer;
    private readonly EngineOptions _opt;
    private readonly Slot[] _slots;
    private readonly List<Seq> _queue = [];
    private readonly List<Seq> _active = [];
    private readonly object _gate = new();
    private readonly Thread _thread;
    private readonly HashSet<int> _eog;
    private readonly TaskCompletionSource _idle = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly int _budget;
    private long _nextId, _tick;
    private int _rr, _transit;
    private bool _stop, _draining;

    public EngineMetrics Metrics { get; } = new();
    public ITokenizer Tokenizer => _tokenizer;
    public int SlotCount => _decoder.SlotCount;
    public int ContextLength => _decoder.ContextLength;
    public bool IsDraining => _draining;

    public InferenceEngine(IBatchDecoder decoder, ITokenizer tokenizer, EngineOptions? options = null)
    {
        _decoder = decoder;
        _tokenizer = tokenizer;
        _opt = options ?? new();
        _slots = Enumerable.Range(0, decoder.SlotCount).Select(i => new Slot { Index = i }).ToArray();
        _eog = [.. tokenizer.EndOfGenerationIds];
        _budget = Math.Max(1, _opt.MaxBatchTokens > 0 ? Math.Min(_opt.MaxBatchTokens, decoder.MaxBatchTokens) : decoder.MaxBatchTokens);
        _thread = new Thread(Run) { Name = "sharplight-scheduler", IsBackground = true };
        _thread.Start();
    }

    public EngineMetricsSnapshot GetMetrics()
    {
        int q, a;
        lock (_gate)
        {
            q = _queue.Count;
            a = _active.Count;
        }
        return Metrics.Snapshot(q, a, _slots.Length);
    }

    // ---- submission -------------------------------------------------------------------------------------------------

    /// <summary>Queues a request. Throws an <see cref="EngineException"/> when it cannot be admitted (queue full, prompt too long, draining).</summary>
    public GenerationHandle Submit(GenerationRequest request, CancellationToken ct = default)
    {
        int n = request.PromptTokens.Length;
        if (n == 0)
        {
            Metrics.Rejected();
            throw new EngineException("The prompt is empty.", "invalid_request");
        }
        foreach (var t in request.PromptTokens)
        {
            if ((uint)t >= (uint)_decoder.VocabSize)
            {
                Metrics.Rejected();
                throw new EngineException($"Token id {t} is outside the vocabulary.", "invalid_request");
            }
        }

        if (n >= _decoder.ContextLength)
        {
            Metrics.Rejected();
            throw new PromptTooLongException(n, _decoder.ContextLength - 1);
        }
        int room = _decoder.ContextLength - n;
        var ch = Channel.CreateUnbounded<GenerationChunk>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = false });
        long now = Stopwatch.GetTimestamp();
        Seq? seq = null;
        var handle = new GenerationHandle(Interlocked.Increment(ref _nextId), ch, () => RequestCancel(seq!));
        seq = new Seq
        {
            Id = handle.Id, Req = request, Writer = ch.Writer, Prompt = request.PromptTokens, SubmitTicks = now,
            MaxTokens = request.MaxTokens > 0 ? Math.Min(request.MaxTokens, room) : room,
        };
        if ((request.Timeout ?? _opt.DefaultTimeout) is { } to)
        {
            seq.DeadlineTicks = now + (long)(to.TotalSeconds * Stopwatch.Frequency);
        }

        if ((request.QueueTimeout ?? _opt.DefaultQueueTimeout) is { } qt)
        {
            seq.QueueDeadlineTicks = now + (long)(qt.TotalSeconds * Stopwatch.Frequency);
        }

        lock (_gate)
        {
            if (_stop || _draining)
            {
                Metrics.Rejected();
                throw new EngineStoppedException();
            }
            if (_queue.Count >= _opt.MaxQueue)
            {
                Metrics.Rejected();
                throw new QueueFullException(_opt.MaxQueue);
            }
            _queue.Add(seq);
            Metrics.Submitted();
            Monitor.PulseAll(_gate);
        }
        if (ct.CanBeCanceled)
        {
            seq.Reg = ct.Register(() => RequestCancel(seq));
        }

        return handle;
    }

    /// <summary>Submits a request and streams its chunks (admission errors are thrown on the first MoveNext).</summary>
    public async IAsyncEnumerable<GenerationChunk> GenerateAsync(GenerationRequest request, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        var h = Submit(request, ct);
        await foreach (var c in h.ReadAllAsync(ct).ConfigureAwait(false))
        {
            yield return c;
        }
    }

    private void RequestCancel(Seq seq)
    {
        seq.CancelRequested = true;
        bool queued;
        lock (_gate)
        {
            queued = _queue.Remove(seq);
        }

        if (queued)
        {
            Finalize(seq, FinishReason.Cancelled, null, null);
        }
        else
        {
            lock (_gate)
            {
                Monitor.PulseAll(_gate);
            }
        }
    }

    // ---- shutdown ---------------------------------------------------------------------------------------------------

    /// <summary>
    /// Stops admitting requests and waits until queued and running ones finished. Returns false when <paramref name="timeout"/>
    /// elapsed first (the remaining requests are then cancelled). Call <see cref="Dispose"/> afterwards.
    /// </summary>
    public async Task<bool> DrainAsync(TimeSpan timeout)
    {
        lock (_gate)
        {
            _draining = true;
            if (_queue.Count == 0 && _active.Count == 0 && _transit == 0)
            {
                return true;
            }

            Monitor.PulseAll(_gate);
        }
        var done = await Task.WhenAny(_idle.Task, Task.Delay(timeout)).ConfigureAwait(false) == _idle.Task;
        if (!done)
        {
            CancelAll();
        }

        return done;
    }

    private void CancelAll()
    {
        List<Seq> qs;
        lock (_gate)
        {
            qs = [.. _queue];
            _queue.Clear();
            foreach (var s in _active) { s.CancelRequested = true; }
            Monitor.PulseAll(_gate);
        }
        foreach (var s in qs)
        {
            Finalize(s, FinishReason.Cancelled, null, null);
        }
    }

    public void Dispose()
    {
        lock (_gate) { _draining = true; }
        CancelAll();
        lock (_gate)
        {
            _stop = true;
            Monitor.PulseAll(_gate);
        }
        if (Thread.CurrentThread != _thread)
        {
            _thread.Join();
        }

        _decoder.Dispose();
    }

    // ---- scheduler --------------------------------------------------------------------------------------------------

    private void Run()
    {
        while (true)
        {
            lock (_gate)
            {
                while (_queue.Count == 0 && _active.Count == 0 && !_stop)
                {
                    if (_draining)
                    {
                        _idle.TrySetResult();
                    }

                    Monitor.Wait(_gate);
                }
                if (_stop && _active.Count == 0)
                {
                    break;
                }
            }
            try
            {
                Reap();
                Admit();
                if (_active.Count > 0)
                {
                    StepOnce();
                }
            }
            catch (Exception ex)
            {
                FailAll(ex.Message);
            }
        }
        lock (_gate) { _idle.TrySetResult(); }
    }

    private void FailAll(string message)
    {
        foreach (var s in _active.ToArray())
        {
            Complete(s, FinishReason.Error, message, null, reset: true);
        }
    }

    /// <summary>Handles cancellations and expired deadlines of running and waiting requests.</summary>
    private void Reap()
    {
        long now = Stopwatch.GetTimestamp();
        List<Seq>? expired = null;
        lock (_gate)
        {
            foreach (var s in _queue)
            {
                if (s.CancelRequested || now > s.DeadlineTicks || now > s.QueueDeadlineTicks)
                {
                    (expired ??= []).Add(s);
                }
            }
        }

        if (expired is not null)
        {
            foreach (var s in expired)
            {
                lock (_gate)
                {
                    if (!_queue.Remove(s))
                    {
                        continue;
                    }
                }

                Finalize(s, s.CancelRequested ? FinishReason.Cancelled : FinishReason.Timeout, s.CancelRequested ? null : "timed out waiting for a free slot", null);
            }
        }

        foreach (var s in _active.ToArray())
        {
            if (s.CancelRequested)
            {
                Complete(s, FinishReason.Cancelled, null, null);
            }
            else if (now > s.DeadlineTicks)
            {
                Complete(s, FinishReason.Timeout, "request timed out", null);
            }
        }
    }

    private void Admit()
    {
        while (true)
        {
            Slot? free = null;
            foreach (var s in _slots)
            {
                if (s.Active is null)
                {
                    free = s;
                    break;
                }
            }

            if (free is null)
            {
                return;
            }

            Seq seq;
            lock (_gate)
            {
                if (_queue.Count == 0)
                {
                    return;
                }

                int best = 0;
                for (int i = 1; i < _queue.Count; i++)
                {
                    if (_queue[i].Req.Priority > _queue[best].Req.Priority)
                    {
                        best = i;   // stable: earliest wins ties
                    }
                }

                seq = _queue[best];
                _queue.RemoveAt(best);
                _transit++;
            }
            try { AssignSlot(seq); }
            finally { lock (_gate)
                {
                    _transit--;
                }
            }
        }
    }

    private int ReusableTokens(Slot slot, int[] prompt)
    {
        if (!_opt.PrefixCache || slot.Tokens.Count == 0)
        {
            return 0;
        }

        int common = 0, max = Math.Min(slot.Tokens.Count, prompt.Length);
        while (common < max && slot.Tokens[common] == prompt[common])
        {
            common++;
        }
        // One prompt token must always be (re)computed to obtain the logits of the last prompt position.
        if (_decoder.CanTruncate)
        {
            return Math.Min(common, prompt.Length - 1);
        }

        return common == slot.Tokens.Count && common < prompt.Length ? common : 0;
    }

    private void AssignSlot(Seq seq)
    {
        Slot? best = null;
        int bestReuse = -1;
        foreach (var s in _slots)
        {
            if (s.Active is not null)
            {
                continue;
            }

            int reuse = ReusableTokens(s, seq.Prompt);
            // Equal reuse (typically none): take an empty slot first, then evict the least recently used one.
            static long Rank(Slot x) => x.Tokens.Count == 0 ? long.MinValue : x.LastUsed;
            bool better = best is null || reuse > bestReuse || (reuse == bestReuse && Rank(s) < Rank(best));
            if (better)
            {
                best = s;
                bestReuse = reuse;
            }
        }
        var slot = best!;
        int keep = Math.Max(0, bestReuse);
        if (slot.Tokens.Count > keep)
        {
            if (keep == 0)
            {
                _decoder.Reset(slot.Index);
            }
            else
            {
                _decoder.Truncate(slot.Index, keep);
            }

            slot.Tokens.RemoveRange(keep, slot.Tokens.Count - keep);
        }
        seq.Slot = slot;
        slot.Active = seq;
        seq.PromptPos = keep;
        seq.Cached = keep;
        seq.Sampler = new RequestSampler(seq.Req.Sampling, seq.Prompt);
        seq.Decoder = new StreamingDecoder(_tokenizer, seq.Req.SkipSpecialTokens);
        seq.Filter = new StopStringFilter(seq.Req.StopStrings);
        seq.StopIds = [.. seq.Req.StopTokenIds];
        _active.Add(seq);
        Metrics.Admitted(seq.Prompt.Length, keep, TimeSpan.FromSeconds((Stopwatch.GetTimestamp() - seq.SubmitTicks) / (double)Stopwatch.Frequency));
    }

    private void StepOnce()
    {
        int budget = _budget;
        var inputs = new List<SequenceInput>(_active.Count);
        var seqs = new List<Seq>(_active.Count);
        int nDecode = 0;
        // Decode first: one token for every running sequence (rotating the start when the budget cannot cover all of them).
        int count = _active.Count;
        _rr = (_rr + 1) % Math.Max(1, count);
        for (int k = 0; k < count && budget > 0; k++)
        {
            var s = _active[(k + _rr) % count];
            if (s.Prefilling)
            {
                continue;
            }

            inputs.Add(new SequenceInput(s.Slot!.Index, new[] { s.Pending }, true));
            seqs.Add(s);
            budget--;
            nDecode++;
        }
        foreach (var s in _active)
        {
            if (!s.Prefilling || budget <= 0)
            {
                continue;
            }

            int chunk = Math.Min(Math.Min(_opt.PrefillChunk, s.Prompt.Length - s.PromptPos), budget);
            bool last = s.PromptPos + chunk == s.Prompt.Length;
            inputs.Add(new SequenceInput(s.Slot!.Index, s.Prompt.AsMemory(s.PromptPos, chunk), last));
            seqs.Add(s);
            budget -= chunk;
        }
        if (inputs.Count == 0)
        {
            return;
        }

        var sw = Stopwatch.StartNew();
        var results = _decoder.Step(inputs);
        Metrics.Step(_budget - budget, sw.Elapsed);

        for (int i = 0; i < inputs.Count; i++)
        {
            var s = seqs[i];
            if (s.Finished != 0)
            {
                continue;
            }

            var slot = s.Slot!;
            var toks = inputs[i].Tokens;
            for (int j = 0; j < toks.Length; j++)
            {
                slot.Tokens.Add(toks.Span[j]);
            }

            if (s.Prefilling)
            {
                s.PromptPos += toks.Length;
            }

            if (!inputs[i].WantLogits)
            {
                continue;
            }

            try { OnLogits(s, results[i]!); }
            catch (Exception ex) { Complete(s, FinishReason.Error, ex.Message, null, reset: true); }
        }
    }

    private void OnLogits(Seq s, float[] logits)
    {
        long now = Stopwatch.GetTimestamp();
        int tok = s.Sampler.Sample(logits);
        List<TokenLogprob>? lps = null;
        if (s.Req.Logprobs)
        {
            var (lp, top) = RequestSampler.Logprobs(logits, tok, s.Req.TopLogprobs);
            lps = [new TokenLogprob(tok, TokenText(tok), lp, [.. top.Select(t => new TokenAlternative(t.Id, TokenText(t.Id), t.Logprob))])];
        }
        s.Sampler.Accept(tok);
        s.Generated++;
        if (s.Generated == 1)
        {
            Metrics.FirstToken(TimeSpan.FromSeconds((now - s.SubmitTicks) / (double)Stopwatch.Frequency));
        }

        if (s.StopIds.Contains(tok) || (s.Req.StopAtEog && _eog.Contains(tok)))
        {
            s.Generated--;   // the stop token itself is not part of the completion
            Complete(s, FinishReason.Stop, null, null, flush: true);
            return;
        }
        Metrics.Token(s.Generated == 1 ? null : TimeSpan.FromSeconds((now - s.LastTokenTicks) / (double)Stopwatch.Frequency));
        s.LastTokenTicks = now;
        string text = s.Filter.Push(s.Decoder.Push(tok));
        if (text.Length > 0 || lps is not null)
        {
            s.Writer.TryWrite(new GenerationChunk(text, lps));
        }

        if (s.Filter.Stopped)
        {
            Complete(s, FinishReason.Stop, null, s.Filter.Matched);
            return;
        }
        if (s.Generated >= s.MaxTokens || s.Slot!.Tokens.Count + 1 > _decoder.ContextLength)
        {
            Complete(s, FinishReason.Length, null, null, flush: true);
            return;
        }
        s.Pending = tok;
    }

    private string TokenText(int id) => _tokenizer.Decode([id]);

    /// <summary>Ends a running sequence: emits held-back text (when <paramref name="flush"/>), the final chunk, frees the slot.</summary>
    private void Complete(Seq s, FinishReason reason, string? error, string? stopSeq, bool flush = false, bool reset = false)
    {
        if (s.Finished != 0)
        {
            return;
        }

        if (flush)
        {
            string tail = s.Filter.Push(s.Decoder.Flush());
            if (!s.Filter.Stopped)
            {
                tail += s.Filter.Flush();
            }
            else
            {
                stopSeq = s.Filter.Matched;
            }

            if (tail.Length > 0)
            {
                s.Writer.TryWrite(new GenerationChunk(tail));
            }
        }
        var slot = s.Slot!;
        _active.Remove(s);
        slot.Active = null;
        slot.LastUsed = ++_tick;
        if (reset || reason == FinishReason.Error)
        {
            try { _decoder.Reset(slot.Index); } catch { /* the decoder is broken; later requests will fail visibly */ }
            slot.Tokens.Clear();
        }
        Finalize(s, reason, error, stopSeq);
    }

    private void Finalize(Seq s, FinishReason reason, string? error, string? stopSeq)
    {
        if (Interlocked.Exchange(ref s.Finished, 1) != 0)
        {
            return;
        }

        var usage = new GenerationUsage(s.Prompt.Length, s.Generated, s.Cached);
        s.Writer.TryWrite(new GenerationChunk("", null, reason, usage, error, stopSeq));
        s.Writer.TryComplete();
        s.Reg.Dispose();
        Metrics.Finished(reason, TimeSpan.FromSeconds((Stopwatch.GetTimestamp() - s.SubmitTicks) / (double)Stopwatch.Frequency));
    }
}
