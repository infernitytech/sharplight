using System.Diagnostics;
using System.Threading.Channels;
using Sharplight.Graph;
using Sharplight.Llm.Serving.Speculative;
using Sharplight.Llm.Structured;
using Sharplight.Tokenizers;

namespace Sharplight.Llm.Serving;

/// <summary>
/// Continuous-batching scheduler over an <see cref="IBatchDecoder"/>. A single scheduler thread owns the decoder; every step it
/// mixes the decode tokens of all running sequences with prompt chunks of newly admitted ones (decode first, then chunked
/// prefill, within a per-step token budget, so a long prompt never stalls the running streams), samples, streams text deltas
/// and frees finished slots. With a one-slot decoder requests simply run one after another; with a multi-slot decoder they are
/// batched together.
/// <para>
/// Optional per-request features: output constraints (<see cref="GenerationRequest.Constraint"/>, applied as a token mask
/// before sampling), minimum lengths, priorities with per-tenant fair queuing, and speculative decoding
/// (<see cref="EngineOptions.Speculator"/>) that verifies several draft tokens of a sequence in one forward pass.
/// </para>
/// <para>
/// Thread safety: all members may be called from any thread. The engine owns the decoder and disposes it. Idle slots keep their
/// context, so a later prompt that shares a prefix with it skips recomputing the shared part (<see cref="EngineOptions.PrefixCache"/>).
/// </para>
/// </summary>
/// <example>
/// <code>
/// using var model = LlmModel.Load(path, loadOptions);
/// using var engine = new InferenceEngine(model.CreateBatchDecoder(slots: 4), model.Tokenizer);
/// var request = new GenerationRequest
/// {
///     PromptTokens = model.Tokenizer.Encode("Once upon a time", addSpecial: true, parseSpecial: false),
///     MaxTokens = 64,
///     Sampling = SamplingParams.GreedyDefault,
/// };
/// await foreach (var chunk in engine.GenerateAsync(request))
/// {
///     Console.Write(chunk.Text);
/// }
/// </code>
/// </example>
public sealed class InferenceEngine : IDisposable
{
    /// <summary>How a sequence ends after a token was emitted (null = it continues).</summary>
    private readonly record struct Ending(FinishReason Reason, string? StopSequence, bool Flush);

    /// <summary>Result of sampling one step's logits of a sequence (computed in parallel, applied on the scheduler thread).</summary>
    private struct Outcome
    {
        public int Accepted;
        public Ending? Ending;
        public string? Error;
    }

    /// <summary>Below this many sequences with logits, sampling runs on the scheduler thread alone.</summary>
    private const int ParallelSamplingThreshold = 4;

    private readonly IBatchDecoder _decoder;
    private readonly ITokenizer _tokenizer;
    private readonly EngineOptions _opt;
    private readonly EngineSlot[] _slots;
    private readonly RequestQueue _queue;
    private readonly List<EngineSequence> _active = [];
    private readonly object _gate = new();
    private readonly Thread _thread;
    private readonly HashSet<int> _eog;
    private readonly TaskCompletionSource _idle = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly int _budget;
    private long _nextId, _tick;
    private int _rr, _transit;
    private bool _stop, _draining;
    private volatile int _kvTokens, _kvActiveTokens;

    // Scratch of the scheduler thread, reused every step.
    private readonly List<SequenceInput> _inputs = [];
    private readonly List<EngineSequence> _stepSeqs = [];
    private readonly List<int> _stepBaseLength = [];
    private Outcome[] _outcomes = [];

    // Device sampling (null when the decoder cannot sample on the device): per-input rows of the current step and the tokens /
    // log-probabilities the device chose for them.
    private readonly IDeviceSamplingDecoder? _deviceSampler;
    private SamplingRow?[] _deviceRows = [];
    private int[] _sampledTokens = [];
    private float[] _sampledLogprobs = [];

    // Speculative decoding (null when off or unsupported by the decoder).
    private readonly ISpeculator? _speculator;
    private readonly List<SpeculationRequest> _specBatch = [];
    private readonly SpeculationRequest[] _specRequests;
    private readonly ContextView[] _specViews;

    /// <summary>The live counters; most callers want <see cref="GetMetrics"/>.</summary>
    public EngineMetrics Metrics { get; } = new();
    /// <summary>The tokenizer of the served model, for encoding prompts.</summary>
    public ITokenizer Tokenizer => _tokenizer;
    /// <summary>Number of sequences that can run at the same time.</summary>
    public int SlotCount => _decoder.SlotCount;
    /// <summary>Context length of one sequence: prompt plus generated tokens.</summary>
    public int ContextLength => _decoder.ContextLength;
    /// <summary>Vocabulary size of the served model (length of a logits row; the domain of token masks).</summary>
    public int VocabSize => _decoder.VocabSize;
    /// <summary>True once <see cref="DrainAsync"/> or <see cref="Dispose"/> began: new requests are rejected.</summary>
    public bool IsDraining => _draining;
    /// <summary>True when speculative decoding is active (a speculator was configured and the decoder can roll back rows).</summary>
    public bool Speculates => _speculator is not null;

    /// <summary>Starts the scheduler thread. The engine takes ownership of <paramref name="decoder"/> and of <see cref="EngineOptions.Speculator"/>.</summary>
    /// <param name="decoder">The model execution; obtain it from <see cref="LlmModel.CreateBatchDecoder"/>.</param>
    /// <param name="tokenizer">Tokenizer of the same model (end-of-generation tokens, detokenization).</param>
    /// <param name="options">Scheduler settings; defaults when null.</param>
    public InferenceEngine(IBatchDecoder decoder, ITokenizer tokenizer, EngineOptions? options = null)
    {
        _decoder = decoder;
        _deviceSampler = decoder is IDeviceSamplingDecoder { SamplesOnDevice: true } d ? d : null;
        _tokenizer = tokenizer;
        _opt = options ?? new();
        _slots = [.. Enumerable.Range(0, decoder.SlotCount).Select(i => new EngineSlot(i))];
        _queue = new RequestQueue(_opt.FairShareHalfLife);
        _eog = [.. tokenizer.EndOfGenerationIds];
        _budget = Math.Max(1, _opt.MaxBatchTokens > 0 ? Math.Min(_opt.MaxBatchTokens, decoder.MaxBatchTokens) : decoder.MaxBatchTokens);
        _specRequests = new SpeculationRequest[decoder.SlotCount];
        _specViews = new ContextView[decoder.SlotCount];
        if (_opt.Speculator is { } spec)
        {
            if (decoder.CanTruncate && decoder.MaxLogitsRows > 1)
            {
                _speculator = spec;
                for (int i = 0; i < _specRequests.Length; i++)
                {
                    _specViews[i] = new ContextView();
                    _specRequests[i] = new SpeculationRequest { Slot = i, Draft = new int[spec.MaxDraftTokens] };
                }
            }
            else
            {
                // Recurrent state cannot be rolled back over rejected drafts, and without spare logits rows nothing can be verified.
                spec.Dispose();
            }
        }

        _thread = new Thread(Run) { Name = "sharplight-scheduler", IsBackground = true };
        _thread.Start();
    }

    /// <summary>A consistent snapshot of queue depth, slot usage, token counters and latency histograms.</summary>
    public EngineMetricsSnapshot GetMetrics()
    {
        int q, a;
        lock (_gate)
        {
            q = _queue.Count;
            a = _active.Count;
        }
        long capacity = (long)_slots.Length * _decoder.ContextLength;
        return Metrics.Snapshot(q, a, _slots.Length) with { KvCapacityTokens = capacity, KvTokens = _kvTokens, KvActiveTokens = _kvActiveTokens };
    }

    // ---- submission -------------------------------------------------------------------------------------------------

    /// <summary>Queues a request and returns immediately; read the output through the returned handle.</summary>
    /// <param name="request">The prompt tokens and generation settings.</param>
    /// <param name="ct">Cancels the request (it then finishes with <see cref="FinishReason.Cancelled"/>).</param>
    /// <exception cref="PromptTooLongException">The prompt leaves no room for at least one generated token.</exception>
    /// <exception cref="QueueFullException"><see cref="EngineOptions.MaxQueue"/> requests are already waiting.</exception>
    /// <exception cref="EngineStoppedException">The engine is draining or disposed.</exception>
    /// <exception cref="EngineException">The prompt is empty or contains ids outside the vocabulary (code <c>invalid_request</c>).</exception>
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
        EngineSequence? seq = null;
        var handle = new GenerationHandle(Interlocked.Increment(ref _nextId), ch, () => RequestCancel(seq!));
        seq = new EngineSequence(handle.Id, request, ch.Writer, request.MaxTokens > 0 ? Math.Min(request.MaxTokens, room) : room, now);
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
            seq.Registration = ct.Register(() => RequestCancel(seq));
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

    /// <summary>
    /// Cancels a request: a waiting one is finished right away; a running one at the end of the step in flight, which frees its
    /// slot for the next admission.
    /// </summary>
    private void RequestCancel(EngineSequence seq)
    {
        seq.CancelRequested = true;
        bool queued;
        lock (_gate)
        {
            queued = _queue.Remove(seq);
            if (!queued)
            {
                Monitor.PulseAll(_gate);
            }
        }

        if (queued)
        {
            Finalize(seq, FinishReason.Cancelled, null, null);
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
        List<EngineSequence> queued;
        lock (_gate)
        {
            queued = [.. _queue.Items];
            _queue.Clear();
            foreach (var s in _active)
            {
                s.CancelRequested = true;
            }

            Monitor.PulseAll(_gate);
        }
        foreach (var s in queued)
        {
            Finalize(s, FinishReason.Cancelled, null, null);
        }
    }

    /// <summary>Cancels all requests, stops the scheduler thread and disposes the decoder (and the speculator).</summary>
    public void Dispose()
    {
        lock (_gate)
        {
            _draining = true;
        }

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

        _speculator?.Dispose();
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
        lock (_gate)
        {
            _idle.TrySetResult();
        }
    }

    private void FailAll(string message)
    {
        for (int i = _active.Count - 1; i >= 0; i--)
        {
            Complete(_active[i], FinishReason.Error, message, null, reset: true);
        }
    }

    /// <summary>Handles cancellations and expired deadlines of running and waiting requests.</summary>
    private void Reap()
    {
        long now = Stopwatch.GetTimestamp();
        List<EngineSequence>? expired = null;
        lock (_gate)
        {
            foreach (var s in _queue.Items)
            {
                if (s.CancelRequested || now > s.DeadlineTicks || now > s.QueueDeadlineTicks)
                {
                    (expired ??= []).Add(s);
                }
            }

            if (expired is not null)
            {
                foreach (var s in expired)
                {
                    _queue.Remove(s);
                }
            }
        }

        if (expired is not null)
        {
            foreach (var s in expired)
            {
                Finalize(s, s.CancelRequested ? FinishReason.Cancelled : FinishReason.Timeout, s.CancelRequested ? null : "timed out waiting for a free slot", null);
            }
        }

        for (int i = _active.Count - 1; i >= 0; i--)
        {
            var s = _active[i];
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
        while (FreeSlotExists())
        {
            EngineSequence? seq;
            lock (_gate)
            {
                seq = _queue.TakeNext();
                if (seq is null)
                {
                    return;
                }

                _transit++;
            }
            try
            {
                AssignSlot(seq);
            }
            finally
            {
                lock (_gate)
                {
                    _transit--;
                }
            }
        }
    }

    private bool FreeSlotExists()
    {
        foreach (var s in _slots)
        {
            if (s.Active is null)
            {
                return true;
            }
        }

        return false;
    }

    private int ReusableTokens(EngineSlot slot, int[] prompt)
    {
        if (!_opt.PrefixCache || slot.Tokens.Count == 0)
        {
            return 0;
        }

        int common = 0;
        int max = Math.Min(slot.Tokens.Count, prompt.Length);
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

    private void AssignSlot(EngineSequence seq)
    {
        EngineSlot? best = null;
        int bestReuse = -1;
        foreach (var s in _slots)
        {
            if (s.Active is not null)
            {
                continue;
            }

            int reuse = ReusableTokens(s, seq.Prompt);
            // Equal reuse (typically none): take an empty slot first, then evict the least recently used one.
            static long Rank(EngineSlot x) => x.Tokens.Count == 0 ? long.MinValue : x.LastUsed;
            bool better = best is null || reuse > bestReuse || (reuse == bestReuse && Rank(s) < Rank(best));
            if (better)
            {
                best = s;
                bestReuse = reuse;
            }
        }
        var slot = best!;
        int keep = Math.Max(0, bestReuse);
        TruncateSlot(slot, keep);
        seq.Slot = slot;
        slot.Active = seq;
        seq.PromptPos = keep;
        seq.Cached = keep;
        seq.Sampler = new RequestSampler(seq.Request.Sampling, seq.Prompt);
        seq.Decoder = new StreamingDecoder(_tokenizer, seq.Request.SkipSpecialTokens);
        seq.Filter = new StopStringFilter(seq.Request.StopStrings);
        seq.StopIds = [.. seq.Request.StopTokenIds];
        seq.Matcher = seq.Request.Constraint?.CreateMatcher();
        _active.Add(seq);
        Metrics.Admitted(seq.Prompt.Length, keep, TimeSpan.FromSeconds((Stopwatch.GetTimestamp() - seq.SubmitTicks) / (double)Stopwatch.Frequency));
    }

    /// <summary>Shortens a slot's context (and the decoder's) to <paramref name="keep"/> tokens.</summary>
    private void TruncateSlot(EngineSlot slot, int keep)
    {
        if (slot.Tokens.Count <= keep)
        {
            return;
        }

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

    private void StepOnce()
    {
        _inputs.Clear();
        _stepSeqs.Clear();
        _stepBaseLength.Clear();
        int budget = _budget;
        int logitsBudget = _decoder.MaxLogitsRows;
        if (_speculator is not null)
        {
            ProposeDrafts();
        }

        // Decode first: the pending token (plus drafts) of every running sequence, rotating the start when the budget cannot
        // cover all of them.
        int count = _active.Count;
        _rr = (_rr + 1) % Math.Max(1, count);
        for (int k = 0; k < count && budget > 0 && logitsBudget > 0; k++)
        {
            var s = _active[(k + _rr) % count];
            if (s.Prefilling)
            {
                continue;
            }

            int n = Math.Min(Math.Min(s.InputLength, budget), logitsBudget);
            AddInput(s, new SequenceInput(s.Slot!.Index, s.Input.AsMemory(0, n), true, n));
            budget -= n;
            logitsBudget -= n;
        }

        // Then chunked prefill of admitted prompts, oldest admission first, with what is left of the token budget.
        foreach (var s in _active)
        {
            if (!s.Prefilling || budget <= 0)
            {
                continue;
            }

            int chunk = Math.Min(Math.Min(_opt.PrefillChunk, s.Prompt.Length - s.PromptPos), budget);
            bool last = s.PromptPos + chunk == s.Prompt.Length;
            if (last && logitsBudget <= 0)
            {
                // The final chunk needs a logits row; stop one token short and finish next step.
                chunk--;
                last = false;
                if (chunk == 0)
                {
                    continue;
                }
            }

            AddInput(s, new SequenceInput(s.Slot!.Index, s.Prompt.AsMemory(s.PromptPos, chunk), last));
            budget -= chunk;
            logitsBudget -= last ? 1 : 0;
        }

        if (_inputs.Count == 0)
        {
            return;
        }

        long t0 = Stopwatch.GetTimestamp();
        var results = PrepareDeviceSampling()
            ? _deviceSampler!.Step(_inputs, _deviceRows, _sampledTokens, _sampledLogprobs)
            : _decoder.Step(_inputs);
        Metrics.Step(_budget - budget, _inputs.Count, Stopwatch.GetElapsedTime(t0));
        ProcessResults(results);
        UpdateKvGauges();
    }

    /// <summary>
    /// Records the fed tokens, samples every sequence that got logits (in parallel for larger batches: sampling, detokenizing
    /// and streaming are per-sequence work), then applies rollbacks and completions on the scheduler thread.
    /// </summary>
    private void ProcessResults(IReadOnlyList<Memory<float>> results)
    {
        int count = _inputs.Count;
        if (_outcomes.Length < count)
        {
            _outcomes = new Outcome[Math.Max(count, _slots.Length)];
        }

        int withLogits = 0;
        for (int i = 0; i < count; i++)
        {
            var s = _stepSeqs[i];
            _outcomes[i] = default;
            if (s.Finished != 0)
            {
                continue;
            }

            var input = _inputs[i];
            var toks = input.Tokens.Span;
            foreach (int t in toks)
            {
                s.Slot!.Tokens.Add(t);
            }

            _queue.Charge(s.Request.Tenant, toks.Length);
            if (s.Prefilling)
            {
                s.PromptPos += toks.Length;
                // The first token follows the whole prompt: it has no drafts, and its context is everything fed so far.
                _stepBaseLength[i] = s.Slot!.Tokens.Count - 1;
            }

            withLogits += input.WantLogits ? 1 : 0;
        }

        if (withLogits >= ParallelSamplingThreshold)
        {
            Parallel.For(0, count, i => SampleStep(i, results[i]));
        }
        else
        {
            for (int i = 0; i < count; i++)
            {
                SampleStep(i, results[i]);
            }
        }

        for (int i = 0; i < count; i++)
        {
            var s = _stepSeqs[i];
            var input = _inputs[i];
            if (s.Finished != 0 || !input.WantLogits)
            {
                continue;
            }

            ref var outcome = ref _outcomes[i];
            if (outcome.Error is not null)
            {
                Complete(s, FinishReason.Error, outcome.Error, null, reset: true);
                continue;
            }

            int rows = input.OutputRows;
            if (rows > 1)
            {
                Metrics.Speculation(rows - 1, outcome.Accepted);
                TruncateSlot(s.Slot!, _stepBaseLength[i] + 1 + outcome.Accepted);
            }

            s.InputLength = 1;
            if (outcome.Ending is { } e)
            {
                Complete(s, e.Reason, null, e.StopSequence, flush: e.Flush);
            }
        }
    }

    /// <summary>
    /// Fills <see cref="_deviceRows"/> for this step's inputs (null for inputs sampled on the CPU) and returns whether any input is
    /// sampled on the device. The device takes inputs with a single logits row (no drafts to verify) whose request needs no masks
    /// (output constraint, minimum length) and no CPU-only sampling feature (see <see cref="RequestSampler.DeviceRow"/>).
    /// </summary>
    private bool PrepareDeviceSampling()
    {
        if (_deviceSampler is null)
        {
            return false;
        }

        int count = _inputs.Count;
        if (_deviceRows.Length < count)
        {
            int n = Math.Max(count, _slots.Length);
            _deviceRows = new SamplingRow?[n];
            _sampledTokens = new int[n];
            _sampledLogprobs = new float[n];
        }

        bool any = false;
        for (int i = 0; i < count; i++)
        {
            var s = _stepSeqs[i];
            _deviceRows[i] = _inputs[i].OutputRows == 1 && s.Finished == 0 && s.Matcher is null && s.Generated >= s.Request.MinTokens
                ? s.Sampler.DeviceRow(s.Generated, s.Request.Logprobs, s.Request.TopLogprobs)
                : null;
            any |= _deviceRows[i] is not null;
        }

        return any;
    }

    /// <summary>
    /// Samples the logits rows of input <paramref name="i"/>: position by position, keeping drafts while they match what the
    /// model samples. Touches only the sequence's own state, so inputs can be processed in parallel.
    /// </summary>
    private void SampleStep(int i, Memory<float> logits)
    {
        var s = _stepSeqs[i];
        var input = _inputs[i];
        if (s.Finished != 0 || !input.WantLogits)
        {
            return;
        }

        int rows = input.OutputRows;
        int vocab = _decoder.VocabSize;
        int baseLength = _stepBaseLength[i];
        ref var outcome = ref _outcomes[i];
        try
        {
            if (_deviceSampler is not null && _deviceRows[i] is not null)
            {
                // Chosen on the device: only the token (and its log-probability, without alternatives) came back.
                int sampled = _sampledTokens[i];
                var lps = s.Request.Logprobs ? new List<TokenLogprob> { new(sampled, TokenText(sampled), _sampledLogprobs[i], []) } : null;
                outcome.Ending = Commit(s, sampled, lps, baseLength + 1);
                return;
            }

            for (int j = 0; j < rows; j++)
            {
                var row = logits.Span.Slice(j * vocab, vocab);
                int tok = SampleToken(s, row);
                bool matchesDraft = j + 1 < rows && s.Input[j + 1] == tok;
                // Context once this token is the next input: everything fed up to row j, plus the token itself.
                outcome.Ending = Emit(s, tok, row, baseLength + j + 1);
                if (outcome.Ending is not null || !matchesDraft)
                {
                    return;
                }

                outcome.Accepted++;
            }
        }
        catch (Exception ex)
        {
            outcome.Error = ex.Message;
        }
    }

    private void AddInput(EngineSequence s, SequenceInput input)
    {
        _inputs.Add(input);
        _stepSeqs.Add(s);
        _stepBaseLength.Add(s.Slot!.Tokens.Count);
    }

    /// <summary>Asks the speculator for drafts of the decoding sequences (when the batch is small enough to profit).</summary>
    private void ProposeDrafts()
    {
        _specBatch.Clear();
        int decoding = 0;
        foreach (var s in _active)
        {
            s.InputLength = 1;
            decoding += s.Prefilling ? 0 : 1;
        }

        if (decoding == 0 || decoding > _opt.SpeculationMaxBatch)
        {
            return;
        }

        int perSequence = Math.Min(_speculator!.MaxDraftTokens, (_decoder.MaxLogitsRows / decoding) - 1);
        foreach (var s in _active)
        {
            var slot = s.Slot!;
            int room = Math.Min(Math.Min(perSequence, _decoder.ContextLength - slot.Tokens.Count - 1), s.MaxTokens - s.Generated - 1);
            if (s.Prefilling || room <= 0 || s.Matcher is not null)
            {
                continue;
            }

            var view = _specViews[slot.Index];
            view.Set(slot.Tokens, s.Pending);
            var r = _specRequests[slot.Index];
            r.Context = view;
            r.MaxTokens = room;
            r.Count = 0;
            _specBatch.Add(r);
        }

        if (_specBatch.Count == 0)
        {
            return;
        }

        _speculator.Propose(_specBatch);
        foreach (var r in _specBatch)
        {
            var s = _slots[r.Slot].Active!;
            int n = Math.Clamp(r.Count, 0, r.MaxTokens);
            s.EnsureInput(1 + n);
            r.Draft.AsSpan(0, n).CopyTo(s.Input.AsSpan(1));
            s.InputLength = 1 + n;
        }
    }

    /// <summary>Applies the request's masks (minimum length, constraint) to <paramref name="logits"/> in place and samples.</summary>
    private int SampleToken(EngineSequence s, Span<float> logits)
    {
        if (s.Request.Logprobs)
        {
            // Log-probabilities are reported from the model's distribution, before masks and penalties.
            s.RawLogits ??= new float[logits.Length];
            logits.CopyTo(s.RawLogits);
        }

        if (s.Generated < s.Request.MinTokens)
        {
            MaskStopTokens(s, logits);
        }

        if (s.Matcher is { } matcher)
        {
            var mask = s.Mask ??= new TokenMask(_decoder.VocabSize);
            matcher.FillAllowed(mask);
            if (s.Request.StopAtEog)
            {
                foreach (int id in _eog)
                {
                    if (matcher.IsComplete)
                    {
                        mask.Allow(id);
                    }
                    else
                    {
                        mask.Disallow(id);
                    }
                }
            }

            if (mask.Count() == 0)
            {
                throw new InvalidOperationException("The output constraint allows no token here (the grammar cannot be continued with this vocabulary).");
            }

            mask.Apply(logits);
        }

        return s.Sampler.Sample(logits);
    }

    /// <summary>
    /// Commits a sampled token: stop checks, streaming, length limits. Returns how the sequence ends, or null when it continues
    /// with <paramref name="tok"/> as its pending input.
    /// </summary>
    /// <param name="s">The sequence.</param>
    /// <param name="tok">The sampled token.</param>
    /// <param name="logits">The (masked) logits row it was sampled from.</param>
    /// <param name="contextAfter">Context length once <paramref name="tok"/> becomes the next input, minus that input.</param>
    private Ending? Emit(EngineSequence s, int tok, ReadOnlySpan<float> logits, int contextAfter)
    {
        List<TokenLogprob>? lps = null;
        if (s.Request.Logprobs)
        {
            var raw = s.RawLogits.AsSpan(0, logits.Length);
            var (lp, top) = RequestSampler.Logprobs(raw, tok, s.Request.TopLogprobs);
            lps = [new TokenLogprob(tok, TokenText(tok), lp, [.. top.Select(t => new TokenAlternative(t.Id, TokenText(t.Id), t.Logprob))])];
        }

        return Commit(s, tok, lps, contextAfter);
    }

    /// <summary>The part of <see cref="Emit"/> after the log-probabilities, shared with tokens sampled on the device.</summary>
    /// <param name="s">The sequence.</param>
    /// <param name="tok">The sampled token.</param>
    /// <param name="lps">The token's log-probability entry when the request reports them.</param>
    /// <param name="contextAfter">Context length once <paramref name="tok"/> becomes the next input, minus that input.</param>
    private Ending? Commit(EngineSequence s, int tok, List<TokenLogprob>? lps, int contextAfter)
    {
        long now = Stopwatch.GetTimestamp();
        s.Sampler.Accept(tok);
        s.Generated++;
        if (s.Generated == 1)
        {
            Metrics.FirstToken(Stopwatch.GetElapsedTime(s.SubmitTicks, now));
        }

        if (s.StopIds.Contains(tok) || (s.Request.StopAtEog && _eog.Contains(tok)))
        {
            s.Generated--;   // the stop token itself is not part of the completion
            return new Ending(FinishReason.Stop, null, true);
        }

        if (s.Matcher is { } matcher && !matcher.Accept(tok))
        {
            throw new InvalidOperationException($"Token {tok} violates the output constraint.");
        }

        Metrics.Token(s.Generated == 1 ? null : Stopwatch.GetElapsedTime(s.LastTokenTicks, now));
        s.LastTokenTicks = now;
        string text = s.Filter.Push(s.Decoder.Push(tok));
        if (text.Length > 0 || lps is not null)
        {
            s.Writer.TryWrite(new GenerationChunk(text, lps));
        }

        if (s.Filter.Stopped)
        {
            return new Ending(FinishReason.Stop, s.Filter.Matched, false);
        }

        if (s.Matcher is { IsTerminated: true })
        {
            return new Ending(FinishReason.Stop, null, true);
        }

        if (s.Generated >= s.MaxTokens || contextAfter + 1 > _decoder.ContextLength)
        {
            return new Ending(FinishReason.Length, null, true);
        }

        s.Pending = tok;
        return null;
    }

    /// <summary>Makes the tokens that would end <paramref name="s"/> unselectable (<see cref="GenerationRequest.MinTokens"/>).</summary>
    private void MaskStopTokens(EngineSequence s, Span<float> logits)
    {
        foreach (int id in s.StopIds)
        {
            if ((uint)id < (uint)logits.Length)
            {
                logits[id] = float.NegativeInfinity;
            }
        }

        if (!s.Request.StopAtEog)
        {
            return;
        }

        foreach (int id in _eog)
        {
            if ((uint)id < (uint)logits.Length)
            {
                logits[id] = float.NegativeInfinity;
            }
        }
    }

    private string TokenText(int id) => _tokenizer.Decode([id]);

    private void UpdateKvGauges()
    {
        int total = 0;
        int active = 0;
        foreach (var slot in _slots)
        {
            total += slot.Tokens.Count;
            active += slot.Active is null ? 0 : slot.Tokens.Count;
        }

        _kvTokens = total;
        _kvActiveTokens = active;
    }

    /// <summary>Ends a running sequence: emits held-back text (when <paramref name="flush"/>), the final chunk, frees the slot.</summary>
    private void Complete(EngineSequence s, FinishReason reason, string? error, string? stopSeq, bool flush = false, bool reset = false)
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
        _speculator?.Release(slot.Index);
        if (reset || reason == FinishReason.Error)
        {
            try
            {
                _decoder.Reset(slot.Index);
            }
            catch
            {
                // The decoder is broken; later requests will fail visibly.
            }

            slot.Tokens.Clear();
        }
        Finalize(s, reason, error, stopSeq);
    }

    private void Finalize(EngineSequence s, FinishReason reason, string? error, string? stopSeq)
    {
        if (Interlocked.Exchange(ref s.Finished, 1) != 0)
        {
            return;
        }

        var usage = new GenerationUsage(s.Prompt.Length, s.Generated, s.Cached);
        s.Writer.TryWrite(new GenerationChunk("", null, reason, usage, error, stopSeq));
        s.Writer.TryComplete();
        s.Registration.Dispose();
        Metrics.Finished(reason, Stopwatch.GetElapsedTime(s.SubmitTicks));
    }
}
