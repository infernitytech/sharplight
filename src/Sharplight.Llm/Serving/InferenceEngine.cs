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
    private readonly List<bool> _stepReplay = [];
    private Outcome[] _outcomes = [];

    // Device sampling (null when the decoder cannot sample on the device): per-input rows of the current step and the tokens /
    // log-probabilities the device chose for them.
    private readonly IDeviceSamplingDecoder? _deviceSampler;
    private SamplingRow?[] _deviceRows = [];
    private int[] _sampledTokens = [];
    // Per input of the step: index of its first logits row in _sampledTokens / _sampledLogprobs.
    private int[] _deviceRowOffset = [];
    private float[] _sampledLogprobs = [];

    // Speculative decoding (null when off or unsupported by the decoder).
    private readonly ISpeculator? _speculator;
    private readonly List<SpeculationRequest> _specBatch = [];
    private int[] _observeTokens = new int[64];
    private readonly SpeculationRequest[] _specRequests;
    private readonly ContextView[] _specViews;

    /// <summary>The live counters; most callers want <see cref="GetMetrics"/>.</summary>
    public EngineMetrics Metrics { get; } = new();
    /// <summary>The tokenizer of the served model, for encoding prompts.</summary>
    public ITokenizer Tokenizer => _tokenizer;
    /// <summary>Number of sequences that can run at the same time.</summary>
    public int SlotCount => _decoder.SlotCount;
    /// <summary>Context length of one sequence: prompt plus generated tokens (at most the whole paged KV cache).</summary>
    public int ContextLength => _decoder.KvBlockSize > 0 ? (int)Math.Min(_decoder.ContextLength, (long)_decoder.KvBlocksTotal * _decoder.KvBlockSize) : _decoder.ContextLength;
    private bool Paged => _decoder.KvBlockSize > 0;

    /// <summary>
    /// Paged KV cache: generated tokens per sequence that admission keeps room for, so that newly admitted sequences do not
    /// immediately preempt each other as they grow (more is idle capacity, less means more preemptions). At most a quarter
    /// of a slot's share of the pool, so that a small pool still runs several sequences.
    /// </summary>
    private const int KvGrowthReserve = 64;
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
            if (decoder.MaxSpeculativeRows > 1 && decoder.MaxLogitsRows > 1 && (!spec.UsesHiddenStates || decoder.HiddenSize > 0))
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
                // Without rollback of rejected drafts (recurrent state compiled without rollback rows) or spare logits rows nothing
                // can be verified; a draft head needs the decoder's hidden states.
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
        long capacity = Paged ? (long)_decoder.KvBlocksTotal * _decoder.KvBlockSize : (long)_slots.Length * _decoder.ContextLength;
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

        if (n >= ContextLength)
        {
            Metrics.Rejected();
            throw new PromptTooLongException(n, ContextLength - 1);
        }
        int room = ContextLength - n;
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
                seq = _queue.PeekNext();
                if (seq is null || !HasKvRoomFor(seq))
                {
                    return;
                }

                _queue.Remove(seq);
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

    /// <summary>
    /// Paged KV cache: whether the prompt's blocks and some room to grow (up to <see cref="KvGrowthReserve"/> tokens) fit next to
    /// what the running sequences still need: the rest of their prompts plus the same growth reserve. Cache blocks of idle
    /// slots count as free; preempted sequences are recomputed before anything new is admitted.
    /// </summary>
    private bool HasKvRoomFor(EngineSequence seq)
    {
        if (!Paged || _active.Count == 0)
        {
            return true;
        }

        long reserve = 0;
        foreach (var a in _active)
        {
            if (a.Replay is not null)
            {
                return false;
            }

            int pending = a.Prefilling ? a.Prompt.Length - a.PromptPos : 0;
            reserve += _decoder.KvBlocksNeeded(a.Slot!.Index, pending + Growth(a));
        }

        long available = _decoder.KvBlocksFree;
        foreach (var slot in _slots)
        {
            if (slot.Active is null && slot.Tokens.Count > 0)
            {
                available += _decoder.KvBlocksReleasable(slot.Index);
            }
        }

        long need = (seq.Prompt.Length + Growth(seq) + _decoder.KvBlockSize - 1) / _decoder.KvBlockSize;
        return available - reserve >= need;

        int Growth(EngineSequence s)
        {
            long share = (long)_decoder.KvBlocksTotal * _decoder.KvBlockSize / (4 * _slots.Length);
            return (int)Math.Clamp(s.MaxTokens - s.Generated, 1, Math.Max(1, Math.Min(KvGrowthReserve, share)));
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
        keep = slot.Tokens.Count;
        if (Paged && _opt.PrefixCache && _decoder.CanTruncate)
        {
            keep = SharePrefix(slot, seq.Prompt, keep);
        }

        seq.AdmitTick = ++_tick;
        seq.AdmitTicks = Stopwatch.GetTimestamp();
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

    /// <summary>
    /// Paged KV cache: when another slot's context (running or idle) starts with more of <paramref name="prompt"/> than the
    /// <paramref name="keep"/> tokens the chosen slot already holds, the slot restarts from that context's full blocks
    /// (shared, not copied). Returns the prompt tokens now in the slot.
    /// </summary>
    private int SharePrefix(EngineSlot slot, int[] prompt, int keep)
    {
        EngineSlot? source = null;
        int best = 0;
        foreach (var other in _slots)
        {
            if (other == slot)
            {
                continue;
            }

            int common = 0;
            int max = Math.Min(other.Tokens.Count, prompt.Length - 1);   // the last prompt token is always computed
            while (common < max && other.Tokens[common] == prompt[common])
            {
                common++;
            }

            if (common > best)
            {
                best = common;
                source = other;
            }
        }

        int bs = _decoder.KvBlockSize;
        if (source is null || best / bs * bs <= keep)
        {
            return keep;
        }

        TruncateSlot(slot, 0);
        int shared = _decoder.ShareKvPrefix(slot.Index, source.Index, best / bs * bs);
        for (int i = 0; i < shared; i++)
        {
            slot.Tokens.Add(prompt[i]);
        }

        Metrics.SharedPrefix(shared);
        return shared;
    }

    /// <summary>
    /// Shortens a slot's context (and the decoder's) to <paramref name="keep"/> tokens; a paged decoder may keep fewer (a cut
    /// inside a shared block), so the slot's tokens follow the decoder's length.
    /// </summary>
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
            keep = _decoder.Length(slot.Index);
        }

        slot.Tokens.RemoveRange(keep, slot.Tokens.Count - keep);
    }

    /// <summary>What is left of a step's capacity while its batch is being assembled.</summary>
    private struct StepBudget
    {
        /// <summary>Tokens the batch may still hold.</summary>
        public int Tokens;

        /// <summary>Logits rows the decoder may still return.</summary>
        public int LogitsRows;

        /// <summary>Free KV blocks (only meaningful with a paged cache).</summary>
        public int KvBlocks;
    }

    private void StepOnce()
    {
        _inputs.Clear();
        _stepSeqs.Clear();
        _stepBaseLength.Clear();
        _stepReplay.Clear();
        if (_speculator is not null)
        {
            ProposeDrafts();
        }

        var budget = new StepBudget { Tokens = _budget, LogitsRows = _decoder.MaxLogitsRows };
        budget.KvBlocks = EnsureKvBlocksForDecode();
        ScheduleDecode(ref budget);
        ScheduleReplay(ref budget);
        ScheduleChunkedPrefill(ref budget);

        if (_inputs.Count == 0)
        {
            // Paged cache: every running sequence waits for blocks the others hold; the newest gives its blocks back.
            if (Paged && _active.Count > 1)
            {
                PreemptNewest();
            }

            return;
        }

        // A draft head reads the target's hidden states, while speculation is on (small batches).
        bool observe = _speculator is { UsesHiddenStates: true } && _active.Count <= _opt.SpeculationMaxBatch;
        if (_speculator is { UsesHiddenStates: true })
        {
            _decoder.CaptureHiddenStates = observe;
        }

        long t0 = Stopwatch.GetTimestamp();
        var results = PrepareDeviceSampling()
            ? _deviceSampler!.Step(_inputs, _deviceRows, _sampledTokens, _sampledLogprobs)
            : _decoder.Step(_inputs);
        Metrics.Step(_budget - budget.Tokens, _inputs.Count, Stopwatch.GetElapsedTime(t0));
        ProcessResults(results);
        if (observe)
        {
            ObserveHiddenStates();
        }

        UpdateKvGauges();
    }

    /// <summary>
    /// Hands the speculator the rows each input kept (all fed rows, or for a verified input the pending token plus the accepted
    /// drafts) with their hidden states and the token that follows each row.
    /// </summary>
    private void ObserveHiddenStates()
    {
        for (int i = 0; i < _inputs.Count; i++)
        {
            var s = _stepSeqs[i];
            if (s.Finished != 0 || s.Slot is not { } slot)
            {
                continue;
            }

            var input = _inputs[i];
            int rows = input.OutputRows;
            int kept = rows > 1 ? 1 + _outcomes[i].Accepted : input.Tokens.Length;
            int start = slot.Tokens.Count - kept;
            if (kept <= 0 || start < 0)
            {
                continue;
            }

            if (_observeTokens.Length < kept)
            {
                _observeTokens = new int[Math.Max(kept, 2 * _observeTokens.Length)];
            }

            for (int j = 0; j < kept; j++)
            {
                int p = start + j + 1;
                _observeTokens[j] = p < slot.Tokens.Count ? slot.Tokens[p] : NextInputToken(s);
            }

            _speculator!.Observe(slot.Index, start, _observeTokens.AsSpan(0, kept), _decoder.HiddenStates(i, kept).Span);
        }
    }

    /// <summary>The token a sequence feeds next: replay, prompt, or the pending sampled token.</summary>
    private static int NextInputToken(EngineSequence s)
        => s.Replay is { } replay ? replay[s.ReplayPos] : s.Prefilling ? s.Prompt[s.PromptPos] : s.Pending;

    /// <summary>
    /// Decode first: the pending token (plus drafts) of every running sequence, rotating the start when the budget cannot cover
    /// all of them.
    /// </summary>
    private void ScheduleDecode(ref StepBudget budget)
    {
        int count = _active.Count;
        _rr = (_rr + 1) % Math.Max(1, count);
        for (int k = 0; k < count && budget.Tokens > 0 && budget.LogitsRows > 0; k++)
        {
            var s = _active[(k + _rr) % count];
            if (s.Prefilling || s.Replay is not null)
            {
                continue;
            }

            int n = Math.Min(Math.Min(s.InputLength, budget.Tokens), budget.LogitsRows);
            int blocks = _decoder.KvBlocksNeeded(s.Slot!.Index, n);
            if (blocks > budget.KvBlocks)
            {
                continue;
            }

            budget.KvBlocks -= blocks;
            AddInput(s, new SequenceInput(s.Slot!.Index, s.Input.AsMemory(0, n), true, n, Speculative: n > 1));
            budget.Tokens -= n;
            budget.LogitsRows -= n;
        }
    }

    /// <summary>Preempted sequences recompute their context (no logits) before new prompts.</summary>
    private void ScheduleReplay(ref StepBudget budget)
    {
        foreach (var s in _active)
        {
            if (s.Replay is not { } replay || budget.Tokens <= 0)
            {
                continue;
            }

            int chunk = Math.Min(Math.Min(Math.Min(_opt.PrefillChunk, replay.Length - s.ReplayPos), budget.Tokens), KvRoom(s.Slot!, budget.KvBlocks));
            if (chunk <= 0)
            {
                continue;
            }

            budget.KvBlocks -= _decoder.KvBlocksNeeded(s.Slot!.Index, chunk);
            AddInput(s, new SequenceInput(s.Slot!.Index, replay.AsMemory(s.ReplayPos, chunk), false), replay: true);
            budget.Tokens -= chunk;
        }
    }

    /// <summary>Chunked prefill of admitted prompts, oldest admission first, with what is left of the token budget.</summary>
    private void ScheduleChunkedPrefill(ref StepBudget budget)
    {
        foreach (var s in _active)
        {
            if (!s.Prefilling || s.Replay is not null || budget.Tokens <= 0)
            {
                continue;
            }

            int chunk = Math.Min(Math.Min(Math.Min(_opt.PrefillChunk, s.Prompt.Length - s.PromptPos), budget.Tokens), KvRoom(s.Slot!, budget.KvBlocks));
            if (chunk <= 0)
            {
                continue;
            }

            bool last = s.PromptPos + chunk == s.Prompt.Length;
            if (last && budget.LogitsRows <= 0)
            {
                // The final chunk needs a logits row; stop one token short and finish next step.
                chunk--;
                last = false;
                if (chunk == 0)
                {
                    continue;
                }
            }

            budget.KvBlocks -= _decoder.KvBlocksNeeded(s.Slot!.Index, chunk);
            AddInput(s, new SequenceInput(s.Slot!.Index, s.Prompt.AsMemory(s.PromptPos, chunk), last));
            budget.Tokens -= chunk;
            budget.LogitsRows -= last ? 1 : 0;
        }
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

        int withLogits = RecordFedTokens();
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

        ApplyOutcomes();
    }

    /// <summary>Appends the fed tokens to their slots, advances replay and prefill positions and returns how many inputs want logits.</summary>
    private int RecordFedTokens()
    {
        int withLogits = 0;
        for (int i = 0; i < _inputs.Count; i++)
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

            if (_stepReplay[i])
            {
                s.ReplayPos += toks.Length;
                if (s.ReplayPos == s.Replay!.Length)
                {
                    s.Replay = null;
                }

                continue;
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

        return withLogits;
    }

    /// <summary>Applies sampling errors, speculation rollbacks and completions on the scheduler thread.</summary>
    private void ApplyOutcomes()
    {
        for (int i = 0; i < _inputs.Count; i++)
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
                s.DraftTokens += rows - 1;
                s.AcceptedDraftTokens += outcome.Accepted;
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
    /// sampled on the device. The device takes inputs whose request needs no masks (output constraint, minimum length) and no
    /// CPU-only sampling feature (see <see cref="RequestSampler.DeviceRow"/>), including speculative ones: every verification row is
    /// sampled there (row j with the random-stream counter advanced by j, exactly as decoding without drafts would), and only the
    /// tokens come back; the accepted prefix is then the drafts that equal them.
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
            _deviceRowOffset = new int[n];
        }

        if (_sampledTokens.Length < _decoder.MaxLogitsRows)
        {
            _sampledTokens = new int[_decoder.MaxLogitsRows];
            _sampledLogprobs = new float[_decoder.MaxLogitsRows];
        }

        bool any = false;
        for (int i = 0, row = 0; i < count; i++)
        {
            var s = _stepSeqs[i];
            _deviceRowOffset[i] = row;
            row += _inputs[i].OutputRows;
            _deviceRows[i] = _inputs[i].OutputRows >= 1 && s.Finished == 0 && s.Matcher is null && s.Generated >= s.Request.MinTokens
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
                // Chosen on the device: only the tokens (and their log-probabilities, without alternatives) came back, one per row;
                // drafts are kept while they equal the row before them.
                int first = _deviceRowOffset[i];
                for (int j = 0; j < rows; j++)
                {
                    int sampled = _sampledTokens[first + j];
                    var lps = s.Request.Logprobs ? new List<TokenLogprob> { new(sampled, TokenText(sampled), _sampledLogprobs[first + j], []) } : null;
                    bool matchesDraft = j + 1 < rows && s.Input[j + 1] == sampled;
                    outcome.Ending = Commit(s, sampled, lps, baseLength + j + 1);
                    if (outcome.Ending is not null || !matchesDraft)
                    {
                        return;
                    }

                    outcome.Accepted++;
                }

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

    private void AddInput(EngineSequence s, SequenceInput input, bool replay = false)
    {
        _inputs.Add(input);
        _stepSeqs.Add(s);
        _stepBaseLength.Add(s.Slot!.Tokens.Count);
        _stepReplay.Add(replay);
    }

    /// <summary>Tokens a slot can still append with <paramref name="freeBlocks"/> more blocks (unbounded without a paged cache).</summary>
    private int KvRoom(EngineSlot slot, int freeBlocks)
    {
        if (!Paged)
        {
            return int.MaxValue;
        }

        int bs = _decoder.KvBlockSize;
        int length = _decoder.Length(slot.Index);
        long room = (long)(length + bs - 1) / bs * bs - length + (long)freeBlocks * bs;
        return (int)Math.Min(int.MaxValue, room);
    }

    /// <summary>
    /// Paged KV cache: frees blocks until every decoding sequence can append its next tokens, first from idle slots' cached
    /// contexts (least recently used first), then by preempting the newest running sequences (their context is recomputed when
    /// blocks are free again). Idle contexts also give way to the next chunks of prompts and recomputations. Returns the free
    /// blocks (<see cref="int.MaxValue"/> without a paged cache).
    /// </summary>
    private int EnsureKvBlocksForDecode()
    {
        if (!Paged)
        {
            return int.MaxValue;
        }

        while (true)
        {
            int decode = 0;
            int chunks = 0;
            foreach (var s in _active)
            {
                int slot = s.Slot!.Index;
                if (s.Replay is { } replay)
                {
                    chunks += _decoder.KvBlocksNeeded(slot, Math.Min(_opt.PrefillChunk, replay.Length - s.ReplayPos));
                }
                else if (s.Prefilling)
                {
                    chunks += _decoder.KvBlocksNeeded(slot, Math.Min(_opt.PrefillChunk, s.Prompt.Length - s.PromptPos));
                }
                else
                {
                    decode += _decoder.KvBlocksNeeded(slot, s.InputLength);
                }
            }

            int free = _decoder.KvBlocksFree;
            if (decode > free)
            {
                if (!EvictIdleSlot() && !PreemptNewest())
                {
                    return free;
                }
            }
            else if (decode + chunks <= free || !EvictIdleSlot())
            {
                return free;
            }
        }
    }

    /// <summary>Drops the cached context of the least recently used idle slot that holds unshared blocks.</summary>
    private bool EvictIdleSlot()
    {
        EngineSlot? victim = null;
        foreach (var slot in _slots)
        {
            if (slot.Active is null && slot.Tokens.Count > 0 && _decoder.KvBlocksReleasable(slot.Index) > 0 && (victim is null || slot.LastUsed < victim.LastUsed))
            {
                victim = slot;
            }
        }

        if (victim is null)
        {
            return false;
        }

        TruncateSlot(victim, 0);
        return true;
    }

    /// <summary>
    /// Takes the KV blocks of the most recently admitted running sequence that holds unshared blocks (never the only running one);
    /// it keeps its slot and recomputes its context later.
    /// </summary>
    private bool PreemptNewest()
    {
        EngineSequence? victim = null;
        int candidates = 0;
        foreach (var s in _active)
        {
            if (_decoder.KvBlocksReleasable(s.Slot!.Index) == 0)
            {
                continue;
            }

            candidates++;
            if (victim is null || s.AdmitTick > victim.AdmitTick)
            {
                victim = s;
            }
        }

        if (victim is null || (candidates == 1 && _active.Count == 1))
        {
            return false;
        }

        var slot = victim.Slot!;
        // A sequence preempted again while recomputing restarts the same replay (its slot holds only a prefix of it).
        victim.Replay = victim.Replay ?? (slot.Tokens.Count > 0 ? [.. slot.Tokens] : null);
        victim.ReplayPos = 0;
        victim.InputLength = 1;
        _speculator?.Release(slot.Index);
        _decoder.Reset(slot.Index);
        slot.Tokens.Clear();
        Metrics.Preempted();
        return true;
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

        int perSequence = Math.Min(Math.Min(_speculator!.MaxDraftTokens, (_decoder.MaxLogitsRows / decoding) - 1), _decoder.MaxSpeculativeRows - 1);
        perSequence = Math.Min(perSequence, Math.Max(1, _opt.SpeculationMaxDraftRows / decoding));
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
            s.FirstTokenTicks = now;
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

        if (s.Generated >= s.MaxTokens || contextAfter + 1 > ContextLength)
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
        var timings = Timings(s, Stopwatch.GetTimestamp());
        s.Writer.TryWrite(new GenerationChunk("", null, reason, usage, error, stopSeq, timings));
        s.Writer.TryComplete();
        s.Registration.Dispose();
        Metrics.Finished(reason, timings.Total);
        if (_opt.RequestFinished is { } finished)
        {
            try
            {
                finished(new FinishedRequest(s.Id, s.Request.Tenant, reason, usage, timings, error));
            }
            catch
            {
                // Observers must not break the scheduler.
            }
        }
    }

    /// <summary>The timings of a request finishing at <paramref name="now"/>; phases it never reached are zero.</summary>
    private static GenerationTimings Timings(EngineSequence s, long now)
    {
        long admitted = s.AdmitTicks != 0 ? s.AdmitTicks : now;
        long first = s.FirstTokenTicks != 0 ? s.FirstTokenTicks : now;
        return new GenerationTimings(
            Stopwatch.GetElapsedTime(s.SubmitTicks, admitted),
            Stopwatch.GetElapsedTime(admitted, first),
            s.FirstTokenTicks != 0 ? Stopwatch.GetElapsedTime(first, now) : TimeSpan.Zero,
            Stopwatch.GetElapsedTime(s.SubmitTicks, now),
            s.DraftTokens,
            s.AcceptedDraftTokens);
    }
}
