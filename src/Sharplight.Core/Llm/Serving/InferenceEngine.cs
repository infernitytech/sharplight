using System.Diagnostics;
using System.Runtime.InteropServices;
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
/// Masked-diffusion models (a <see cref="DiffusionBatchDecoder"/>) are scheduled the same way, but a running request is a whole sequence
/// (prompt plus a span of <see cref="GenerationRequest.MaxTokens"/> mask tokens) that every step denoises once, together with the other
/// running sequences, on the schedule of <see cref="GenerationRequest.Diffusion"/>; the tokens are streamed as their blocks become final
/// (the whole span at once without blocks). The reveal rule picks the most likely tokens, so sampling settings, log-probabilities and
/// minimum lengths do not apply; output constraints are rejected.
/// </para>
/// <para>
/// Thread safety: all members may be called from any thread. The engine owns the decoder and disposes it. Idle slots keep their
/// context, so a later prompt that shares a prefix with it skips recomputing the shared part (<see cref="EngineOptions.PrefixCache"/>).
/// </para>
/// </summary>
/// <example>
/// <code>
/// using var model = LlmModel.Load(loader, path, loadOptions);
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
    private readonly DiffusionBatchDecoder? _diffusion;
    private static readonly DiffusionOptions DefaultDiffusion = new();
    private readonly List<DiffusionSequence> _diffusionBatch = [];
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
    private readonly int _minBudget;
    private int _load;

    /// <summary>Whether the current step's budget took the smaller prompt chunks (<see cref="PromptRowsPerSequence"/>; for the trace).</summary>
    private bool _smallChunks;
    private long _loadTimestamp;
    /// <summary>A step this long after the previous one sees the load of its own sequences only (see <see cref="StepTokens"/>).</summary>
    private static readonly TimeSpan LoadHoldTime = TimeSpan.FromSeconds(1);
    private readonly StepProfiler? _profiler = StepProfiler.Create();
    /// <summary>
    /// Whether committed tokens are decoded and streamed while the next step runs on the device (an asynchronous decoder, see
    /// <see cref="FlushEmits()"/>), instead of before it is scheduled.
    /// </summary>
    private readonly bool _deferEmits;
    /// <summary>Sequences with deferred tokens (<see cref="EngineSequence.PendingEmits"/>).</summary>
    private readonly List<EngineSequence> _emitPending = [];
    private readonly int _prefillChunk;
    private long _nextId, _tick;
    private int _rr, _transit;
    private bool _stop, _draining;
    private int _disposed;
    private bool _admissionPaused;   // tests: queued requests wait until ResumeAdmission (running ones keep stepping)
    private volatile int _kvTokens, _kvActiveTokens, _kvBlocksCached;
    private long _kvBlocksEvicted;

    // Prefix caching by block content (paged decoders): the block hashes a prefilling sequence will compute (others with the same
    // blocks wait for them instead of computing them twice) and, for models with recurrent state, the block boundaries whose state
    // should be snapshotted (shared prefixes seen at admission and the ends of prompts).
    private readonly Dictionary<UInt128, EngineSequence> _computing = [];
    private readonly HashSet<UInt128> _wantSnapshot = [];     // ends of prefixes shared with other contexts
    private readonly HashSet<UInt128> _wantPromptEnd = [];    // ends of prompts (a conversation's next turn)
    private readonly HashSet<UInt128> _takeSnapshot = [];     // boundaries a prompt's chunk was cut at to keep the state there
    // Prompt-end snapshots taken and later restored (how often a prompt is continued by a later one), counted apart for prompts that
    // continued an earlier prompt themselves (index 1: a conversation's later turns) and others (index 0); the keys not restored yet
    // with the class of the prompt that left them.
    private readonly Dictionary<UInt128, int> _promptEndKeys = [];
    private readonly long[] _promptEndSaved = new long[2], _promptEndRestored = new long[2];
    // What steps and state copies cost, learnt from the steps run (prices the prefix cache's waits, recomputations and snapshots).
    private readonly StepCostModel _cost;
    private readonly TileStepCost? _tiles;
    /// <summary>The admission policy at KV saturation (<see cref="EngineOptions.KvAdmission"/>, or the process override).</summary>
    private readonly KvAdmissionPolicy _kvAdmission;
    /// <summary>Prompt rows the current step held back for a later one (<see cref="HoldRowsPastTile"/>; scheduler trace).</summary>
    private int _heldRows;

    // Image and audio prompts: encoder outputs by content hash, and the contents the next encode stage runs (scratch).
    private readonly MediaEmbeddingCache _mediaCache;
    private readonly List<(EngineSequence Sequence, int Index)> _mediaPending = [];

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
    /// <summary>How long preparing the decode steps took when the engine started (<see cref="EngineOptions.PrepareDecodeSteps"/>).</summary>
    public TimeSpan PrepareTime { get; }
    /// <summary>The tokenizer of the served model, for encoding prompts.</summary>
    public ITokenizer Tokenizer => _tokenizer;
    /// <summary>Number of sequences that can run at the same time.</summary>
    public int SlotCount => _decoder.SlotCount;
    /// <summary>Context length of one sequence: prompt plus generated tokens (at most the whole paged KV cache).</summary>
    public int ContextLength => _decoder.KvBlockSize > 0 ? (int)Math.Min(_decoder.ContextLength, (long)_decoder.KvBlocksTotal * _decoder.KvBlockSize) : _decoder.ContextLength;
    private bool Paged => _decoder.KvBlockSize > 0;
    /// <summary>Prefix caching by block content: a paged decoder that publishes and attaches blocks, and the prefix cache enabled.</summary>
    private bool CachesBlocks => _opt.PrefixCache && _decoder.CachesKvBlocks;
    /// <summary>Prefix caching of a model with recurrent state (blocks attach together with state snapshots).</summary>
    private bool SnapshotsState => CachesBlocks && _decoder.KvStateSnapshots > 0;

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
        _diffusion = decoder as DiffusionBatchDecoder;
        _deviceSampler = decoder is IDeviceSamplingDecoder { SamplesOnDevice: true } d ? d : null;
        _tokenizer = tokenizer;
        _opt = options ?? new();
        _slots = [.. Enumerable.Range(0, decoder.SlotCount).Select(i => new EngineSlot(i))];
        _queue = new RequestQueue(_opt.FairShareHalfLife);
        _eog = [.. tokenizer.EndOfGenerationIds];
        _budget = Math.Max(1, _opt.MaxBatchTokens > 0 ? Math.Min(_opt.MaxBatchTokens, decoder.MaxBatchTokens) : decoder.MaxBatchTokens);
        _minBudget = _opt.MinBatchTokens > 0 ? Math.Min(_opt.MinBatchTokens, _budget) : _budget;
        _prefillChunk = _opt.PrefillChunk > 0 ? _opt.PrefillChunk : _budget;
        _cost = _opt.StepCost ?? new StepCostModel();
        _kvAdmission = DiagnosticSwitches.KvAdmission ?? _opt.KvAdmission;
        _tiles = (DiagnosticSwitches.TileAwarePrompts ?? _opt.TileAwarePrompts) && _opt.BatchTokensGranule > 1 ? new TileStepCost(_opt.BatchTokensGranule) : null;
        _mediaCache = new MediaEmbeddingCache(Math.Max(0, _opt.MediaCacheBytes));
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

        if (_opt.PrepareDecodeSteps && decoder.BatchesSequences && _diffusion is null)
        {
            long t0 = Stopwatch.GetTimestamp();
            int shapes = decoder.PrepareDecodeSteps(_deviceSampler is not null);
            PrepareTime = Stopwatch.GetElapsedTime(t0);
            if (DiagnosticSwitches.SchedulerTrace)
            {
                Console.Error.WriteLine($"[sched] prepared {shapes} decode step shapes in {PrepareTime.TotalMilliseconds:F0} ms");
            }
        }

        if (decoder is MultiSequenceDecoder multi)
        {
            multi.PadsStepShapes = _opt.StepShapeBuckets;
        }

        if (decoder is MultiSequenceDecoder { RunsAsynchronously: true } asynchronous && DiagnosticSwitches.DeferEmits)
        {
            _deferEmits = true;
            asynchronous.WhileRunning = FlushEmits;
        }

        _thread = new Thread(Run) { Name = "sharplight-scheduler", IsBackground = true };
        _thread.Start();
    }

    /// <summary>
    /// Tests: holds submitted requests in the queue (running sequences keep stepping) until <see cref="ResumeAdmission"/>, so that requests
    /// submitted one after another are admitted in one pass, as if they had arrived together.
    /// </summary>
    internal void PauseAdmission()
    {
        lock (_gate)
        {
            _admissionPaused = true;
        }
    }

    /// <summary>Tests: admits the requests queued since <see cref="PauseAdmission"/> (in one admission pass).</summary>
    internal void ResumeAdmission()
    {
        lock (_gate)
        {
            _admissionPaused = false;
            Monitor.PulseAll(_gate);
        }
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
        return Metrics.Snapshot(q, a, _slots.Length) with
        {
            KvCapacityTokens = capacity, KvTokens = _kvTokens, KvActiveTokens = _kvActiveTokens,
            KvBlocksCached = _kvBlocksCached, KvBlocksEvicted = Interlocked.Read(ref _kvBlocksEvicted),
        };
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
        var layout = MediaLayout(request);
        int room = ContextLength - n;
        int maxTokens = request.MaxTokens > 0 ? Math.Min(request.MaxTokens, room) : room;
        if (_diffusion is not null)
        {
            maxTokens = DiffusionSpan(request, maxTokens);
        }

        var ch = Channel.CreateUnbounded<GenerationChunk>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = false });
        long now = Stopwatch.GetTimestamp();
        EngineSequence? seq = null;
        var handle = new GenerationHandle(Interlocked.Increment(ref _nextId), ch, () => RequestCancel(seq!));
        seq = new EngineSequence(handle.Id, request, ch.Writer, maxTokens, now, layout);
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

    /// <summary>
    /// The layout of a request with media rows, prompt positions or media weights (validated against the decoder), or null for a plain
    /// token prompt.
    /// </summary>
    /// <exception cref="EngineException">The request's media or positions do not fit the prompt or the decoder (code <c>invalid_request</c>).</exception>
    private PromptLayout? MediaLayout(GenerationRequest request)
    {
        if (request.PromptMedia.Count == 0 && request.PromptPositions is null && request.PromptMediaWeights is null)
        {
            return null;
        }

        int n = request.PromptTokens.Length;
        string? error = null;
        int end = 0;
        foreach (var m in request.PromptMedia.OrderBy(m => m.Start))
        {
            if (m.Content is null || m.Start < end || m.End > n)
            {
                error = "Prompt media must lie inside the prompt without overlapping.";
                break;
            }

            if (m.Content.Encoder.EmbeddingWidth != _decoder.EmbeddingWidth)
            {
                error = _decoder.EmbeddingWidth == 0
                    ? "This model is served without image or audio input."
                    : $"Media rows of width {m.Content.Encoder.EmbeddingWidth} do not fit the model's embedding rows of width {_decoder.EmbeddingWidth}.";
                break;
            }

            end = m.End;
        }

        if (error is null && request.PromptPositions is { } positions && positions.Length != n * _decoder.PositionAxes)
        {
            error = $"The prompt positions hold {positions.Length} values; the model takes {_decoder.PositionAxes} per prompt token ({n * _decoder.PositionAxes}).";
        }

        if (error is null && request.PromptMediaWeights is { } weights && weights.Length != n)
        {
            error = $"The prompt media weights hold {weights.Length} flags for {n} prompt tokens.";
        }

        if (error is not null)
        {
            Metrics.Rejected();
            throw new EngineException(error, "invalid_request");
        }

        return new PromptLayout(request, _decoder.PositionAxes, _decoder.TextPositionAxes);
    }

    /// <summary>
    /// The span a diffusion request generates: <paramref name="maxTokens"/> (already capped by the context), rounded down to whole blocks.
    /// </summary>
    /// <exception cref="EngineException">The request has an output constraint or its schedule does not fit the span (code <c>invalid_request</c>).</exception>
    private int DiffusionSpan(GenerationRequest request, int maxTokens)
    {
        var d = request.Diffusion ?? DefaultDiffusion;
        int span = d.BlockLength > 0 ? maxTokens / d.BlockLength * d.BlockLength : maxTokens;
        string? error = request.Constraint is not null
            ? "Output constraints are not supported for masked-diffusion models."
            : !DiffusionSequence.FitsSchedule(span, d)
                ? $"The diffusion schedule (steps {d.Steps}, block length {d.BlockLength}) does not fit a span of {span} tokens: the span must be a "
                    + "multiple of the block length and the steps a multiple of the block count."
                : null;
        if (error is not null)
        {
            Metrics.Rejected();
            throw new EngineException(error, "invalid_request");
        }

        return span;
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
    /// <param name="timeout">How long to wait for running requests.</param>
    /// <param name="cancellationToken">Stops waiting; admission stays closed.</param>
    /// <returns>True when every request finished in time.</returns>
    public async Task<bool> DrainAsync(TimeSpan timeout, CancellationToken cancellationToken = default)
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
        var done = await Task.WhenAny(_idle.Task, Task.Delay(timeout, cancellationToken)).ConfigureAwait(false) == _idle.Task;
        cancellationToken.ThrowIfCancellationRequested();
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

    /// <summary>
    /// Cancels all requests, stops the scheduler thread and disposes the decoder (and the speculator). Only the first call does this; later
    /// ones (also concurrent ones) return at once, so the owner must not dispose from two threads when it needs to know the memory is freed.
    /// </summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

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
        _profiler?.Dispose();
    }

    // ---- scheduler --------------------------------------------------------------------------------------------------

    private void Run()
    {
        while (true)
        {
            lock (_gate)
            {
                while ((_queue.Count == 0 || _admissionPaused) && _active.Count == 0 && !_stop)
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
                long admitStart = Stopwatch.GetTimestamp();
                Reap();
                if (!Volatile.Read(ref _admissionPaused))
                {
                    Admit();
                }

                if (_profiler is not null)
                {
                    _profiler.AdmitMs += Stopwatch.GetElapsedTime(admitStart).TotalMilliseconds;
                }

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
    /// Paged KV cache: whether the prompt's blocks and some room to grow (up to <see cref="KvGrowthReserve"/> tokens; one token under
    /// <see cref="KvAdmissionPolicy.Preempt"/>) fit next to what the running sequences still need: the rest of their prompts plus the
    /// same growth reserve. Cache blocks of idle slots count as free; preempted sequences are recomputed before anything new is admitted.
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

            int pending = a.Prefilling && a.Leader is null ? a.Prompt.Length - a.PromptPos : 0;
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
        if (FindLeader(seq) is not null)
        {
            // Forked from the same prompt: its blocks are shared; only growth and the copy of the shared last block are new.
            need = (Growth(seq) + _decoder.KvBlockSize - 1) / _decoder.KvBlockSize + 1;
        }
        else if (CachesBlocks && !seq.Request.BypassPrefixCache)
        {
            // Cached blocks other sequences hold are shared, not taken from the free ones (cached free blocks are counted as free).
            _decoder.FindKvBlocks(CollectionsMarshal.AsSpan(PromptHashes(seq)), out int inUse);
            need -= inUse;
        }

        return available - reserve >= need;

        int Growth(EngineSequence s)
        {
            if (_kvAdmission == KvAdmissionPolicy.Preempt)
            {
                // The next token only: sequences that outgrow the pool later are preempted (EnsureKvBlocksForDecode).
                return 1;
            }

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

    private int ReusableTokens(EngineSlot slot, EngineSequence seq)
    {
        var r = seq.Request;
        if (!_opt.PrefixCache || r.BypassPrefixCache || slot.Tokens.Count == 0 || !_decoder.ReusesPrefixes || slot.Partition != r.CachePartition || slot.Salt != r.CacheSalt)
        {
            return 0;
        }

        var prompt = seq.Prompt;
        int common = 0;
        int max = Math.Min(Math.Min(slot.Tokens.Count, prompt.Length), SameContentUntil(slot.Content, seq.Content));
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

            int reuse = ReusableTokens(s, seq);
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
        if (FindLeader(seq) is { } leader)
        {
            AssignFollower(seq, slot, leader);
            return;
        }

        int keep = Math.Max(0, bestReuse);
        // Recurrent state: the prefix shared with known contexts (including the chosen slot's before it is cut) gets a state snapshot.
        int sharedBlocks = SnapshotsState ? SharedBlocks(PromptHashes(seq)) : 0;
        TruncateSlot(slot, keep);
        slot.Partition = seq.Request.CachePartition;
        slot.Salt = seq.Request.CacheSalt;
        slot.Content = seq.Content;
        keep = slot.Tokens.Count;
        if (CachesBlocks)
        {
            slot.Seed = KvBlockHash.Seed(seq.Request.CachePartition, seq.Request.CacheSalt);
            var hashes = PromptHashes(seq);
            if (!seq.Request.BypassPrefixCache)
            {
                // Cached blocks reaching further than the slot's own context replace it.
                if (_decoder.FindKvBlocks(CollectionsMarshal.AsSpan(hashes), out _) * _decoder.KvBlockSize > keep)
                {
                    TruncateSlot(slot, 0);
                    keep = 0;
                }

                keep += AttachCached(seq, slot, seq.Prompt, hashes, keep);
            }

            RegisterComputing(seq, keep);
            if (SnapshotsState)
            {
                WantSnapshots(seq, sharedBlocks, keep);
            }
        }

        Activate(seq, slot, keep);
    }

    /// <summary>Gives an admitted sequence its slot and per-request state; <paramref name="keep"/> prompt tokens are already in the slot.</summary>
    private void Activate(EngineSequence seq, EngineSlot slot, int keep)
    {
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
        _speculator?.SetLayout(slot.Index, seq.Layout);
        _active.Add(seq);
        Metrics.Admitted(seq.Prompt.Length, keep, TimeSpan.FromSeconds((Stopwatch.GetTimestamp() - seq.SubmitTicks) / (double)Stopwatch.Frequency));
    }

    /// <summary>
    /// Another sampling of a prompt that a running sequence is still prefilling (n &gt; 1 choices, or the same prompt sent twice at once):
    /// a sequence in the same cache namespace, with the same tokens and content, that is not itself waiting for another.
    /// </summary>
    private EngineSequence? FindLeader(EngineSequence seq)
    {
        var r = seq.Request;
        if (!Paged || !_opt.PrefixCache || _diffusion is not null || r.BypassPrefixCache)
        {
            return null;
        }

        foreach (var a in _active)
        {
            if (a.Leader is null && a.Prefilling && a.Replay is null && !a.Request.BypassPrefixCache && a.Request.CachePartition == r.CachePartition
                && a.Request.CacheSalt == r.CacheSalt && a.Prompt.AsSpan().SequenceEqual(seq.Prompt) && SameContentUntil(a.Content, seq.Content) == int.MaxValue)
            {
                return a;
            }
        }

        return null;
    }

    /// <summary>Admits a sequence that will fork from <paramref name="leader"/> once the leader's prompt is computed (it computes nothing itself).</summary>
    private void AssignFollower(EngineSequence seq, EngineSlot slot, EngineSequence leader)
    {
        TruncateSlot(slot, 0);
        slot.Partition = seq.Request.CachePartition;
        slot.Salt = seq.Request.CacheSalt;
        slot.Content = seq.Content;
        if (CachesBlocks)
        {
            slot.Seed = KvBlockHash.Seed(seq.Request.CachePartition, seq.Request.CacheSalt);
            PromptHashes(seq);
        }

        seq.Leader = leader;
        (leader.Followers ??= []).Add(seq);
        Activate(seq, slot, 0);
    }

    /// <summary>The followers of <paramref name="leader"/> stop waiting for it (it ended or lost its context) and compute their prompts.</summary>
    private void ReleaseFollowers(EngineSequence leader)
    {
        if (leader.Followers is not { } followers)
        {
            return;
        }

        leader.Followers = null;
        foreach (var f in followers)
        {
            f.Leader = null;
            f.ForkLogits = null;
            if (CachesBlocks && f.PromptHashes is not null)
            {
                RegisterComputing(f, f.PromptPos);
            }
        }
    }

    /// <summary>
    /// After the step that computed a leader's last prompt token: every follower takes the leader's context (shared KV blocks, a copy of
    /// recurrent state) and samples its first token from its copy of the leader's logits.
    /// </summary>
    private void ForkFollowers()
    {
        for (int i = 0; i < _inputs.Count; i++)
        {
            var s = _stepSeqs[i];
            if (s.Followers is not { } followers || s.Prefilling || !_inputs[i].WantLogits || _outcomes[i].Error is not null || s.Slot is not { } source)
            {
                continue;
            }

            s.Followers = null;
            foreach (var f in followers)
            {
                f.Leader = null;
                var logits = f.ForkLogits;
                f.ForkLogits = null;
                if (f.Finished != 0)
                {
                    continue;
                }

                var slot = f.Slot!;
                if (logits is null || !_decoder.ForkKv(slot.Index, source.Index))
                {
                    if (CachesBlocks && f.PromptHashes is not null)
                    {
                        RegisterComputing(f, f.PromptPos);
                    }

                    continue;
                }

                slot.Tokens.AddRange(source.Tokens);
                slot.BlockHashes.AddRange(source.BlockHashes);
                f.PromptPos = f.Prompt.Length;
                f.Cached = f.Prompt.Length;
                f.SnapshotPos = s.SnapshotPos;
                Metrics.Forked();
                try
                {
                    int tok = SampleToken(f, logits);
                    if (Emit(f, tok, logits, slot.Tokens.Count) is { } e)
                    {
                        Complete(f, e.Reason, null, e.StopSequence, flush: e.Flush);
                    }
                }
                catch (Exception ex)
                {
                    Complete(f, FinishReason.Error, ex.Message, null);
                }
            }
        }
    }

    /// <summary>Positions before the first content key in which two prompts' non-text content differs (<see cref="int.MaxValue"/>: none differs).</summary>
    private static int SameContentUntil(IReadOnlyList<PromptContentKey> a, IReadOnlyList<PromptContentKey> b)
    {
        int n = Math.Min(a.Count, b.Count);
        for (int i = 0; i < n; i++)
        {
            if (a[i] != b[i])
            {
                return Math.Min(a[i].Start, b[i].Start);
            }
        }

        return a.Count > n ? a[n].Start : b.Count > n ? b[n].Start : int.MaxValue;
    }

    /// <summary>The content hashes of the prompt's full blocks before its last token (which is always computed).</summary>
    private List<UInt128> PromptHashes(EngineSequence seq)
    {
        if (seq.PromptHashes is null)
        {
            var r = seq.Request;
            seq.PromptHashes = new List<UInt128>(seq.Prompt.Length / _decoder.KvBlockSize);
            KvBlockHash.Extend(seq.PromptHashes, KvBlockHash.Seed(r.CachePartition, r.CacheSalt), seq.Prompt, seq.Prompt.Length - 1, _decoder.KvBlockSize, seq.Content);
        }

        return seq.PromptHashes;
    }

    /// <summary>
    /// Attaches the cached blocks that continue a slot holding exactly the first <paramref name="pos"/> tokens of <paramref name="tokens"/>
    /// (a whole number of blocks; <paramref name="hashes"/>: the hashes of the blocks of <paramref name="tokens"/>) and returns the tokens added.
    /// Nothing is attached when that would take more than <paramref name="freeBudget"/> free (cached) blocks.
    /// </summary>
    private int AttachCached(EngineSequence seq, EngineSlot slot, IReadOnlyList<int> tokens, List<UInt128> hashes, int pos, int freeBudget = int.MaxValue)
    {
        int bs = _decoder.KvBlockSize;
        int first = pos / bs;
        if (pos % bs != 0 || first >= hashes.Count || slot.Tokens.Count != pos || _decoder.Length(slot.Index) != pos)
        {
            return 0;
        }

        PublishSlot(slot);
        if (slot.BlockHashes.Count != first)
        {
            return 0;
        }

        if (freeBudget != int.MaxValue && _decoder.FindKvBlocks(CollectionsMarshal.AsSpan(hashes)[first..], out int inUse) - inUse > freeBudget)
        {
            return 0;
        }

        int added = _decoder.AttachKvBlocks(slot.Index, CollectionsMarshal.AsSpan(hashes)[first..]);
        for (int i = 0; i < added; i++)
        {
            slot.Tokens.Add(tokens[pos + i]);
        }

        for (int j = 0; j < added / bs; j++)
        {
            slot.BlockHashes.Add(hashes[first + j]);
            if (_computing.TryGetValue(hashes[first + j], out var owner) && owner == seq)
            {
                _computing.Remove(hashes[first + j]);
            }
        }

        if (added > 0)
        {
            Metrics.SharedPrefix(added, added / bs);
            bool continues = _promptEndKeys.Remove(hashes[first + added / bs - 1], out int cls);
            if (continues)
            {
                _promptEndRestored[cls]++;
            }

            seq.SnapshotPos = pos + added;
            seq.Continues |= continues;
        }

        return added;
    }

    /// <summary>Notes the prompt blocks the sequence will compute itself (from <paramref name="pos"/> on) that no earlier sequence computes.</summary>
    private void RegisterComputing(EngineSequence seq, int pos)
    {
        var hashes = seq.PromptHashes!;
        for (int j = (pos + _decoder.KvBlockSize - 1) / _decoder.KvBlockSize; j < hashes.Count; j++)
        {
            _computing.TryAdd(hashes[j], seq);
        }
    }

    /// <summary>Forgets the blocks a sequence was going to compute (it finished, was preempted, or computed or attached them).</summary>
    private void UnregisterComputing(EngineSequence seq)
    {
        if (seq.PromptHashes is not { } hashes || _computing.Count == 0)
        {
            return;
        }

        foreach (var h in hashes)
        {
            if (_computing.TryGetValue(h, out var owner) && owner == seq)
            {
                _computing.Remove(h);
            }
        }
    }

    /// <summary>
    /// Recurrent state: asks for snapshots at the end of the prefix the prompt shares with another context (a running prompt or a slot's
    /// context), so the next prompt with that prefix starts from it, and at the prompt's last full block (a later turn of the same chat).
    /// </summary>
    private void WantSnapshots(EngineSequence seq, int shared, int pos)
    {
        var hashes = seq.PromptHashes!;
        int bs = _decoder.KvBlockSize;
        if (_wantSnapshot.Count + _wantPromptEnd.Count > 4 * _slots.Length + 64)
        {
            _wantSnapshot.Clear();   // boundaries of requests that ended before reaching them
            _wantPromptEnd.Clear();
            _takeSnapshot.Clear();
        }

        if (shared * bs > pos)
        {
            _wantSnapshot.Add(hashes[shared - 1]);
        }

        // The prompt's end, one block before its last full block: a chat template's generation prompt (e.g. "<|im_start|>assistant\n"
        // and an empty thinking block) is rendered differently once the reply is part of the history, so the next turn diverges a few
        // tokens before this prompt's end.
        int end = (seq.Prompt.Length - 1 - bs) / bs;
        if (end > 0 && end * bs > pos && end <= hashes.Count)
        {
            _wantPromptEnd.Add(hashes[end - 1]);
        }
    }

    /// <summary>Leading blocks of <paramref name="hashes"/> that a slot's context or a running prompt has in common with it.</summary>
    private int SharedBlocks(List<UInt128> hashes)
    {
        int shared = 0;
        foreach (var other in _slots)
        {
            shared = Math.Max(shared, CommonBlocks(hashes, other.BlockHashes));
            if (other.Active?.PromptHashes is { } running)
            {
                shared = Math.Max(shared, CommonBlocks(hashes, running));
            }
        }

        return shared;
    }

    /// <summary>
    /// Hashes and publishes the slot's newly full blocks (they stay cached for later prompts), and snapshots recurrent state at a wanted
    /// block boundary.
    /// </summary>
    private void PublishSlot(EngineSlot slot)
    {
        int before = slot.BlockHashes.Count;
        KvBlockHash.Extend(slot.BlockHashes, slot.Seed, slot.Tokens, slot.Tokens.Count, _decoder.KvBlockSize, slot.Content);
        if (slot.BlockHashes.Count == before)
        {
            return;
        }

        _decoder.PublishKvBlocks(slot.Index, before, CollectionsMarshal.AsSpan(slot.BlockHashes)[before..]);
        if (slot.Active is { } owner)
        {
            for (int j = before; j < slot.BlockHashes.Count; j++)
            {
                if (_computing.TryGetValue(slot.BlockHashes[j], out var o) && o == owner)
                {
                    _computing.Remove(slot.BlockHashes[j]);
                }
            }
        }

        var last = slot.BlockHashes[^1];
        int length = slot.Tokens.Count;
        if (SnapshotsState && slot.Active is { InsideSnapshot: > 0 } inside)
        {
            // The state the step kept at a boundary inside its prompt chunk.
            int b = inside.InsideSnapshot;
            int bs = _decoder.KvBlockSize;
            if (b <= slot.BlockHashes.Count * bs)
            {
                inside.InsideSnapshot = 0;
                var hash = slot.BlockHashes[b / bs - 1];
                if (_decoder.SaveKvStateSnapshotAt(slot.Index, b, hash))
                {
                    NoteSnapshot(slot, hash, b);
                }
            }
        }

        if (!SnapshotsState || length != slot.BlockHashes.Count * _decoder.KvBlockSize)
        {
            return;
        }

        bool chosen = _takeSnapshot.Remove(last);
        bool promptEnd = _wantPromptEnd.Contains(last);
        // A chunk that ended here anyway (the budget): the state is kept when recomputing the tokens since the previous snapshot would
        // cost more than the copies, and no closer than the configured interval.
        bool chunkEnd = slot.Active is { } s && (s.Prefilling || s.Replay is not null)
            && length - s.SnapshotPos >= Math.Max(1, _opt.StateSnapshotInterval)
            && RecomputeMs(length - s.SnapshotPos) > 2 * _cost.CopyMs;
        if (!(chosen || promptEnd || _wantSnapshot.Contains(last) || chunkEnd))
        {
            return;
        }

        long t0 = Stopwatch.GetTimestamp();
        if (!_decoder.SaveKvStateSnapshot(slot.Index, last, speculative: !(chosen || promptEnd || _wantSnapshot.Contains(last))))
        {
            return;
        }

        _cost.ObserveCopy(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
        NoteSnapshot(slot, last, length);
    }

    /// <summary>Bookkeeping of a snapshot taken of the slot's state after <paramref name="length"/> tokens, the block boundary <paramref name="hash"/>.</summary>
    private void NoteSnapshot(EngineSlot slot, UInt128 hash, int length)
    {
        _wantSnapshot.Remove(hash);
        if (_wantPromptEnd.Remove(hash))
        {
            int cls = slot.Active is { Continues: true } ? 1 : 0;
            _promptEndSaved[cls]++;
            if (_promptEndKeys.Count > 4 * _slots.Length + 256)
            {
                _promptEndKeys.Clear();
            }

            _promptEndKeys[hash] = cls;
        }

        if (slot.Active is { } owner && owner.SnapshotPos < length)
        {
            owner.SnapshotPos = length;
        }

        Metrics.StateSnapshot();
    }

    /// <summary>
    /// Prompt rows of a chunk starting at the sequence's prompt position. Blocks another sequence is computing are waited for (and
    /// attached once published) when waiting a pass costs less than computing them here; a recurrent model's chunk stops at a block
    /// boundary whose state snapshot is worth the extra pass it costs this prompt (or costs none); and a chunk that does not end the
    /// prompt ends on a block boundary (so the next chunk can attach cached blocks, and its state can be kept there for free).
    /// All prices come from <see cref="StepCostModel"/>.
    /// </summary>
    private int ClipPrefillChunk(EngineSequence s, int chunk)
    {
        if (!CachesBlocks || s.PromptHashes is not { } hashes)
        {
            return chunk;
        }

        int bs = _decoder.KvBlockSize;
        int pos = s.PromptPos;
        int end = pos + chunk;
        s.InsideSnapshot = 0;
        if (!s.Request.BypassPrefixCache)
        {
            for (int j = (pos + bs - 1) / bs; j < hashes.Count && j * bs < end; j++)
            {
                if (_computing.TryGetValue(hashes[j], out var owner) && owner != s)
                {
                    if (WaitPays(s, owner, CommonBlocks(hashes, owner.PromptHashes!) * bs))
                    {
                        end = j * bs;
                        s.TraceClip += " wait";
                    }
                    else
                    {
                        s.TraceClip += " recompute";
                    }

                    break;
                }
            }
        }

        bool inside = _decoder.SnapshotsInsideSteps;
        if (SnapshotsState && !inside && end - pos > bs)
        {
            int clip = SnapshotClip(s, hashes, pos, end, chunk, inside: false);
            if (clip > 0)
            {
                end = clip;
                _takeSnapshot.Add(hashes[clip / bs - 1]);
                s.TraceClip += " snap";
            }
        }

        if (end < s.Prompt.Length && end - pos > bs)
        {
            end = end / bs * bs;
        }

        if (SnapshotsState && inside && end - pos > bs)
        {
            // The step keeps the state at the boundary itself: the chunk goes on.
            int b = SnapshotClip(s, hashes, pos, end, chunk, inside: true);
            if (b > 0)
            {
                s.InsideSnapshot = b;
                s.TraceClip += " snap-in";
            }
        }

        return Math.Max(0, end - pos);
    }

    /// <summary>
    /// Rows of the prompt chunk after which the step keeps the state for the snapshot <see cref="ClipPrefillChunk"/> chose inside it (0 for
    /// none). A chunk that was shortened since to end exactly there takes the snapshot after the step instead; one that ends before it drops it.
    /// </summary>
    private int InsideSnapshotRows(EngineSequence s, int chunk)
    {
        int b = s.InsideSnapshot;
        int end = s.PromptPos + chunk;
        if (b <= s.PromptPos || b > end || s.Slot!.Tokens.Count != s.PromptPos)
        {
            s.InsideSnapshot = 0;
            return 0;
        }

        if (b == end)
        {
            _takeSnapshot.Add(s.PromptHashes![b / _decoder.KvBlockSize - 1]);
            s.InsideSnapshot = 0;
            return 0;
        }

        return b - s.PromptPos;
    }

    /// <summary>Running sequences, at least 1: how many requests a longer step delays (each step advances every one of them).</summary>
    private int Concurrency => Math.Max(1, _active.Count);

    /// <summary>Milliseconds that computing <paramref name="rows"/> more prompt rows in some step costs, summed over the requests it delays.</summary>
    private double RecomputeMs(int rows) => _cost.PerRowMs * Math.Max(0, rows) * Concurrency;

    /// <summary>
    /// Milliseconds one more pass costs <paramref name="self"/> when it still has <paramref name="rows"/> rows to compute then: the next
    /// step also carries the decode rows of the running requests and the prompt rows the current step's budget leaves of the others, and
    /// the prompt waits for all of it.
    /// </summary>
    private double PassMs(EngineSequence self, int rows)
    {
        long decode = 0, prompts = 0;
        foreach (var a in _active)
        {
            if (a == self)
            {
                continue;
            }

            if (a.Replay is { } replay)
            {
                prompts += replay.Length - a.ReplayPos;
            }
            else if (a.Prefilling)
            {
                prompts += a.Leader is null ? a.Prompt.Length - a.PromptPos : 0;
            }
            else
            {
                decode += a.InputLength;
            }
        }

        long next = Math.Max(1, rows) + decode + Math.Max(0, prompts + decode - _budget);
        return _cost.StepMs((int)Math.Min(_budget, next));
    }

    /// <summary>
    /// Whether <paramref name="waiter"/> should wait for the blocks <paramref name="owner"/> is computing up to position
    /// <paramref name="until"/> (where their prompts part) instead of computing them itself: waiting delays it by the passes the owner
    /// still needs to get there; computing them makes this step longer for every running request.
    /// </summary>
    private bool WaitPays(EngineSequence waiter, EngineSequence owner, int until)
    {
        int rows = until - waiter.PromptPos;
        if (rows <= 0)
        {
            return false;
        }

        int step = Math.Max(_decoder.KvBlockSize, _prefillChunk);
        int passes = Math.Max(1, (until - owner.PromptPos + step - 1) / step);
        return RecomputeMs(rows) > passes * PassMs(waiter, Math.Min(step, waiter.Prompt.Length - until));
    }

    /// <summary>
    /// Recurrent state: the block boundary inside the chunk [<paramref name="pos"/>, <paramref name="end"/>) at which to stop and keep the
    /// state, or 0 for none. A snapshot at boundary b saves later prompts recomputing the tokens since the previous snapshot of this
    /// prompt: prompts running now that part from this one at b and wait for it, a prefix this prompt shares with an idle context (expected
    /// to come again), and the prompt's end (the next turn of a conversation, weighted by how often such snapshots were used). It costs this
    /// prompt one more pass, unless the chunk is cut short by the budget anyway and stopping at b adds no pass to the rest of the prompt, or
    /// the step keeps the state at b itself (<paramref name="inside"/>: <see cref="IBatchDecoder.SnapshotsInsideSteps"/>; the chunk is not cut).
    /// </summary>
    private int SnapshotClip(EngineSequence s, List<UInt128> hashes, int pos, int end, int chunk, bool inside)
    {
        int bs = _decoder.KvBlockSize;
        int length = s.Prompt.Length;
        int cls = s.Continues ? 1 : 0;
        double promptEndUse = (_promptEndRestored[cls] + 1.0) / (_promptEndSaved[cls] + 2.0);
        int best = 0;
        double bestGain = 0;
        for (int j = pos / bs; j < hashes.Count && (j + 1) * bs < end; j++)
        {
            int b = (j + 1) * bs;
            if (b <= pos)
            {
                continue;
            }

            double uses = (_wantSnapshot.Contains(hashes[j]) ? 1 : 0) + (_wantPromptEnd.Contains(hashes[j]) ? promptEndUse : 0);
            double value = uses * RecomputeMs(b - s.SnapshotPos) + WaitingValue(s, b);
            if (value <= 0)
            {
                continue;
            }

            // Free when the chunk ends early anyway (the budget) and the rest of the prompt needs no more passes from b than from end.
            bool free = inside || (end < length && Passes(length - b, chunk) == Passes(length - end, chunk));
            double gain = value - _cost.CopyMs - (free ? 0 : PassMs(s, end - b));
            if (gain > bestGain)
            {
                best = b;
                bestGain = gain;
            }
        }

        return best;

        static int Passes(int rows, int perPass) => (rows + Math.Max(1, perPass) - 1) / Math.Max(1, perPass);
    }

    /// <summary>
    /// What a snapshot of <paramref name="owner"/>'s state at <paramref name="boundary"/> is worth to the prompts running now that part from
    /// it exactly there and wait for its blocks: the rows each of them would otherwise compute.
    /// </summary>
    private double WaitingValue(EngineSequence owner, int boundary)
    {
        double value = 0;
        int bs = _decoder.KvBlockSize;
        foreach (var w in _active)
        {
            if (w == owner || !w.Prefilling || w.Replay is not null || w.Request.BypassPrefixCache || w.PromptHashes is not { } wh || w.PromptPos >= boundary
                || CommonBlocks(wh, owner.PromptHashes!) * bs != boundary || !WaitPays(w, owner, boundary))
            {
                continue;
            }

            value += RecomputeMs(boundary - w.PromptPos);
        }

        return value;
    }

    /// <summary>Leading blocks two hash chains have in common.</summary>
    private static int CommonBlocks(List<UInt128> a, List<UInt128> b)
    {
        int n = Math.Min(a.Count, b.Count);
        int k = 0;
        while (k < n && a[k] == b[k])
        {
            k++;
        }

        return k;
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
        int blocks = Paged ? keep / _decoder.KvBlockSize : 0;
        if (slot.BlockHashes.Count > blocks)
        {
            slot.BlockHashes.RemoveRange(blocks, slot.BlockHashes.Count - blocks);
        }
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

    /// <summary>
    /// The token budget of the next step: <see cref="EngineOptions.BatchTokensPerSequence"/> rows per sequence holding a slot,
    /// within <see cref="EngineOptions.MinBatchTokens"/> and the full budget. Few sequences: a wave of prompts is prefilled in
    /// smaller steps and its first prompts stream sooner; many: full steps, whose larger GEMMs prefill more rows per second.
    /// </summary>
    private int StepTokens()
    {
        if (_minBudget >= _budget)
        {
            return _budget;
        }

        // The floor policy (EngineOptions.StepFloor, SHARPLIGHT_STEP_FLOOR_WAVES): while some sequence decodes, no floor (Waves), the
        // whole budget (Full), or the floor as in a wave (Always).
        var policy = DiagnosticSwitches.StepFloor ?? _opt.StepFloor;
        bool decoding = policy != StepFloorPolicy.Always && _active.Exists(s => !s.Prefilling && s.Replay is null);
        if (decoding && policy == StepFloorPolicy.Full)
        {
            return _budget;
        }

        // The oldest prompt still prefilling may always take a whole step: a lone long prompt keeps the full budget, only prompts
        // that compete with each other are spread over smaller steps.
        long oldest = 0;
        foreach (var s in _active)
        {
            if (s.Replay is { } replay)
            {
                oldest = replay.Length - s.ReplayPos;
                break;
            }

            if (s.Prefilling)
            {
                oldest = s.Prompt.Length - s.PromptPos;
                break;
            }
        }

        // The load is the number of sequences holding a slot, held for a while when it drops (by one per step): a group of streams that
        // finished together is replaced by new requests over the next few steps, which should not see the budget of a light load.
        // (After a pause, e.g. an idle engine, the load starts over.)
        long now = Stopwatch.GetTimestamp();
        _load = Stopwatch.GetElapsedTime(_loadTimestamp, now) > LoadHoldTime ? _active.Count : Math.Max(_active.Count, _load - 1);
        _loadTimestamp = now;
        long rows = Math.Max((long)_load * PromptRowsPerSequence(decoding), _active.Count + oldest);
        // Whole token tiles: a prompt-sized GEMM's time follows its count of token tiles, so a budget inside a tile is rounded up.
        long granule = Math.Max(1, _opt.BatchTokensGranule);
        rows = (rows + granule - 1) / granule * granule;
        return (int)Math.Clamp(rows, decoding ? Math.Min(granule, _budget) : _minBudget, _budget);
    }

    /// <summary>
    /// Rows per sequence holding a slot of the next step's budget: <see cref="EngineOptions.BatchTokensPerSequence"/>, or while some
    /// sequence decodes <see cref="EngineOptions.SmallChunkRowsPerSequence"/> when the measured prices say prompt rows cost less per row
    /// in such smaller chunks (<see cref="TileStepCost.PrefersSmallChunks"/>; <see cref="EngineOptions.PricedPromptChunks"/>).
    /// </summary>
    private int PromptRowsPerSequence(bool decoding)
    {
        int large = Math.Max(1, DiagnosticSwitches.BatchTokensPerSequence ?? _opt.BatchTokensPerSequence);
        _smallChunks = false;
        if (!decoding || _tiles is null || !_opt.PricedPromptChunks)
        {
            return large;
        }

        int small = Math.Max(1, Math.Min(large, _opt.SmallChunkRowsPerSequence));
        _smallChunks = _tiles.PrefersSmallChunks((int)Math.Min((long)_load * small, _budget), (int)Math.Min((long)_load * large, _budget));
        return _smallChunks ? small : large;
    }

    private void StepOnce()
    {
        if (_diffusion is not null)
        {
            StepDiffusion(_diffusion);
            return;
        }

        long stepStart = Stopwatch.GetTimestamp();
        _inputs.Clear();
        _stepSeqs.Clear();
        _stepBaseLength.Clear();
        _stepReplay.Clear();
        if (_speculator is not null)
        {
            ProposeDrafts();
        }

        int stepTokens = StepTokens();
        EncodeMedia();
        var budget = new StepBudget { Tokens = stepTokens, LogitsRows = _decoder.MaxLogitsRows };
        budget.KvBlocks = EnsureKvBlocksForDecode();
        ScheduleDecode(ref budget);
        ScheduleReplay(ref budget);
        ScheduleChunkedPrefill(ref budget);

        if (_inputs.Count == 0)
        {
            FlushEmits();
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
        var elapsed = Stopwatch.GetElapsedTime(t0);
        Metrics.Step(stepTokens - budget.Tokens, _inputs.Count, elapsed);
        if (_tiles is not null && !(_cost.Seen > 16 && elapsed.TotalMilliseconds > 3 * _cost.StepMs(stepTokens - budget.Tokens) + 1))
        {
            // Steps far slower than the linear estimate (a new step shape recorded, a shared device) would distort the tile prices.
            var (decodeRows, promptRows, _) = ProfileRows();
            _tiles.Observe(decodeRows, promptRows, elapsed.TotalMilliseconds);
        }

        _cost.Observe(stepTokens - budget.Tokens, elapsed.TotalMilliseconds);
        if (DiagnosticSwitches.SchedulerTrace)
        {
            TraceStep(elapsed, stepTokens - budget.Tokens);
        }

        long processStart = Stopwatch.GetTimestamp();
        var profileRows = _profiler is null ? default : ProfileRows();
        ProcessResults(results);
        if (observe)
        {
            ObserveHiddenStates();
        }

        UpdateKvGauges();
        if (_deferEmits)
        {
            foreach (var s in _stepSeqs)
            {
                if (s.PendingEmits.Count > 0 && (_emitPending.Count == 0 || _emitPending[^1] != s))
                {
                    _emitPending.Add(s);
                }
            }
        }

        if (_profiler is not null)
        {
            _profiler.Write(stepStart, profileRows.Decode, profileRows.Prompt, profileRows.PromptInputs, _active.Count, Stopwatch.GetElapsedTime(stepStart, t0).TotalMilliseconds,
                (_decoder as MultiSequenceDecoder)?.LastStepPhases ?? new StepPhaseTimes(0, 0, elapsed.TotalMilliseconds, 0, default),
                Stopwatch.GetElapsedTime(processStart).TotalMilliseconds);
        }
    }

    /// <summary>The decode rows, prompt rows and prompt inputs of the step being processed (<see cref="StepProfiler"/>).</summary>
    private (int Decode, int Prompt, int PromptInputs) ProfileRows()
    {
        int decode = 0;
        int prompt = 0;
        int promptInputs = 0;
        for (int i = 0; i < _inputs.Count; i++)
        {
            int n = _inputs[i].Tokens.Length;
            if (_stepReplay[i] || _stepSeqs[i].Prefilling)
            {
                prompt += n;
                promptInputs++;
            }
            else
            {
                decode += n;
            }
        }

        return (decode, prompt, promptInputs);
    }

    /// <summary>Writes one line about the step that just ran (<see cref="DiagnosticSwitches.SchedulerTrace"/>).</summary>
    private void TraceStep(TimeSpan elapsed, int rows)
    {
        var sb = new System.Text.StringBuilder();
        int decode = 0;
        for (int i = 0; i < _inputs.Count; i++)
        {
            var s = _stepSeqs[i];
            int n = _inputs[i].Tokens.Length;
            if (_stepReplay[i])
            {
                sb.Append($" R{s.Id}:{n}");
            }
            else if (s.Prefilling)
            {
                sb.Append($" P{s.Id}@{s.PromptPos}+{n}/{s.Prompt.Length}{s.TraceClip}");
            }
            else
            {
                decode += n;
            }
        }

        if (_heldRows > 0 && _tiles is not null)
        {
            sb.Append(System.Globalization.CultureInfo.InvariantCulture, $" held<={_heldRows} (tile {_tiles.TileMs:F2} ms, row {_tiles.RowMs:F4} ms)");
        }

        if (_smallChunks)
        {
            sb.Append(" small");
        }

        Console.Error.WriteLine($"[sched] {elapsed.TotalMilliseconds,7:F2} ms rows {rows,5} decode {decode,3} active {_active.Count,3}{sb}");
    }

    /// <summary>
    /// Masked diffusion: one denoising pass over every running sequence (their spans start as mask tokens at admission), then the tokens of
    /// newly completed blocks are committed like sampled tokens (stop tokens, stop strings, streaming).
    /// </summary>
    private void StepDiffusion(DiffusionBatchDecoder decoder)
    {
        _stepSeqs.Clear();
        _diffusionBatch.Clear();
        int rows = 0;
        foreach (var s in _active)
        {
            s.Diffusion ??= new DiffusionSequence(s.Prompt, s.MaxTokens, s.Request.Diffusion ?? DefaultDiffusion, decoder.MaskToken, decoder.ShiftLogits);
            s.PromptPos = s.Prompt.Length;
            _stepSeqs.Add(s);
            _diffusionBatch.Add(s.Diffusion);
            rows += s.Diffusion.Tokens.Length;
        }

        long t0 = Stopwatch.GetTimestamp();
        var logits = decoder.Denoise(_diffusionBatch);
        Metrics.Step(rows, _diffusionBatch.Count, Stopwatch.GetElapsedTime(t0));
        int vocab = decoder.VocabSize;
        if (_diffusionBatch.Count >= ParallelSamplingThreshold)
        {
            Parallel.For(0, _diffusionBatch.Count, i => _diffusionBatch[i].Apply(logits[i], vocab));
        }
        else
        {
            for (int i = 0; i < _diffusionBatch.Count; i++)
            {
                _diffusionBatch[i].Apply(logits[i], vocab);
            }
        }

        for (int i = 0; i < _stepSeqs.Count; i++)
        {
            var s = _stepSeqs[i];
            var sequence = _diffusionBatch[i];
            _queue.Charge(s.Request.Tenant, sequence.Tokens.Length);
            try
            {
                // Generated counts the committed tokens; each final token is committed as if it had been sampled after the ones before it.
                while (s.Finished == 0 && s.Generated < sequence.Final)
                {
                    if (Commit(s, sequence.Generated.Span[s.Generated], null, s.Prompt.Length + s.Generated) is { } e)
                    {
                        Complete(s, e.Reason, null, e.StopSequence, flush: e.Flush);
                    }
                }
            }
            catch (Exception ex)
            {
                Complete(s, FinishReason.Error, ex.Message, null);
            }
        }
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

            if (CachesBlocks && s.ReplayHashes is { } rh)
            {
                // Blocks of the preempted context that are still cached come back without recomputation (taking cached free blocks).
                int free = _decoder.KvBlocksFree;
                s.ReplayPos += AttachCached(s, s.Slot!, replay, rh, s.ReplayPos, budget.KvBlocks);
                budget.KvBlocks -= free - _decoder.KvBlocksFree;
                if (s.ReplayPos == replay.Length)
                {
                    s.Replay = null;
                    s.ReplayHashes = null;
                    continue;
                }
            }

            int chunk = Math.Min(Math.Min(Math.Min(_prefillChunk, replay.Length - s.ReplayPos), budget.Tokens), KvRoom(s.Slot!, budget.KvBlocks));
            chunk = ClipToEncodedMedia(s, s.ReplayPos, chunk);
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
        int held = HoldRowsPastTile(ref budget);
        _heldRows = held;
        for (int i = 0; i < _active.Count; i++)
        {
            var s = _active[i];
            if (!s.Prefilling || s.Replay is not null || s.Leader is not null || budget.Tokens <= 0)
            {
                continue;
            }

            s.TraceClip = "";

            if (CachesBlocks && !s.Request.BypassPrefixCache && s.PromptHashes is { } hashes)
            {
                // Blocks published since admission (another prompt computed them meanwhile) are attached mid-prompt.
                int free = _decoder.KvBlocksFree;
                int added = AttachCached(s, s.Slot!, s.Prompt, hashes, s.PromptPos, budget.KvBlocks);
                budget.KvBlocks -= free - _decoder.KvBlocksFree;
                if (added > 0)
                {
                    s.TraceClip = $" attach{added}";
                    Metrics.CachedLater(added, s.Cached == 0);
                    s.PromptPos += added;
                    s.Cached += added;
                }
            }

            int rest = s.Prompt.Length - s.PromptPos;
            int room = KvRoom(s.Slot!, budget.KvBlocks);
            int chunk = Math.Min(Math.Min(Math.Min(_prefillChunk, rest), budget.Tokens), room);
            bool budgetCut = chunk == budget.Tokens && chunk < rest && rest <= Math.Min(_prefillChunk, room);
            chunk = ClipPrefillChunk(s, chunk);
            if (held == 0 && budgetCut && chunk < rest && TakesWholeRest(i, chunk, rest))
            {
                chunk = rest;
                s.TraceClip += " whole";
            }

            chunk = ClipToEncodedMedia(s, s.PromptPos, chunk);
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
            int insideRows = InsideSnapshotRows(s, chunk);
            AddInput(s, new SequenceInput(s.Slot!.Index, s.Prompt.AsMemory(s.PromptPos, chunk), last) { StateSnapshotRows = insideRows });
            budget.Tokens -= chunk;
            budget.LogitsRows -= last ? 1 : 0;
        }

        // The rows held back stay unused (the step's row count is the budget it spent).
        budget.Tokens += held;
    }

    /// <summary>Rows of the inputs scheduled so far in this step, and whether one of them is a decode row.</summary>
    private (int Rows, bool Decodes) ScheduledRows()
    {
        int rows = 0;
        bool decodes = false;
        for (int i = 0; i < _inputs.Count; i++)
        {
            rows += _inputs[i].Tokens.Length;
            decodes |= !_stepReplay[i] && !_stepSeqs[i].Prefilling;
        }

        return (rows, decodes);
    }

    /// <summary>
    /// Holds back the prompt rows that would make the step start another token tile (<see cref="EngineOptions.BatchTokensGranule"/>) when
    /// every waiting prompt row fits the step, at most <see cref="TileStepCost.HeldMaxRows"/> rows end past the tile, the step decodes
    /// (the held rows join a later step's decode rows) and the measured prices say the tile costs more than those rows will in a later
    /// step (<see cref="TileStepCost"/>). Returns the rows taken from the budget.
    /// </summary>
    private int HoldRowsPastTile(ref StepBudget budget)
    {
        if (_tiles is null || budget.Tokens <= 0 || SnapshotsState)
        {
            return 0;
        }

        long prompt = 0;
        foreach (var s in _active)
        {
            if (s.Prefilling && s.Replay is null && s.Leader is null)
            {
                prompt += Math.Min(_prefillChunk, s.Prompt.Length - s.PromptPos);
            }
        }

        var (scheduled, decodes) = ScheduledRows();
        if (prompt == 0 || prompt >= budget.Tokens || !decodes)
        {
            return 0;
        }

        int rows = scheduled + (int)prompt;
        int over = rows % _tiles.TileRows;
        if (rows < _tiles.TileRows || over == 0 || over > _tiles.HeldMaxRows || over >= prompt
            || !_tiles.Moves(rows, over))
        {
            return 0;
        }

        int held = budget.Tokens - ((int)prompt - over);
        budget.Tokens -= held;
        return held;
    }

    /// <summary>
    /// Whether the prompt chunk of the sequence at <paramref name="index"/> in <see cref="_active"/>, cut by the step's budget to
    /// <paramref name="chunk"/> of its <paramref name="rest"/> rows, takes the whole rest instead: when at most
    /// <see cref="TileStepCost.HeldMaxRows"/> rows would be left, no other prompt waits (they would ride with it), the step decodes, the
    /// decoder takes the rows, and the measured prices say those rows cost less here (with the token tile they may start) than in a
    /// later step of decode rows.
    /// </summary>
    private bool TakesWholeRest(int index, int chunk, int rest)
    {
        int left = rest - chunk;
        if (_tiles is null || left <= 0 || left > _tiles.HeldMaxRows || SnapshotsState)
        {
            return false;
        }

        for (int j = index + 1; j < _active.Count; j++)
        {
            if (_active[j].Prefilling || _active[j].Replay is not null)
            {
                return false;
            }
        }

        var (scheduled, decodes) = ScheduledRows();
        int rows = scheduled + rest;
        return decodes && rows <= _budget && _tiles.Takes(rows, left);
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

        ForkFollowers();
        ApplyOutcomes();
        if (CachesBlocks)
        {
            foreach (var s in _stepSeqs)
            {
                if (s.Slot is { } slot && slot.Tokens.Count > 0)
                {
                    PublishSlot(slot);
                }
            }
        }
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
            _deviceRows[i] = _inputs[i].OutputRows >= 1 && s.Finished == 0 && s.Matcher is null && s.Generated >= s.Request.MinTokens && s.Followers is null
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

            if (s.Followers is { } followers && !s.Prefilling && rows == 1)
            {
                // The leader's first token: its followers sample theirs from the same row (before this sequence masks it).
                foreach (var f in followers)
                {
                    f.ForkLogits = logits.Span[..vocab].ToArray();
                }
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

    /// <summary>
    /// The encoder stage before a step: the media that the next prompt (or recomputation) chunk of every sequence may reach and that have no
    /// rows yet take them from the cache of encoder outputs, or are encoded, up to <see cref="EngineOptions.MediaEncodeRowsPerStep"/> rows, in
    /// one call per encoder (batched across requests). Media inside KV blocks a prompt attached from the prefix cache are never reached.
    /// </summary>
    private void EncodeMedia()
    {
        _mediaPending.Clear();
        int rows = 0;
        int limit = _opt.MediaEncodeRowsPerStep > 0 ? _opt.MediaEncodeRowsPerStep : _budget;
        foreach (var s in _active)
        {
            if (s.Layout is not { } layout || layout.Media.Count == 0 || s.Leader is not null)
            {
                continue;
            }

            int pos = s.Replay is not null ? s.ReplayPos : s.Prefilling ? s.PromptPos : -1;
            if (pos < 0)
            {
                continue;
            }

            for (int i = 0; i < layout.Media.Count; i++)
            {
                var m = layout.Media[i];
                if (m.End <= pos || m.Start >= pos + _prefillChunk || layout.Rows(i) is not null)
                {
                    continue;
                }

                if (_mediaCache.Get(m.Content.Hash) is { } cached)
                {
                    layout.SetRows(i, cached);
                    Metrics.MediaCacheHit();
                    continue;
                }

                if (_mediaPending.Count > 0 && rows + m.Content.Rows > limit)
                {
                    continue;
                }

                _mediaPending.Add((s, i));
                rows += m.Content.Rows;
            }
        }

        if (_mediaPending.Count == 0)
        {
            return;
        }

        long t0 = Stopwatch.GetTimestamp();
        int encodedCount = 0;
        foreach (var group in _mediaPending.GroupBy(p => ContentOf(p).Encoder).ToList())
        {
            try
            {
                encodedCount += EncodeGroup(group.Key, [.. group]);
            }
            catch (Exception e) when (e is not OutOfMemoryException)
            {
                // A content the encoder cannot take (or a failing encoder) ends the requests that need it, not the engine.
                foreach (var (seq, _) in group)
                {
                    Complete(seq, FinishReason.Error, "Encoding the prompt's media failed: " + e.Message, null);
                }
            }
        }

        Metrics.MediaEncoded(encodedCount, Stopwatch.GetElapsedTime(t0));
    }

    private static MediaContent ContentOf((EngineSequence Sequence, int Index) pending) => pending.Sequence.Layout!.Media[pending.Index].Content;

    /// <summary>Encodes the contents of one encoder (each distinct content once), caches the rows and hands them to the layouts; returns the contents encoded.</summary>
    private int EncodeGroup(IMediaEncoder encoder, List<(EngineSequence Sequence, int Index)> group)
    {
        var distinct = group.Select(ContentOf).DistinctBy(c => c.Hash).ToList();
        var encoded = encoder.Encode(distinct);
        var byHash = new Dictionary<UInt128, float[]>();
        for (int k = 0; k < distinct.Count; k++)
        {
            if (k >= encoded.Count || encoded[k].Length != (long)distinct[k].Rows * _decoder.EmbeddingWidth)
            {
                throw new InvalidOperationException($"The media encoder did not return {distinct[k].Rows} rows of width {_decoder.EmbeddingWidth}.");
            }

            byHash[distinct[k].Hash] = encoded[k];
        }

        foreach (var (hash, rows) in byHash)
        {
            _mediaCache.Add(hash, rows);
        }

        foreach (var pending in group)
        {
            pending.Sequence.Layout!.SetRows(pending.Index, byHash[ContentOf(pending).Hash]);
        }

        return distinct.Count;
    }

    /// <summary>A prompt or recomputation chunk from context row <paramref name="pos"/> cut before the first media whose rows are not encoded yet.</summary>
    private int ClipToEncodedMedia(EngineSequence s, int pos, int chunk)
    {
        if (chunk <= 0 || s.Layout is not { } layout || layout.FirstMissing(pos, chunk) is not (>= 0 and var missing))
        {
            return chunk;
        }

        // A chunk that does not end the prompt ends on a block boundary where it can.
        int start = layout.Media[missing].Start;
        int bs = _decoder.KvBlockSize;
        int aligned = bs > 0 ? start / bs * bs - pos : 0;
        return Math.Max(0, aligned > 0 ? aligned : start - pos);
    }

    private void AddInput(EngineSequence s, SequenceInput input, bool replay = false)
    {
        if (s.Layout is { } layout)
        {
            input = layout.Apply(input, s.Slot!.Tokens.Count);
        }

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
                    chunks += _decoder.KvBlocksNeeded(slot, Math.Min(_prefillChunk, replay.Length - s.ReplayPos));
                }
                else if (s.Prefilling)
                {
                    chunks += s.Leader is null ? _decoder.KvBlocksNeeded(slot, Math.Min(_prefillChunk, s.Prompt.Length - s.PromptPos)) : 0;
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
        if (victim.Replay is null)
        {
            victim.Replay = slot.Tokens.Count > 0 ? [.. slot.Tokens] : null;
            victim.ReplayHashes = CachesBlocks && slot.BlockHashes.Count > 0 ? [.. slot.BlockHashes] : null;
        }

        victim.ReplayPos = 0;
        UnregisterComputing(victim);
        ReleaseFollowers(victim);
        slot.BlockHashes.Clear();
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
        if (_deferEmits && lps is null && !s.Filter.CanStop)
        {
            // Nothing in the text can end the sequence: it is decoded and streamed while the next step runs (FlushEmits).
            s.PendingEmits.Add(tok);
        }
        else
        {
            string text = s.Filter.Push(s.Decoder.Push(tok));
            if (text.Length > 0 || lps is not null)
            {
                s.Writer.TryWrite(new GenerationChunk(text, lps));
            }
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

    /// <summary>
    /// Decodes and streams the tokens deferred by <see cref="Commit"/> (called while the next step runs on the device, before a sequence
    /// completes, and when no step follows), in commit order per sequence.
    /// </summary>
    private void FlushEmits()
    {
        foreach (var s in _emitPending)
        {
            FlushEmits(s);
        }

        _emitPending.Clear();
    }

    /// <summary>Decodes and streams one sequence's deferred tokens.</summary>
    private static void FlushEmits(EngineSequence s)
    {
        if (s.PendingEmits.Count == 0)
        {
            return;
        }

        var text = new System.Text.StringBuilder();
        foreach (int tok in s.PendingEmits)
        {
            text.Append(s.Filter.Push(s.Decoder.Push(tok)));
        }

        s.PendingEmits.Clear();
        if (text.Length > 0)
        {
            s.Writer.TryWrite(new GenerationChunk(text.ToString(), null));
        }
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
        _kvBlocksCached = _decoder.KvBlocksCached;
        Interlocked.Exchange(ref _kvBlocksEvicted, _decoder.KvBlocksEvicted);
    }

    /// <summary>Ends a running sequence: emits held-back text (when <paramref name="flush"/>), the final chunk, frees the slot.</summary>
    private void Complete(EngineSequence s, FinishReason reason, string? error, string? stopSeq, bool flush = false, bool reset = false)
    {
        if (s.Finished != 0)
        {
            return;
        }

        FlushEmits(s);
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
        UnregisterComputing(s);
        ReleaseFollowers(s);
        s.Leader?.Followers?.Remove(s);
        s.Leader = null;
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
            slot.BlockHashes.Clear();
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
