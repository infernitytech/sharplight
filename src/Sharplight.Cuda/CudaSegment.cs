using System.Diagnostics;
using Sharplight.Cuda.Interop;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Shaders.Lowering;

namespace Sharplight.Cuda;

/// <summary>
/// A compiled segment on a CUDA device: owns the device buffers of its <see cref="KernelProgram"/> and launches its kernels on a stream of
/// its own, from a recording per distinct binding of the symbolic dimensions (kernels, arguments and grids resolved once). Launches on one
/// stream run in order, so no barriers are needed; a captured step graph runs the launches of each hazard batch (the Vulkan recorder's
/// barrier placement) side by side on side streams (<see cref="CudaConcurrentLanes"/>). Small inputs and outputs are copied asynchronously on the stream through page-locked staging (<see cref="CudaHostStaging"/>), larger ones synchronously. An execution returns once its work is
/// submitted (the host can prepare the next step meanwhile); the segment waits for it before it reads an output, takes an input, copies
/// state or runs again, and before another segment of the device runs (<see cref="CudaDevice.Pending"/>). Profiled executions wait at once.
/// </summary>
public sealed unsafe class CudaSegment : IStreamedSegment, IPartialOutputSegment, IDeviceForwardingSegment
{
    /// <summary>
    /// Most steps kept recorded per segment (one per distinct symbol binding): the decode steps of every batch size of a 64-slot
    /// decoder and its prompt-bearing step shapes (each graph also counts against the device's memory, see <see cref="RunStep"/>).
    /// </summary>
    private const int MaxRecordings = 128;

    /// <summary>Runtime scalar slots (uints) of the runtime buffer.</summary>
    private const int RuntimeSlots = 64;

    private readonly CudaDevice _dev;
    private readonly KernelProgram _program;
    private readonly CompileOptions _options;
    private readonly Dictionary<BufferDecl, CudaBuffer> _alloc = [];
    /// <summary>Weight buffers held through <see cref="CompileOptions.SharedWeights"/> (their buffers are freed by the lease).</summary>
    private readonly Dictionary<BufferDecl, SharedWeightLease<CudaBuffer>> _leases = [];
    /// <summary>State buffers another segment of the same graph allocated (and frees).</summary>
    private readonly HashSet<BufferDecl> _borrowedStates = [];
    /// <summary>Activation buffers shared with the model's other segments on this device (see <see cref="SharedArena{T}"/>).</summary>
    private readonly Dictionary<BufferDecl, SharedArenaBuffer<CudaBuffer>> _sharedArenas = [];
    /// <summary>With <see cref="CompileOptions.StreamWeights"/>: the weight buffers, copied in at every execution (see <see cref="StreamWeights"/>).</summary>
    private readonly List<BufferDecl> _streamedWeights = [];
    /// <summary>Streamed weights read from page-locked mapped memory: the buffer, its source and the registered (page-aligned) byte range.</summary>
    private readonly List<(BufferDecl Buffer, nint Source, long Offset, long Bytes)> _registeredWeights = [];
    /// <summary>
    /// With <see cref="CompileOptions.SparseExpertSlots"/>: the expert-indexed weight banks, each holding that many experts (shared with the
    /// model's other sparse segments on the device), their mapped source and the bytes of one expert.
    /// </summary>
    private readonly List<(BufferDecl Buffer, nint Source, long ExpertBytes)> _sparseWeights = [];
    /// <summary>Per sparse bank whose experts stay on disk (<see cref="CompileOptions.DiskExperts"/>): the weight in the expert cache.</summary>
    private readonly Dictionary<BufferDecl, Runtime.Experts.DiskExpertTensor> _diskBanks = [];
    /// <summary>Per expert of a disk bank, the address of its cache slot while this segment holds it (one table per bank, by bank).</summary>
    private readonly Dictionary<BufferDecl, nint[]> _diskSlots = [];
    /// <summary>The experts of disk banks this segment holds until the copies from their slots completed.</summary>
    private readonly List<(Runtime.Experts.DiskExpertTensor Tensor, int[] Experts)> _diskHeld = [];
    /// <summary>
    /// With <see cref="CompileOptions.SparseExpertSlots"/>: the expert ids input (rewritten to slots), the experts of the bank and the
    /// per-expert tables the segment gathers rows of by those ids (e.g. the experts' biases).
    /// </summary>
    private readonly SparseExpertLayout? _sparse;
    /// <summary>The experts the last <see cref="SetInput"/> of the ids selected, in slot order (their weights are copied in at the next execution).</summary>
    private readonly SparseExpertSelection? _sparseSelection;
    /// <summary>The copies of the selected experts of an execution (destination, source, bytes, from page-locked memory), reused.</summary>
    private readonly List<(ulong Destination, ulong Source, nuint Bytes, bool Locked)> _sparseCopies = [];
    /// <summary>Most copies in one batched call.</summary>
    private const int MaxBatchedCopies = 256;
    /// <summary>
    /// The mapped expert banks of the streamed weights in their stored layout (page-locked or copied through staging) and their experts: <see cref="BeginStreaming(ReadOnlySpan{int})"/>
    /// copies only the experts a pass selects of them.
    /// </summary>
    private readonly Dictionary<BufferDecl, int> _streamedBanks = [];
    /// <summary>Per expert of the bank being copied, whether the pass selects it (reused).</summary>
    private bool[] _usedExperts = [];
    /// <summary>Recorded on the upload stream after the streamed weights' copies (<see cref="BeginStreaming()"/>).</summary>
    private nint _streamDone;
    /// <summary>Input and output buffers allocated so far by a streamed or sparse compilation (their ordinals key the shared buffers).</summary>
    private int _streamedInputs;
    private int _streamedOutputs;
    private bool _streaming;
    /// <summary>Inputs copied from an earlier segment's output buffer instead of from <see cref="SetInput"/> (<see cref="TryForwardFrom"/>).</summary>
    private readonly Dictionary<BufferDecl, (CudaBuffer Source, long SourceOffset)> _forwarded = [];
    private readonly Dictionary<string, CudaRecordedLaunch[]> _recorded = [];
    /// <summary>Executions of each recording so far, and its instantiated CUDA graph once it has repeated (see <see cref="RunStep"/>).</summary>
    private readonly Dictionary<string, (int Runs, nint Graph, nint Exec)> _graphs = [];
    /// <summary>The driver memory counted for each instantiated graph of <see cref="_graphs"/> (<see cref="CudaDevice.TryChargeDriverMemory"/>).</summary>
    private readonly Dictionary<string, long> _graphBytes = [];
    /// <summary>Steps whose graph did not fit the device (or the driver refused): they run launch by launch.</summary>
    private readonly HashSet<string> _uncaptured = [];

    /// <summary>
    /// True when the stream's previous launch was a kernel whose grid the GPU holds at once (one wave). A kernel is launched as the
    /// programmatic dependent of the previous one only when both fit in one wave: a dependent's waiting blocks take SM slots that a
    /// kernel with further waves would need, and a multi-wave dependent's early blocks are placed while the previous kernel still
    /// holds the SMs (RTX PRO 4500 decode steps at 1 / 16 / 64 sequences, mean of two runs, benchmarks.md §58.3: every kernel early:
    /// Qwen3-8B-FP8 -0.3 / +1.5 / +1.9 %, gpt-oss-20b -1.5 / -1.2 / -0.9 %; after one-wave kernels: -0.2 / -1.0 / +0.9 % and
    /// -1.1 / +0.1 / 0 %; both one wave: -0.2 / -0.4 / -0.2 % and -0.7 / +0.2 / +0.1 %).
    /// </summary>
    private bool _previousOneWave;

    /// <summary>
    /// Shared memory per multiprocessor of the stream's previous kernel launch at its occupancy (<see cref="CudaKernel.SharedPerBlock"/>
    /// times <see cref="CudaKernel.BlocksPerMultiprocessor"/>): the shared share of the L1 / shared split its blocks run with. A kernel is
    /// launched as its programmatic dependent only when the blocks it places on one multiprocessor (one wave spread over all of them)
    /// fit that share: the dependent's blocks are placed while the previous kernel's last blocks run, and a multiprocessor whose split
    /// has no room for them cannot take them, so they pile up on the few idle multiprocessors (RTX PRO 4500, bench/s99_pdl_placement.cu:
    /// 32 blocks of 4.2 KB after 108 blocks without shared memory land 6 per multiprocessor on 3-4 of them and take 6x as long; after
    /// 108 blocks of 1 KB they spread one per multiprocessor; benchmarks.md section 99.1). A dependent is also launched early when the
    /// two grids fit the multiprocessors at one block each (<see cref="_previousBlocks"/>: its blocks find idle multiprocessors; counted
    /// at the dependent's full occupancy instead, bert-base b1 gives back 3 %), and so is a kernel with an independent prologue
    /// (<see cref="CudaKernel.HasIndependentPrologue"/>: the FP8 GEMMs' weight loads before their wait), whose overlap outweighs the
    /// placement (Qwen3-8B-FP8 decode at 16 sequences 0.5 % faster).
    /// </summary>
    private long _previousShared;

    /// <summary>
    /// Shared memory per multiprocessor (driver reservation included) a dependent's blocks always find next to a running kernel, even one
    /// without shared memory: twice the driver's reservation per block (<c>SHARPLIGHT_CUDA_PDL_SHARED_FLOOR</c> overrides it). RTX PRO 4500,
    /// bench/s99_pdl_placement.cu, 16 blocks after 108 / 180 / 328 blocks without shared memory: up to 1.5 KB of dynamic shared memory
    /// per block (2.5 KB with the reservation) they spread one per multiprocessor, from 2 KB (3 KB) they pile up (benchmarks.md
    /// section 100.3). It admits the one-row norms (1 KB of reduction scratch) after full-GPU reductions.
    /// </summary>
    private long DependentSharedFloor => DiagnosticSwitches.DependentSharedFloor ?? 2L * _dev.Properties.ReservedSharedMemoryPerBlock;

    /// <summary>Blocks of the stream's previous kernel launch (see <see cref="_previousShared"/>).</summary>
    private long _previousBlocks;
    /// <summary>When each recording last ran (a counter of executions): the least recently run one is evicted first.</summary>
    private readonly Dictionary<string, long> _lastExecuted = [];
    private long _executions;
    private readonly nint _stream;
    /// <summary>Whether the stream's driver memory is counted as the device's (<see cref="CudaDevice.ChargeDriverMemory"/>).</summary>
    private readonly bool _streamCharged;

    /// <summary>Barrier placement of each recorded launch set (see <see cref="HazardTracker.Order(IReadOnlyList{KernelLaunch}, Bindings, bool, Dictionary{string, ValueTuple{int, bool}[]}, int)"/>).</summary>
    private readonly Dictionary<string, (int Index, bool BarrierBefore)[]> _orders = [];
    private readonly nint _start;
    private readonly nint _end;
    /// <summary>Pinned host copy of the runtime scalar slots, uploaded at the start of every execution.</summary>
    private readonly uint* _runtimeSlots;
    /// <summary>Whether <see cref="_runtimeSlots"/> is page-locked (see <see cref="CudaDevice.AllocateHost"/>).</summary>
    private readonly bool _runtimeSlotsPinned;
    /// <summary>Page-locked copies of the small inputs and outputs (asynchronous copies on the segment's stream).</summary>
    private readonly CudaHostStaging _staging;
    /// <summary>Inputs uploaded from <see cref="_staging"/> since the last execution (a second upload of one waits for the first).</summary>
    private readonly HashSet<Value> _stagedInputs = [];
    /// <summary>Outputs the last execution downloads into <see cref="_staging"/>, with their byte counts.</summary>
    private readonly Dictionary<Value, long> _stagedOutputs = [];
    private static readonly HashSet<Value> s_noSkips = [];
    private int _disposed;

    internal CudaSegment(CudaDevice dev, GraphSegment segment, KernelProgram program, CompileOptions options)
    {
        _dev = dev;
        _program = program;
        _options = options;
        Segment = segment;
        try
        {
            dev.Enter();
            dev.ChargeDriverMemory(CudaDevice.StreamBytes, $"the stream of segment {segment.Index}");
            _streamCharged = true;
            nint stream;
            int created = CudaApi.cuStreamCreate(&stream, CudaApi.StreamNonBlocking);
            if (created != 0)
            {
                throw created == CudaApi.ErrorOutOfMemory
                    ? dev.OutOfMemory(CudaDevice.StreamBytes, $"the stream of segment {segment.Index}", "cuStreamCreate: out of memory.")
                    : new CudaException(created, $"CUDA cuStreamCreate failed: {CudaApi.ErrorText(created)}");
            }

            _stream = stream;
            nint start;
            nint end;
            CudaApi.Check(CudaApi.cuEventCreate(&start, CudaApi.EventDefault), "cuEventCreate");
            _start = start;
            CudaApi.Check(CudaApi.cuEventCreate(&end, CudaApi.EventDefault), "cuEventCreate");
            _end = end;
            _runtimeSlots = (uint*)dev.AllocateHost(RuntimeSlots * sizeof(uint), out _runtimeSlotsPinned);
            _staging = new CudaHostStaging(dev);
            new Span<uint>(_runtimeSlots, RuntimeSlots).Clear();

            if (options.StreamWeights && options.SparseExpertSlots > 0)
            {
                _sparse = SparseExpertLayout.Of(segment);
                _sparseSelection = new SparseExpertSelection(_sparse.Experts, options.SparseExpertSlots);
            }

            long bytes = 0;
            foreach (var b in program.Buffers)
            {
                bytes += AllocateBuffer(b);
            }

            DeviceMemoryBytes = bytes;
            var uploadTimer = Stopwatch.StartNew();
            if (_sparse is not null)
            {
                PrepareSparseWeights();
            }
            else if (_streamedWeights.Count == 0)
            {
                UploadWeights();
            }
            else
            {
                RegisterStreamedSources();
            }
            if (DiagnosticSwitches.LoadTiming)
            {
                Console.Error.WriteLine($"[load] upload segment {segment.Index}: {uploadTimer.ElapsedMilliseconds} ms, {program.Buffers.Where(b => b.Kind == BufferKind.Weight).Sum(b => b.Bytes) / 1e9:F2} GB; "
                    + string.Join(", ", program.Buffers.Where(b => b.Kind != BufferKind.Weight).GroupBy(b => b.Kind).Select(g => $"{g.Key} {g.Sum(b => b.Bytes) / 1048576.0:F1} MiB")));
            }

            // Buffers are zeroed and weights copied on the device's upload stream; the segment's stream must see them.
            dev.Synchronize();
            RunLaunches(program.ResetLaunches);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    public GraphSegment Segment { get; }

    /// <inheritdoc />
    public long DeviceMemoryBytes { get; }

    /// <summary>GPU time (first to last launch) of the most recent <see cref="Execute(RunParameters)"/>, in ms (waits for it).</summary>
    public double LastGpuMs
    {
        get
        {
            WaitForCompletion();
            return _lastGpuMs;
        }
        private set => _lastGpuMs = value;
    }

    /// <inheritdoc />
    /// <remarks>Waits for the execution.</remarks>
    public SegmentRunTimings LastRunTimings
    {
        get
        {
            WaitForCompletion();
            return _lastRunTimings;
        }
    }

    private double _lastGpuMs;
    private SegmentRunTimings _lastRunTimings;
    private readonly object _completion = new();
    /// <summary>What the submitted, not yet waited for execution ran (for error messages); null when none is in flight.</summary>
    private string? _pending;
    /// <summary>Whether the pending execution's GPU time comes from the <see cref="_start"/> / <see cref="_end"/> events.</summary>
    private bool _pendingTimed;
    private SegmentRunTimings _pendingTimings;
    private bool _replayed;

    /// <inheritdoc />
    /// <remarks>True unless every execution is profiled (<see cref="Profile"/>) or synchronized per launch.</remarks>
    public bool ExecutesAsynchronously => !Profile && !DiagnosticSwitches.SyncEachLaunch && _program.ProducedSymbols.Count == 0;

    /// <summary>Per-launch GPU timing with events. Defaults to on with <c>SHARPLIGHT_CUDA_PROFILE=1</c>.</summary>
    public bool Profile { get; set; } = DiagnosticSwitches.Profile;

    /// <summary>Accumulated GPU time per launch name (key: "&lt;launch&gt; T=&lt;tokens&gt;").</summary>
    public Dictionary<string, (double Ms, int Count, double MinMs)> ProfileData { get; } = [];

    /// <summary>Clears <see cref="ProfileData"/>.</summary>
    public void ResetProfile() => ProfileData.Clear();

    /// <summary>Formats <see cref="ProfileData"/> as a table, heaviest first.</summary>
    /// <param name="top">Rows to list.</param>
    public string ProfileReport(int top = 30)
    {
        double total = ProfileData.Values.Sum(v => v.Ms);
        var sb = new System.Text.StringBuilder($"GPU time {total:F2} ms\n");
        foreach (var (k, v) in ProfileData.OrderByDescending(kv => kv.Value.Ms).Take(top))
        {
            sb.AppendLine($"{v.Ms,10:F3} ms {100 * v.Ms / total,5:F1}% {v.Count,6}x  min {v.MinMs * 1000,8:F1} us  {k}");
        }

        return sb.ToString();
    }

    /// <summary>What a buffer holds, for out-of-memory messages.</summary>
    private static string Purpose(BufferDecl buffer) => $"{buffer.Kind.ToString().ToLowerInvariant()} buffer {buffer.Name}";

    /// <summary>Allocates (or borrows) the buffer of <paramref name="b"/>; returns the bytes this segment newly holds.</summary>
    private long AllocateBuffer(BufferDecl b)
    {
        if (b.Kind == BufferKind.Weight && _sparse is { } sparse && _options.StateScope is { } sparseScope)
        {
            return AllocateSparseWeight(b, sparse.Experts, sparse.Tables, sparseScope);
        }

        // Streamed weights: the streamed segments of one graph on this device share the memory, by weight ordinal.
        if (b.Kind == BufferKind.Weight && _options.StreamWeights && _options.StateScope is { } streamScope)
        {
            string key = $"stream{_options.StreamBuffer}:" + _streamedWeights.Count;
            _streamedWeights.Add(b);
            return ShareArena(streamScope, key, b);
        }

        if (b.Kind == BufferKind.Weight && b.Weight is { } w && _options.SharedWeights is { } shared)
        {
            // Weights shared with the other compiled models of this source on the device (e.g. a draft head's LM head).
            var lease = shared.Acquire(_dev, WeightBufferKey.For(w, LayoutName(b)), b.Bytes, () => _dev.Allocate(b.Bytes, Purpose(b)), _dev.Free);
            _leases[b] = lease;
            _alloc[b] = lease.Buffer;
            return lease.MustFill ? b.Bytes : 0;
        }

        // The segments of one graph on this device run one after another and share their transient buffers.
        if (b.Kind == BufferKind.Activations && _options.StateScope is { } arenaScope)
        {
            return ShareArena(arenaScope, b.Name, b);
        }

        // The streamed and sparse compilations of a graph's CPU expert segments run one at a time, each from its inputs set by the host to
        // its outputs read back (no device forwarding reaches them): they share their input and output buffers too.
        if (b.Kind is BufferKind.Input or BufferKind.Output && _options.StreamWeights && _options.StateScope is { } ioScope
            && SharesStreamedIo(_program, b))
        {
            // By ordinal: the layers' value names differ, their inputs and outputs correspond.
            return ShareArena(ioScope, $"streamed-{b.Kind}:{(b.Kind == BufferKind.Input ? _streamedInputs++ : _streamedOutputs++)}", b);
        }

        // The segments of one graph on this device share each state (its nodes may be split by other devices' nodes).
        var sharedStates = b.Kind == BufferKind.State && b.State is not null && _options.StateScope is { } scope
            ? _dev.SharedStates.GetOrCreateValue(scope) : null;
        if (sharedStates is not null && sharedStates.TryGetValue(b.State!, out var existing))
        {
            _alloc[b] = existing;
            _borrowedStates.Add(b);
            return 0;
        }

        _alloc[b] = _dev.Allocate(b.Bytes, Purpose(b));
        sharedStates?.Add(b.State!, _alloc[b]);
        return b.Bytes;
    }

    /// <summary>
    /// Whether an input or output buffer of a streamed compilation can be shared with the other streamed compilations: it holds no graph input
    /// (the caller sets those for every segment before a run, so a shared buffer would hold the last segment's).
    /// </summary>
    private static bool SharesStreamedIo(KernelProgram program, BufferDecl b)
        => b.Kind == BufferKind.Output || program.InputRefs.All(kv => kv.Value.Buffer != b || kv.Key.Kind == ValueKind.Intermediate);

    /// <summary>
    /// Takes the buffer of <paramref name="b"/> (<paramref name="bytes"/>, default its size) from the arena <paramref name="key"/> of
    /// <paramref name="scope"/> on this device; returns the bytes newly allocated.
    /// </summary>
    private long ShareArena(object scope, string key, BufferDecl b, long bytes = -1)
    {
        var arenas = _dev.SharedArenas.GetOrCreateValue(scope);
        if (!arenas.TryGetValue(key, out var arena))
        {
            arenas[key] = arena = new SharedArena<CudaBuffer>();
        }

        long size = bytes < 0 ? b.Bytes : bytes;
        var sharedArena = arena.Acquire(size, n => _dev.Allocate(n, Purpose(b)), _dev.Free, out bool allocated);
        _sharedArenas[b] = sharedArena;
        _alloc[b] = sharedArena.Buffer;
        return allocated ? size : 0;
    }

    /// <summary>
    /// Allocates a weight buffer of a sparse segment: an expert bank holds <see cref="CompileOptions.SparseExpertSlots"/> experts (shared with
    /// the model's other sparse segments on the device, by bank ordinal); constant tables (decoding grids) are the segment's own. A gathered
    /// per-expert table keeps all its rows (it is small): gathers clamp the ids to the table, and the rows past the slots stay zero, so the ids
    /// of -1 (no expert) read zeros.
    /// </summary>
    /// <exception cref="NotSupportedException">A bank is repacked or not memory-mapped.</exception>
    private long AllocateSparseWeight(BufferDecl b, int experts, IReadOnlySet<WeightInfo> tables, object scope)
    {
        if (b.Weight is null)
        {
            _alloc[b] = _dev.Allocate(b.Bytes, Purpose(b));
            return b.Bytes;
        }

        nint source = b.Repack == WeightRepack.None && b.InitialData is null ? b.Weight.Source.TryGetPointer() : 0;
        var disk = b.Repack == WeightRepack.None && b.InitialData is null && !tables.Contains(b.Weight) ? _options.DiskExperts?.Find(b.Weight) : null;
        if ((source == 0 && disk is null) || b.Bytes % experts != 0 || b.Bytes != b.Weight.ByteSize)
        {
            throw new NotSupportedException($"Sparse expert streaming: {b.Name} is repacked or not memory-mapped.");
        }

        if (disk is not null)
        {
            // Copied from the expert cache's slots, never from the file's mapping.
            _diskBanks[b] = disk;
            _diskSlots[b] = new nint[disk.Experts];
            source = 0;
        }

        long expertBytes = b.Bytes / experts;
        string key = "sparse:" + _sparseWeights.Count;
        _sparseWeights.Add((b, source, expertBytes));
        return ShareArena(scope, key, b, tables.Contains(b.Weight) ? b.Bytes : _options.SparseExpertSlots * expertBytes);
    }

    /// <summary>
    /// Uploads the constant tables of a sparse segment and page-locks its expert banks' mapped sources, so the selected experts' copies are
    /// asynchronous (a bank the budget or the driver refuses is copied from pageable memory).
    /// </summary>
    private void PrepareSparseWeights()
    {
        nint done;
        CudaApi.Check(CudaApi.cuEventCreate(&done, CudaApi.EventDisableTiming), "cuEventCreate");
        _streamDone = done;
        foreach (var b in _program.Buffers.Where(b => b.Kind == BufferKind.Weight && b.Weight is null && b.InitialData is not null))
        {
            // On the upload stream after the buffer's zero fill; the wait keeps the pageable array pinned until the copy is done.
            fixed (byte* data = b.InitialData)
            {
                CudaApi.Check(CudaApi.cuMemcpyHtoDAsync(_alloc[b].Pointer, data, (nuint)b.InitialData!.Length, _dev.Stream), "cuMemcpyHtoDAsync");
                _dev.Synchronize();
            }
        }

        if (_diskBanks.Count > 0)
        {
            // The cache's slots page-locked for this device, so the selected experts' copies are asynchronous and need no staging.
            _options.DiskExperts!.PinFor(_dev, (start, bytes) => _dev.TryRegisterHost((byte*)start, bytes, committed: true), start => _dev.UnregisterHost((byte*)start));
        }

        const long Page = 4096;
        foreach (var (b, source, _) in _sparseWeights)
        {
            if (_diskBanks.ContainsKey(b))
            {
                continue;
            }

            long start = ((source + Page - 1) & ~(Page - 1)) - source;
            long end = ((source + b.Bytes) & ~(Page - 1)) - source;
            if (end - start >= Page && _dev.TryRegisterHost((byte*)source + start, end - start))
            {
                _registeredWeights.Add((b, source, start, end - start));
            }
        }
    }

    /// <summary>
    /// Holds the selected experts of the banks on disk in the expert cache (reading those it lacks; the layer's banks together) until
    /// <see cref="ReleaseDiskExperts"/>, and records their slots' addresses. Runs before the stream lock is taken: it may wait for the disk.
    /// </summary>
    private void AcquireDiskExperts()
    {
        if (_diskBanks.Count == 0)
        {
            return;
        }

        ReleaseDiskExperts();
        int[] experts = [.. _sparseSelection!.Selected];
        if (experts.Length == 0)
        {
            return;
        }

        var store = _options.DiskExperts!;
        foreach (var (b, tensor) in _diskBanks)
        {
            fixed (nint* slots = _diskSlots[b])
            {
                store.Acquire(tensor, null, experts, (byte**)slots, null, prefetchLayer: false);
            }

            _diskHeld.Add((tensor, experts));
        }
    }

    /// <summary>Lets the expert cache evict the experts <see cref="AcquireDiskExperts"/> held, once the copies from their slots completed.</summary>
    private void ReleaseDiskExperts()
    {
        if (_diskHeld.Count == 0)
        {
            return;
        }

        _dev.Enter();
        CudaApi.Check(CudaApi.cuEventSynchronize(_streamDone), "cuEventSynchronize");
        foreach (var (tensor, experts) in _diskHeld)
        {
            _options.DiskExperts!.Release(tensor, null, experts);
        }

        _diskHeld.Clear();
    }

    /// <summary>Queues the copies of the selected experts (<see cref="SparseExpertSelection.Selected"/>) of every bank into their slots on the upload stream.</summary>
    private void QueueSparseCopies()
    {
        _sparseCopies.Clear();
        foreach (var (b, source, expertBytes) in _sparseWeights)
        {
            if (_diskBanks.TryGetValue(b, out var tensor))
            {
                // One copy per expert from its cache slot (page-locked unless the budget refused the pool).
                bool pinned = _options.DiskExperts!.IsPinnedFor(_dev, tensor);
                var slots = _diskSlots[b];
                var chosen = _sparseSelection!.Selected;
                for (int slot = 0; slot < chosen.Count; slot++)
                {
                    _sparseCopies.Add((_alloc[b].Pointer + (ulong)(slot * expertBytes), (ulong)slots[chosen[slot]], (nuint)expertBytes, pinned));
                }

                continue;
            }

            ulong destination = _alloc[b].Pointer;
            // A copy must not span page-locked and pageable memory: the partial pages around the registered range go apart.
            var registered = _registeredWeights.FirstOrDefault(r => r.Buffer == b);
            long lockedStart = registered.Buffer is null ? 0 : registered.Offset;
            long lockedEnd = registered.Buffer is null ? 0 : registered.Offset + registered.Bytes;
            var selected = _sparseSelection!.Selected;
            for (int slot = 0; slot < selected.Count; slot++)
            {
                // Runs of consecutive experts go in one copy.
                int run = 1;
                while (slot + run < selected.Count && selected[slot + run] == selected[slot] + run)
                {
                    run++;
                }

                long from = selected[slot] * expertBytes;
                long to = from + run * expertBytes;
                ulong target = destination + (ulong)(slot * expertBytes);
                foreach (long cut in (ReadOnlySpan<long>)[lockedStart, lockedEnd, to])
                {
                    if (cut > from && from < to)
                    {
                        long end = Math.Min(cut, to);
                        bool locked = from >= lockedStart && end <= lockedEnd;
                        _sparseCopies.Add((target, (ulong)((byte*)source + from), (nuint)(end - from), locked));
                        target += (ulong)(end - from);
                        from = end;
                    }
                }

                slot += run - 1;
            }
        }

        SubmitSparseCopies();
    }

    /// <summary>
    /// Queues <see cref="_sparseCopies"/> on the upload stream: the page-locked ones in one batched call where the driver has it (a decode
    /// step's split pass copies 3 banks x ~4 experts; one call per copy cost ~4 µs of the host's critical path each), the others one by one.
    /// </summary>
    private void SubmitSparseCopies()
    {
        int batched = 0;
        Span<ulong> destinations = stackalloc ulong[MaxBatchedCopies];
        Span<ulong> sources = stackalloc ulong[MaxBatchedCopies];
        Span<nuint> sizes = stackalloc nuint[MaxBatchedCopies];
        foreach (var (destination, source, bytes, locked) in _sparseCopies)
        {
            if (!locked || !_dev.BatchesCopies || batched == MaxBatchedCopies)
            {
                CudaApi.Check(CudaApi.cuMemcpyHtoDAsync(destination, (void*)source, bytes, _dev.Stream), "cuMemcpyHtoDAsync");
                continue;
            }

            destinations[batched] = destination;
            sources[batched] = source;
            sizes[batched] = bytes;
            batched++;
        }

        if (batched == 0)
        {
            return;
        }

        var attributes = new MemcpyAttributes { SourceAccessOrder = CudaApi.MemcpySourceAccessOrderStream };
        nuint attributeIndex = 0;
        int result;
        fixed (ulong* d = destinations, s = sources)
        fixed (nuint* n = sizes)
        {
            result = CudaApi.cuMemcpyBatchAsync(d, s, n, (nuint)batched, &attributes, &attributeIndex, 1, _dev.Stream);
        }

        if (result == 0)
        {
            return;
        }

        // A driver without batched copies of this form: copy one by one from now on.
        _dev.RefuseBatchedCopies();
        for (int i = 0; i < batched; i++)
        {
            CudaApi.Check(CudaApi.cuMemcpyHtoDAsync(destinations[i], (void*)sources[i], sizes[i], _dev.Stream), "cuMemcpyHtoDAsync");
        }
    }

    /// <summary>
    /// Page-locks the mapped source of every streamed weight stored as is (no repack), the whole pages inside it, so that
    /// <see cref="BeginStreaming()"/> copies it asynchronously; weights the budget or the driver refuses go through staging memory.
    /// </summary>
    private void RegisterStreamedSources()
    {
        nint done;
        CudaApi.Check(CudaApi.cuEventCreate(&done, CudaApi.EventDisableTiming), "cuEventCreate");
        _streamDone = done;
        const long Page = 4096;
        var banks = StreamedExpertBanks.Of(Segment);
        foreach (var b in _streamedWeights)
        {
            nint source = b.Repack == WeightRepack.None && b.InitialData is null ? b.Weight?.Source.TryGetPointer() ?? 0 : 0;
            if (source == 0)
            {
                continue;
            }

            if (banks.TryGetValue(b.Weight!, out int experts) && b.Bytes == b.Weight!.ByteSize && b.Bytes % experts == 0)
            {
                _streamedBanks[b] = experts;
            }

            long start = ((source + Page - 1) & ~(Page - 1)) - source;
            long end = ((source + b.Bytes) & ~(Page - 1)) - source;
            if (end - start >= Page && _dev.TryRegisterHost((byte*)source + start, end - start))
            {
                _registeredWeights.Add((b, source, start, end - start));
            }
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// Weights in page-locked mapped memory are queued as asynchronous copies on the upload stream (they overlap the work running on the
    /// other streams); the others pass through the device's staging buffers, which takes the calling thread.
    /// </remarks>
    public void BeginStreaming()
    {
        if (_sparse is not null && !_streaming)
        {
            AcquireDiskExperts();
            _dev.Enter();
            lock (_dev.StreamLock)
            {
                QueueSparseCopies();
                CudaApi.Check(CudaApi.cuEventRecord(_streamDone, _dev.Stream), "cuEventRecord");
            }

            _streaming = true;
            return;
        }

        QueueStreamedCopies([]);
    }

    /// <inheritdoc />
    /// <remarks>True when an expert bank of the streamed weights is mapped in its stored layout (its experts are copied apart).</remarks>
    public bool StreamsSelectedExperts => _sparse is null && _streamedBanks.Count > 0;

    /// <inheritdoc />
    /// <remarks>Repacked banks and every other weight are copied whole.</remarks>
    public void BeginStreaming(ReadOnlySpan<int> expertIds)
    {
        if (!StreamsSelectedExperts)
        {
            BeginStreaming();
            return;
        }

        QueueStreamedCopies(expertIds, selected: true);
    }

    /// <summary>
    /// Queues the copies of the streamed weights on the upload stream: with <paramref name="selected"/>, of the expert banks
    /// (<see cref="_streamedBanks"/>) only the experts <paramref name="expertIds"/> names, in runs of consecutive experts.
    /// </summary>
    private void QueueStreamedCopies(ReadOnlySpan<int> expertIds, bool selected = false)
    {
        if (_streamedWeights.Count == 0 || _streaming || WeightsResident)
        {
            return;
        }

        var timer = Stopwatch.StartNew();
        long selectedBytes = 0;
        int usedExperts = 0;
        _dev.Enter();
        lock (_dev.StreamLock)
        {
            var registered = _registeredWeights.Select(r => r.Buffer).ToHashSet();
            var staged = _streamedWeights.Where(b => !registered.Contains(b)).ToList();
            if (staged.Count > 0)
            {
                long largestRow = staged.Where(b => b.Repack != WeightRepack.None).Select(b => WeightRepacker.RowBytes(b.Repack, b.RepackRowElements)).DefaultIfEmpty(1).Max();
                var uploader = _dev.StreamUploader(largestRow);
                foreach (var buffer in staged)
                {
                    if (selected && _streamedBanks.TryGetValue(buffer, out int experts))
                    {
                        usedExperts = MarkUsedExperts(expertIds, experts);
                        selectedBytes += UploadSelectedExperts(uploader, buffer, buffer.Bytes / experts);
                        continue;
                    }

                    uploader.Upload(buffer, _alloc[buffer]);
                }
            }

            _sparseCopies.Clear();
            foreach (var (buffer, source, offset, bytes) in _registeredWeights)
            {
                if (selected && _streamedBanks.TryGetValue(buffer, out int experts))
                {
                    usedExperts = MarkUsedExperts(expertIds, experts);
                    selectedBytes += QueueSelectedExperts(buffer, source, offset, offset + bytes, buffer.Bytes / experts);
                    continue;
                }

                // The partial pages around the registered range are copied from pageable memory, still on the upload stream.
                AddCopies(_alloc[buffer].Pointer, source, 0, buffer.Bytes, offset, offset + bytes);
                selectedBytes += buffer.Bytes;
            }

            SubmitSparseCopies();
            CudaApi.Check(CudaApi.cuEventRecord(_streamDone, _dev.Stream), "cuEventRecord");
        }

        _streaming = true;
        if (DiagnosticSwitches.LoadTiming)
        {
            long total = _streamedWeights.Sum(b => b.Bytes);
            string experts = selected ? $", {usedExperts} experts of {_streamedBanks.Values.Max()} selected" : "";
            Console.Error.WriteLine($"[stream] segment {Segment.Index}: {total / 1e9:F2} GB queued in {timer.Elapsed.TotalMilliseconds:F1} ms ({selectedBytes / 1e9:F2} GB copied{experts})");
        }
    }

    /// <summary>Sets <see cref="_usedExperts"/> to the experts of a bank of <paramref name="experts"/> that <paramref name="expertIds"/> names; returns their count.</summary>
    private int MarkUsedExperts(ReadOnlySpan<int> expertIds, int experts)
    {
        if (_usedExperts.Length < experts)
        {
            _usedExperts = new bool[experts];
        }

        var used = _usedExperts.AsSpan(0, experts);
        used.Clear();
        int count = 0;
        foreach (int id in expertIds)
        {
            if ((uint)id < (uint)experts && !used[id])
            {
                used[id] = true;
                count++;
            }
        }

        return count;
    }

    /// <summary>
    /// Adds the copies of the experts <see cref="MarkUsedExperts"/> marked of a bank to <see cref="_sparseCopies"/>, at their places in the
    /// bank's memory; returns the bytes.
    /// </summary>
    private long QueueSelectedExperts(BufferDecl bank, nint source, long lockedStart, long lockedEnd, long expertBytes)
    {
        int experts = (int)(bank.Bytes / expertBytes);
        long bytes = 0;
        for (int e = 0; e < experts; e++)
        {
            if (!_usedExperts[e])
            {
                continue;
            }

            int run = 1;
            while (e + run < experts && _usedExperts[e + run])
            {
                run++;
            }

            AddCopies(_alloc[bank].Pointer, source, e * expertBytes, (e + run) * expertBytes, lockedStart, lockedEnd);
            bytes += run * expertBytes;
            e += run - 1;
        }

        return bytes;
    }

    /// <summary>Copies the experts <see cref="MarkUsedExperts"/> marked of a bank through the staging buffers; returns the bytes.</summary>
    private long UploadSelectedExperts(CudaWeightUploader uploader, BufferDecl bank, long expertBytes)
    {
        int experts = (int)(bank.Bytes / expertBytes);
        long bytes = 0;
        for (int e = 0; e < experts; e++)
        {
            int run = 0;
            while (e + run < experts && _usedExperts[e + run])
            {
                run++;
            }

            if (run > 0)
            {
                uploader.Upload(bank, _alloc[bank], e * expertBytes, (e + run) * expertBytes);
                bytes += run * expertBytes;
                e += run;
            }
        }

        return bytes;
    }

    /// <summary>
    /// Adds the copies of bytes [<paramref name="from"/>, <paramref name="to"/>) of a weight from <paramref name="source"/> to the same offsets
    /// at <paramref name="destination"/> to <see cref="_sparseCopies"/>: a copy must not span page-locked and pageable memory, so the range is
    /// cut at the registered range [<paramref name="lockedStart"/>, <paramref name="lockedEnd"/>).
    /// </summary>
    private void AddCopies(ulong destination, nint source, long from, long to, long lockedStart, long lockedEnd)
    {
        foreach (long cut in (ReadOnlySpan<long>)[lockedStart, lockedEnd, to])
        {
            if (cut > from && from < to)
            {
                long end = Math.Min(cut, to);
                bool locked = from >= lockedStart && end <= lockedEnd;
                _sparseCopies.Add((destination + (ulong)from, (ulong)((byte*)source + from), (nuint)(end - from), locked));
                from = end;
            }
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// False also drops a copy <see cref="BeginStreaming()"/> queued that no execution consumed: the memory has been overwritten since
    /// (another streamed segment of the memory copied in after it), so the next <see cref="BeginStreaming()"/> copies again. The copies of
    /// the memory all go on the upload stream, so the new one lands after the dropped one.
    /// </remarks>
    public bool WeightsResident
    {
        get => _weightsResident;
        set
        {
            _weightsResident = value;
            if (!value && _sparse is null)
            {
                _streaming = false;
            }
        }
    }

    private bool _weightsResident;

    /// <inheritdoc />
    /// <remarks>
    /// A sparse compilation copies asynchronously when the mapped sources of all its expert banks are page-locked (copies from pageable
    /// memory return only after the driver staged them); the small per-expert tables may stay pageable.
    /// </remarks>
    public bool CopiesAsynchronously => _sparse is not { } sparse
        || _sparseWeights.All(w => sparse.Tables.Contains(w.Buffer.Weight!) || _registeredWeights.Any(r => r.Buffer == w.Buffer)
            || (_diskBanks.TryGetValue(w.Buffer, out var disk) && _options.DiskExperts!.IsPinnedFor(_dev, disk)));

    /// <summary>
    /// Makes this segment's stream wait for the streamed weights' copies, starting them when <see cref="BeginStreaming()"/> did not and the
    /// weights are not <see cref="WeightsResident"/> (the wait then is for their last copy, normally long done).
    /// </summary>
    private void StreamWeights()
    {
        BeginStreaming();
        CudaApi.Check(CudaApi.cuStreamWaitEvent(_stream, _streamDone, 0), "cuStreamWaitEvent");
        _streaming = false;
    }

    /// <summary>Device layout of a weight buffer in <see cref="WeightBufferKey"/>: empty for the bytes as stored, else the repack.</summary>
    private static string LayoutName(BufferDecl b) => b.Repack == WeightRepack.None ? "" : $"cuda:{b.Repack}:{b.RepackRowElements}";

    /// <summary>Copies the weights and constants into their buffers through pinned staging memory.</summary>
    private void UploadWeights()
    {
        var weights = _program.Buffers.Where(b => b.Kind == BufferKind.Weight && (!_leases.TryGetValue(b, out var l) || l.MustFill)).ToList();
        var ct = _options.CancellationToken;
        if (weights.Count > 0)
        {
            var materialized = WeightCopier.MaterializeInBackground(weights, ct);
            var progress = new LoadProgressCounter(_options.Progress, LoadStage.Uploading, weights.Sum(b => b.Bytes), _dev.Info.Name);
            try
            {
                long largestRow = weights.Where(b => b.Repack != WeightRepack.None).Select(b => WeightRepacker.RowBytes(b.Repack, b.RepackRowElements)).DefaultIfEmpty(1).Max();
                using var uploader = new CudaWeightUploader(_dev, weights.Sum(b => b.Bytes), largestRow);
                foreach (var buffer in weights)
                {
                    ct.ThrowIfCancellationRequested();
                    uploader.Upload(buffer, _alloc[buffer]);
                    progress.Add(buffer.Bytes);
                }
            }
            finally
            {
                try
                {
                    materialized.Wait(CancellationToken.None);
                }
                catch (AggregateException) when (ct.IsCancellationRequested)
                {
                    // The background reads stopped because of the cancellation.
                }
            }
        }

        foreach (var lease in _leases.Values)
        {
            if (lease.MustFill)
            {
                lease.Filled();
            }
        }

        foreach (var lease in _leases.Values)
        {
            lease.WaitFilled(ct);
        }
    }

    // ---- I/O ---------------------------------------------------------------------------------
    /// <inheritdoc />
    /// <exception cref="ArgumentException">The data exceeds the size compiled for (see <see cref="CompileOptions.MaxTokens"/>).</exception>
    public void SetInput(Value value, ReadOnlySpan<byte> data)
    {
        if (!_program.InputRefs.TryGetValue(value, out var r))
        {
            return;
        }

        WaitForCompletion();
        if (_sparse is { } sparse && value == sparse.Ids)
        {
            data = _sparseSelection!.Remap(data);
        }

        var a = _alloc[r.Buffer];
        if (data.Length > a.Bytes - r.ByteOffset)
        {
            throw new ArgumentException($"Input {value.Name}: {data.Length} bytes exceeds the compiled maximum ({r.Buffer.Bytes}). Increase CompileOptions.MaxTokens.");
        }

        if (data.IsEmpty)
        {
            return;
        }

        _dev.Enter();
        // The input buffer may be memory another segment's pending execution uses (shared activations); the copies below are ordered on
        // this segment's stream only.
        if (_dev.Pending is { } other && other != this)
        {
            other.WaitForCompletion();
        }

        long capacity = 0;
        byte* staged = _stagedInputs.Contains(value) ? null : _staging.Get(value, r.MaxBytes, out capacity);
        if (staged != null && data.Length <= capacity)
        {
            data.CopyTo(new Span<byte>(staged, data.Length));
            CudaApi.Check(CudaApi.cuMemcpyHtoDAsync(a.Pointer + (ulong)r.ByteOffset, staged, (nuint)data.Length, _stream), "cuMemcpyHtoDAsync");
            _stagedInputs.Add(value);
            return;
        }

        // Never a copy on the legacy stream (cuMemcpyHtoD): it waits for every blocking stream of the context and fails while another
        // thread captures a graph on one. The copy goes on this segment's stream after its staged copies, and the wait keeps the caller's
        // pageable memory free to reuse on return.
        fixed (byte* source = data)
        {
            CudaApi.Check(CudaApi.cuMemcpyHtoDAsync(a.Pointer + (ulong)r.ByteOffset, source, (nuint)data.Length, _stream), "cuMemcpyHtoDAsync");
            Synchronize($"the upload of input {value.Name}");
        }
    }

    /// <inheritdoc />
    /// <remarks>The producer's output buffer is copied into the input buffer on the device at the start of each execution.</remarks>
    public bool TryForwardFrom(Value input, ICompiledSegment producer)
    {
        if (producer is not CudaSegment source || source._dev != _dev
            || !_program.InputRefs.TryGetValue(input, out var r) || !source._program.OutputRefs.TryGetValue(input, out var o))
        {
            return false;
        }

        _forwarded[r.Buffer] = (source._alloc[o.Buffer], o.ByteOffset - r.ByteOffset);
        return true;
    }

    /// <inheritdoc />
    public void GetOutput(Value value, Span<byte> destination) => GetOutput(value, 0, destination);

    /// <inheritdoc />
    public void GetOutput(Value value, long byteOffset, Span<byte> destination)
    {
        if (destination.IsEmpty)
        {
            return;
        }

        var r = _program.OutputRefs[value];
        WaitForCompletion();
        if (_stagedOutputs.TryGetValue(value, out long staged) && byteOffset + destination.Length <= staged)
        {
            new ReadOnlySpan<byte>(_staging.Get(value, r.MaxBytes, out _) + byteOffset, destination.Length).CopyTo(destination);
            return;
        }

        _dev.Enter();
        // The output buffer may be memory another segment's pending execution writes (shared activations); the copy is ordered on this
        // segment's stream only (never on the legacy stream, see SetInput).
        if (_dev.Pending is { } other && other != this)
        {
            other.WaitForCompletion();
        }

        fixed (byte* target = destination)
        {
            CudaApi.Check(CudaApi.cuMemcpyDtoHAsync(target, _alloc[r.Buffer].Pointer + (ulong)(r.ByteOffset + byteOffset), (nuint)destination.Length, _stream), "cuMemcpyDtoHAsync");
            Synchronize($"the download of output {value.Name}");
        }
    }

    // ---- execution ---------------------------------------------------------------------------
    /// <inheritdoc />
    public void Execute(RunParameters parameters) => Execute(parameters, s_noSkips);

    /// <inheritdoc />
    /// <remarks>The small outputs not in <paramref name="skipDownloads"/> are downloaded after the launches, on the stream (see <see cref="CudaHostStaging"/>).</remarks>
    public void Execute(RunParameters parameters, IReadOnlySet<Value> skipDownloads)
    {
        foreach (var (name, slot) in _program.ScalarSlots)
        {
            _runtimeSlots[slot] = parameters.Scalars.TryGetValue(name, out var v) ? (uint)v
                : throw new ArgumentException($"Runtime scalar '{name}' not provided.");
        }

        WaitForCompletion();
        if (_dev.Pending is { } other && other != this)
        {
            other.WaitForCompletion();   // its outputs may be this segment's forwarded inputs or shared state
        }

        _stagedInputs.Clear();
        _stagedOutputs.Clear();
        long started = Stopwatch.GetTimestamp();
        _dev.Enter();
        if (_streamedWeights.Count > 0 || _sparse is not null)
        {
            StreamWeights();
        }

        var bindings = new Bindings(RecordingSymbols(parameters));
        var downloads = Downloads(bindings, skipDownloads);
        var key = RecordingKey(bindings, downloads);
        long recordStart = Stopwatch.GetTimestamp();
        bool recorded = false;
        if (!_recorded.TryGetValue(key, out var recording))
        {
            QueueInputCopies(bindings);
            recorded = true;
            CheckWithinCompiledBounds(bindings);
            EvictOldestRecording();
            // A shape's first run is launched launch by launch anyway: each launch goes out as soon as it is resolved, so the device
            // starts while the rest is still being recorded (mixed serving steps of a new row count record ~700 launches).
            recording = Record(_program.Launches, bindings, launch: !Profile);
            _recorded[key] = recording;
            if (DiagnosticSwitches.Trace)
            {
                Console.Error.WriteLine($"[cuda] recorded {key}: {recording.Length} launches");
            }
        }

        _lastExecuted[key] = ++_executions;
        long recordEnd = Stopwatch.GetTimestamp();
        _replayed = false;
        double recordMs = recorded ? Stopwatch.GetElapsedTime(recordStart, recordEnd).TotalMilliseconds : 0;
        if (Profile)
        {
            if (!recorded)
            {
                QueueInputCopies(bindings);
            }

            long profiledStart = Stopwatch.GetTimestamp();
            RunProfiled(recording, key);
            _lastRunTimings = new SegmentRunTimings(_lastGpuMs, recordMs, Stopwatch.GetElapsedTime(started, profiledStart).TotalMilliseconds - recordMs,
                Stopwatch.GetElapsedTime(profiledStart).TotalMilliseconds, recorded, false);
            PublishProducedSymbols(parameters);
            return;
        }

        if (recorded)
        {
            _graphs.TryGetValue(key, out var first);
            _graphs[key] = first with { Runs = first.Runs + 1 };
            CudaApi.Check(CudaApi.cuEventRecord(_end, _stream), "cuEventRecord");
            QueueDownloads(downloads);
            _pendingTimed = true;
            _pending = "the launches of segment " + Segment.Index;
        }
        else
        {
            RunStep(recording, key, bindings, downloads);
        }

        foreach (var (value, _, bytes) in downloads)
        {
            _stagedOutputs[value] = bytes;
        }

        _pendingTimings = new SegmentRunTimings(0, recordMs, Stopwatch.GetElapsedTime(started).TotalMilliseconds - recordMs, 0, recorded, _replayed);
        _dev.Pending = this;
        if (_program.ProducedSymbols.Count > 0 || DiagnosticSwitches.SyncEachLaunch)
        {
            WaitForCompletion();   // the counts are read back now
        }

        PublishProducedSymbols(parameters);
    }

    /// <summary>
    /// The small outputs the caller reads (not in <paramref name="skipDownloads"/>) that fit their staging buffers, each with the buffer and
    /// the bytes the execution downloads into it after its launches.
    /// </summary>
    private List<(Value Value, nint Staging, long Bytes)> Downloads(Bindings bindings, IReadOnlySet<Value> skipDownloads)
    {
        var downloads = new List<(Value, nint, long)>(_program.OutputRefs.Count);
        foreach (var (value, r) in _program.OutputRefs)
        {
            if (skipDownloads.Contains(value))
            {
                continue;
            }

            long bytes = Math.Min(UsedBytes(r, bindings), _alloc[r.Buffer].Bytes - r.ByteOffset);
            long capacity = 0;
            byte* staged = bytes > 0 ? _staging.Get(value, r.MaxBytes, out capacity) : null;
            if (staged != null && bytes <= capacity)
            {
                downloads.Add((value, (nint)staged, bytes));
            }
        }

        return downloads;
    }

    /// <summary>The recording key's part for the downloads: the indices of the outputs downloaded, when that is not every output.</summary>
    /// <summary>The key of a step's recording: its symbol bindings and, since the downloads are part of a step's graph, the outputs it skips.</summary>
    private string RecordingKey(Bindings bindings, List<(Value Value, nint Staging, long Bytes)> downloads)
        => string.Join(",", _program.Symbols.Select(s => $"{s}={bindings[s]}")) + DownloadKey(downloads);

    /// <inheritdoc />
    /// <remarks>
    /// Records the step's launches (variants, grids, arguments, barrier placement) without launching them, kept like those of executed
    /// steps: the first execution of the shape then launches the recording at once instead of selecting every launch's variant first
    /// (2-5 ms of an 8B model's step), the second captures its graph.
    /// </remarks>
    public void Prepare(RunParameters parameters, IReadOnlySet<Value> skipDownloads)
    {
        if (Profile)
        {
            return;
        }

        var bindings = new Bindings(RecordingSymbols(parameters));
        var key = RecordingKey(bindings, Downloads(bindings, skipDownloads));
        if (_recorded.ContainsKey(key))
        {
            return;
        }

        CheckWithinCompiledBounds(bindings);
        EvictOldestRecording();
        _dev.Enter();
        _recorded[key] = Record(_program.Launches, bindings);
        _lastExecuted[key] = ++_executions;
    }

    private string DownloadKey(List<(Value Value, nint Staging, long Bytes)> downloads)
    {
        if (downloads.Count == _program.OutputRefs.Count)
        {
            return "";
        }

        var downloaded = downloads.Select(d => d.Value).ToHashSet();
        return "|" + string.Join(",", _program.OutputRefs.Keys.Select((v, i) => (v, i)).Where(p => downloaded.Contains(p.v)).Select(p => p.i));
    }

    /// <summary>Queues the downloads into the staging buffers, on the stream after the launches.</summary>
    private void QueueDownloads(List<(Value Value, nint Staging, long Bytes)> downloads)
    {
        foreach (var (value, staging, bytes) in downloads)
        {
            var r = _program.OutputRefs[value];
            CudaApi.Check(CudaApi.cuMemcpyDtoHAsync((void*)staging, _alloc[r.Buffer].Pointer + (ulong)r.ByteOffset, (nuint)bytes, _stream), "cuMemcpyDtoHAsync");
        }
    }

    /// <summary>
    /// Queues the copies an execution makes before its launches: the runtime scalars from their page-locked slots, then the inputs forwarded
    /// from other segments' outputs (a step's graph holds them, see <see cref="RunStep"/>).
    /// </summary>
    private void QueueInputCopies(Bindings bindings)
    {
        if (_program.ScalarSlots.Count > 0)
        {
            CudaApi.Check(CudaApi.cuMemcpyHtoDAsync(_alloc[_program.RuntimeBuffer].Pointer, _runtimeSlots, RuntimeSlots * sizeof(uint), _stream), "cuMemcpyHtoDAsync");
        }

        CopyForwardedInputs(bindings);
    }

    /// <summary>
    /// Waits for the submitted execution, if any: checks it for errors and records its GPU time and the host's wait
    /// (<see cref="LastRunTimings"/>). Thread-safe (several threads may read outputs at once).
    /// </summary>
    private void WaitForCompletion()
    {
        ReleaseDiskExperts();
        if (Volatile.Read(ref _pending) is null)
        {
            return;
        }

        lock (_completion)
        {
            if (_pending is not { } what)
            {
                return;
            }

            long start = Stopwatch.GetTimestamp();
            _dev.Enter();
            Synchronize(what);
            double waitMs = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            float ms = 0;
            if (_pendingTimed)
            {
                CudaApi.Check(CudaApi.cuEventElapsedTime(&ms, _start, _end), "cuEventElapsedTime");
            }

            _lastGpuMs = ms;
            _lastRunTimings = _pendingTimings with { GpuMs = ms, WaitMs = waitMs };
            if (_dev.Pending == this)
            {
                _dev.Pending = null;
            }

            Volatile.Write(ref _pending, null);
        }
    }

    /// <summary>
    /// The caller's symbol bindings, plus the data-dependent symbols this segment computes at their compile-time bounds: only the
    /// producing launch's own output depends on them, so recording them at the bound keeps one recording for every count.
    /// </summary>
    private IReadOnlyDictionary<string, long> RecordingSymbols(RunParameters parameters)
    {
        if (_program.ProducedSymbols.Count == 0)
        {
            return parameters.Symbols;
        }

        var symbols = new Dictionary<string, long>(parameters.Symbols);
        foreach (var symbol in _program.ProducedSymbols.Keys)
        {
            symbols[symbol] = _options.Bound(symbol);
        }

        return symbols;
    }

    /// <summary>Reads the data-dependent counts computed by the execution and binds them for the later segments and the caller.</summary>
    private void PublishProducedSymbols(RunParameters parameters)
    {
        foreach (var (symbol, slot) in _program.ProducedSymbols)
        {
            int count;
            CudaApi.Check(CudaApi.cuMemcpyDtoHAsync(&count, _alloc[slot.Buffer].Pointer + (ulong)slot.ByteOffset, sizeof(int), _stream), "cuMemcpyDtoHAsync");
            Synchronize($"the count {symbol} of segment {Segment.Index}");
            parameters.Symbols[symbol] = Math.Clamp(count, 0, _options.Bound(symbol));
        }
    }

    private void CopyForwardedInputs(Bindings bindings)
    {
        foreach (var r in _program.InputRefs.Values)
        {
            if (!_forwarded.TryGetValue(r.Buffer, out var f))
            {
                continue;
            }

            long size = Math.Min(UsedBytes(r, bindings), _alloc[r.Buffer].Bytes - r.ByteOffset);
            if (size > 0)
            {
                ulong source = f.Source.Pointer + (ulong)(r.ByteOffset + f.SourceOffset);
                CudaApi.Check(CudaApi.cuMemcpyDtoDAsync(_alloc[r.Buffer].Pointer + (ulong)r.ByteOffset, source, (nuint)size, _stream), "cuMemcpyDtoDAsync");
            }
        }
    }

    /// <summary>Bytes of a tensor actually used under the given symbol bindings.</summary>
    private static long UsedBytes(TensorRef r, Bindings b)
    {
        long elems = r.MaxElements;
        if (r.Symbol is not null && r.Shape.Length > 0 && r.Shape[0] > 0)
        {
            elems = elems / r.Shape[0] * Math.Min(r.Shape[0], b[r.Symbol]);
        }

        return r.Type.ByteSize(elems);
    }

    private void CheckWithinCompiledBounds(Bindings bindings)
    {
        foreach (var symbol in _program.Symbols)
        {
            if (bindings[symbol] > _options.Bound(symbol))
            {
                throw new ArgumentException($"Symbol {symbol}={bindings[symbol]} exceeds the compiled bound.");
            }
        }
    }

    /// <summary>
    /// Keeps the recording cache bounded (data-dependent symbols can take a new value on every run) by evicting the recording that ran
    /// least recently: a serving decoder's decode steps, recorded first, stay while prompt-bearing shapes come and go.
    /// </summary>
    private void EvictOldestRecording()
    {
        if (_recorded.Count < MaxRecordings)
        {
            return;
        }

        string key = _lastExecuted.MinBy(p => p.Value).Key;
        _lastExecuted.Remove(key);
        _recorded.Remove(key);
        DestroyGraph(key);
    }

    /// <summary>
    /// Runs a recorded step. With <see cref="DeviceFeatures.LaunchGraphs"/> a step recording that runs a second time is captured into
    /// a CUDA graph, and from then on replayed with one <c>cuGraphLaunch</c> instead of a host call per launch (decode steps repeat
    /// their recording: positions and lengths are runtime scalars uploaded before the launches). Steps that run once (prefill
    /// chunks of new lengths) are not captured, so they pay no instantiation.
    /// </summary>
    private void RunStep(CudaRecordedLaunch[] recording, string key, Bindings bindings, List<(Value Value, nint Staging, long Bytes)> downloads)
    {
        // A graph's scalar upload must read page-locked memory (a capture refuses pageable copies).
        bool scalarsCapturable = _program.ScalarSlots.Count == 0 || _runtimeSlotsPinned;
        if (!_dev.LaunchGraphs || DiagnosticSwitches.SyncEachLaunch || recording.Length < 2 || !scalarsCapturable)
        {
            Submit(recording, bindings, downloads, "the launches of segment " + Segment.Index);
            return;
        }

        _graphs.TryGetValue(key, out var entry);
        if (entry.Exec == 0 && entry.Runs >= 1 && !_uncaptured.Contains(key))
        {
            // The graph's driver memory is counted as the device's; when the device has no room for it, the step keeps running launch by
            // launch (a placement of many segments must not fail at its first capture).
            long bytes = CudaDevice.GraphBytes + recording.Length * CudaDevice.GraphBytesPerNode;
            if (_dev.TryChargeDriverMemory(bytes))
            {
                entry = Capture(recording, bindings, downloads, entry.Runs);
                if (entry.Exec != 0)
                {
                    _graphBytes[key] = bytes;
                }
                else
                {
                    _dev.ReleaseDriverMemory(bytes);
                    _uncaptured.Add(key);
                }
            }
            else
            {
                _uncaptured.Add(key);
            }
        }

        entry.Runs++;
        _graphs[key] = entry;
        if (entry.Exec == 0)
        {
            Submit(recording, bindings, downloads, "the launches of segment " + Segment.Index);
            return;
        }

        // The graph records the timing events itself (event record nodes around its launches).
        CudaApi.Check(CudaApi.cuGraphLaunch(entry.Exec, _stream), "cuGraphLaunch");
        _replayed = true;
        _pendingTimed = true;
        _pending = "the graph of segment " + Segment.Index;
    }

    /// <summary>
    /// Launches a recording between the timing events, after its input copies and before its downloads, without waiting for it
    /// (<see cref="WaitForCompletion"/> does).
    /// </summary>
    private void Submit(ReadOnlySpan<CudaRecordedLaunch> launches, Bindings bindings, List<(Value Value, nint Staging, long Bytes)> downloads, string what)
    {
        QueueInputCopies(bindings);
        CudaApi.Check(CudaApi.cuEventRecord(_start, _stream), "cuEventRecord");
        _previousOneWave = false;   // copies may precede the first kernel: it is never a dependent
        foreach (var launch in launches)
        {
            Launch(launch);
        }

        CudaApi.Check(CudaApi.cuEventRecord(_end, _stream), "cuEventRecord");
        QueueDownloads(downloads);
        _pendingTimed = true;
        _pending = what;
    }

    /// <summary>
    /// Captures a step into an instantiated graph (or none, when the driver refuses the capture): its input copies, the launches of its
    /// recording and its downloads, with the timing events recorded around the launches, so a replay is one <c>cuGraphLaunch</c>. The
    /// copies read the page-locked scalar slots and staging buffers, and the other segments' output buffers, at the replay.
    /// </summary>
    private (int Runs, nint Graph, nint Exec) Capture(CudaRecordedLaunch[] recording, Bindings bindings, List<(Value Value, nint Staging, long Bytes)> downloads, int runs)
    {
        // The side streams are the device's, lent for the capture (a captured graph keeps its branches, not the streams); without them (no
        // device memory left for streams) the launches are captured one after another.
        var lanes = DiagnosticSwitches.ConcurrentLaunches ? _dev.TryRentLanes() : null;
        bool lanesReusable = false;
        try
        {
            var captured = Capture(recording, bindings, downloads, runs, lanes);
            lanesReusable = true;
            return captured;
        }
        finally
        {
            if (lanes is not null)
            {
                _dev.ReturnLanes(lanes, lanesReusable);
            }
        }
    }

    private (int Runs, nint Graph, nint Exec) Capture(CudaRecordedLaunch[] recording, Bindings bindings, List<(Value Value, nint Staging, long Bytes)> downloads, int runs,
        CudaConcurrentLanes? lanes)
    {
        CudaApi.Check(CudaApi.cuStreamBeginCapture(_stream, CudaApi.CaptureModeThreadLocal), "cuStreamBeginCapture");
        _previousOneWave = false;   // the graph's first kernel follows whatever the stream ran before it
        nint graph = 0;
        try
        {
            QueueInputCopies(bindings);
            CudaApi.Check(CudaApi.cuEventRecordWithFlags(_start, _stream, CudaApi.EventRecordExternal), "cuEventRecordWithFlags");
            _previousOneWave = false;
            if (lanes is not null)
            {
                LaunchBatches(recording, lanes);
            }
            else
            {
                foreach (var launch in recording)
                {
                    Launch(launch);
                }
            }

            CudaApi.Check(CudaApi.cuEventRecordWithFlags(_end, _stream, CudaApi.EventRecordExternal), "cuEventRecordWithFlags");
            QueueDownloads(downloads);
        }
        finally
        {
            CudaApi.Check(CudaApi.cuStreamEndCapture(_stream, &graph), "cuStreamEndCapture");
        }

        nint exec;
        if (CudaApi.cuGraphInstantiateWithFlags(&exec, graph, 0) != 0)
        {
            CudaApi.cuGraphDestroy(graph);
            return (runs, 0, 0);
        }

        if (DiagnosticSwitches.Trace)
        {
            Console.Error.WriteLine($"[cuda] segment {Segment.Index}: captured a graph of {recording.Length} launches");
        }

        return (runs, graph, exec);
    }

    private void DestroyGraph(string key)
    {
        _uncaptured.Remove(key);
        if (_graphs.Remove(key, out var entry) && entry.Exec != 0)
        {
            CudaApi.cuGraphExecDestroy(entry.Exec);
            CudaApi.cuGraphDestroy(entry.Graph);
        }

        if (_graphBytes.Remove(key, out long bytes))
        {
            _dev.ReleaseDriverMemory(bytes);
        }
    }

    /// <inheritdoc />
    public void ResetState()
    {
        WaitForCompletion();
        _dev.Enter();
        RunLaunches(_program.ResetLaunches);
    }

    /// <inheritdoc/>
    public bool CanCopyStateRows => true;

    /// <inheritdoc/>
    /// <remarks>Device-to-device copies on the segment's stream, waited for (another segment's stream may read the state next).</remarks>
    public void CopyStateRows(IReadOnlyCollection<string> states, int from, int to, int count)
    {
        var regions = StateRowCopies.Regions(_program, _borrowedStates, states, from, to, count, b => _alloc[b].Bytes);
        if (regions.Count == 0)
        {
            return;
        }

        WaitForCompletion();
        if (_dev.Pending is { } other && other != this)
        {
            other.WaitForCompletion();   // the state may be another segment's (borrowed)
        }

        _dev.Enter();
        foreach (var (b, source, destination, bytes) in regions)
        {
            ulong pointer = _alloc[b].Pointer;
            CudaApi.Check(CudaApi.cuMemcpyDtoDAsync(pointer + (ulong)destination, pointer + (ulong)source, (nuint)bytes, _stream), "cuMemcpyDtoDAsync");
        }

        Synchronize("the state copies of segment " + Segment.Index);
    }

    private void RunLaunches(IReadOnlyList<KernelLaunch> launches)
    {
        if (launches.Count == 0)
        {
            return;
        }

        Run(Record(launches, new Bindings(new Dictionary<string, long>())));
    }

    /// <summary>Resolves the launches for <paramref name="bindings"/>: variant, kernel, packed arguments and grid, in program order.</summary>
    /// <param name="launches">The program's launches.</param>
    /// <param name="bindings">Symbol values.</param>
    /// <param name="launch">Also launch each recorded launch at once, after the start event (the caller records the end event).</param>
    private CudaRecordedLaunch[] Record(IReadOnlyList<KernelLaunch> launches, Bindings bindings, bool launch = false)
    {
        var recorded = new List<CudaRecordedLaunch>(launches.Count);
        if (launch)
        {
            CudaApi.Check(CudaApi.cuEventRecord(_start, _stream), "cuEventRecord");
            _previousOneWave = false;
        }

        // The order and hazard batches of the Vulkan recorder (independent launches moved into the same batch); a captured graph runs a
        // batch's launches side by side (LaunchBatches), everything else runs them in this order on one stream. Profiling keeps program order.
        bool reorder = DiagnosticSwitches.ConcurrentLaunches && !Profile;
        bool barrierPending = true;
        // The order cache holds the program's launch list only (other lists, such as state copies, are ordered afresh).
        var ordered = ReferenceEquals(launches, _program.Launches) ? HazardTracker.Order(launches, bindings, reorder, _orders, 0)
            : HazardTracker.Order(launches, bindings, reorder);
        foreach (var (item, _, variant, barrierBefore) in ordered)
        {
            barrierPending |= barrierBefore;
            var grid = variant.Grid(bindings);
            if (grid.X == 0 || grid.Y == 0 || grid.Z == 0)
            {
                continue;
            }

            var kernel = _dev.Kernel(variant.Kernel);
            recorded.Add(new CudaRecordedLaunch(item.Name, ProfileName(item.Name, variant.Kernel.EntryPoint), kernel, grid.X, grid.Y, grid.Z,
                PackArguments(item, bindings, kernel.TensorMaps), barrierPending));
            barrierPending = false;
            if (launch)
            {
                Launch(recorded[^1]);
            }
        }

        return [.. recorded];
    }

    /// <summary>
    /// The kernel's parameter struct: every pointer (8 bytes) first, then the 32-bit scalars, as the kernels declare them, then the
    /// kernel's tensor maps (<paramref name="tensorMaps"/>, 128 bytes each from the next 64-byte boundary), encoded over the pointers
    /// they describe.
    /// </summary>
    private byte[] PackArguments(KernelLaunch launch, Bindings bindings, CudaTensorMapSpec[] tensorMaps)
    {
        int pointers = launch.Args.Count(a => a is PointerArg);
        int bytes = pointers * 8 + (launch.Args.Count - pointers) * 4;
        int mapsOffset = (bytes + 63) & ~63;
        if (tensorMaps.Length > 0)
        {
            bytes = mapsOffset + tensorMaps.Length * 128;
        }

        var packed = new byte[Math.Max(16, (bytes + 15) & ~15)];
        var addresses = new ulong[pointers];
        fixed (byte* p = packed)
        {
            int offset = 0;
            foreach (var arg in launch.Args)
            {
                if (arg is PointerArg pointer)
                {
                    addresses[offset / 8] = _alloc[pointer.Buffer].Pointer + (ulong)pointer.ByteOffset;
                    *(ulong*)(p + offset) = addresses[offset / 8];
                    offset += 8;
                }
            }

            foreach (var arg in launch.Args)
            {
                switch (arg)
                {
                    case UIntArg u:
                        *(uint*)(p + offset) = u.Value;
                        offset += 4;
                        break;
                    case FloatArg f:
                        *(float*)(p + offset) = f.Value;
                        offset += 4;
                        break;
                    case SymbolArg s:
                        *(uint*)(p + offset) = (uint)(bindings[s.Symbol] * s.Multiplier);
                        offset += 4;
                        break;
                }
            }

            for (int m = 0; m < tensorMaps.Length; m++)
            {
                EncodeTensorMap(p + mapsOffset + m * 128, tensorMaps[m], addresses[tensorMaps[m].Pointer], launch.Name);
            }
        }

        return packed;
    }

    /// <summary>Encodes a tensor map of byte elements with 128-byte swizzled boxes over <paramref name="address"/>.</summary>
    private static void EncodeTensorMap(byte* map, CudaTensorMapSpec spec, ulong address, string launch)
    {
        int rank = spec.Dims.Length;
        var dims = stackalloc ulong[rank];
        var strides = stackalloc ulong[Math.Max(1, rank - 1)];
        var box = stackalloc uint[rank];
        var elementStrides = stackalloc uint[rank];
        for (int d = 0; d < rank; d++)
        {
            dims[d] = spec.Dims[d];
            box[d] = spec.Box[d];
            elementStrides[d] = 1;
            if (d > 0)
            {
                strides[d - 1] = spec.Strides[d - 1];
            }
        }

        int result = CudaApi.cuTensorMapEncodeTiled(map, CudaApi.TensorMapUInt8, (uint)rank, address, dims, strides, box, elementStrides, 0,
            CudaApi.TensorMapSwizzle128, CudaApi.TensorMapL2Promotion256, 0);
        if (result != 0)
        {
            throw new CudaException(result, $"Tensor map of {launch} (address 0x{address:x}, dims {string.Join(',', spec.Dims)}) could not be encoded: {CudaApi.ErrorText(result)}");
        }
    }

    /// <summary>Launches <paramref name="launches"/> on the segment's stream, waits for them and returns the GPU time in ms.</summary>
    private double Run(ReadOnlySpan<CudaRecordedLaunch> launches)
    {
        CudaApi.Check(CudaApi.cuEventRecord(_start, _stream), "cuEventRecord");
        _previousOneWave = false;   // copies may precede the first kernel: it is never a dependent
        foreach (var launch in launches)
        {
            Launch(launch);
        }

        CudaApi.Check(CudaApi.cuEventRecord(_end, _stream), "cuEventRecord");
        Synchronize("the launches of segment " + Segment.Index);
        float ms;
        CudaApi.Check(CudaApi.cuEventElapsedTime(&ms, _start, _end), "cuEventElapsedTime");
        return ms;
    }

    /// <summary>
    /// Launches a recording inside a stream capture, its hazard batches one after another and the launches of a batch side by side: the
    /// first on the segment's stream, the others on the side lanes forked from it and joined back before the next batch. Launches of a
    /// batch of several are never programmatic dependents (their stream predecessor is not their producer); a single-launch batch keeps
    /// the dependent launch where the previous batch was one launch too.
    /// </summary>
    private void LaunchBatches(CudaRecordedLaunch[] recording, CudaConcurrentLanes lanes)
    {
        for (int begin = 0; begin < recording.Length;)
        {
            int end = begin + 1;
            while (end < recording.Length && !recording[end].BarrierBefore)
            {
                end++;
            }

            if (end - begin == 1)
            {
                Launch(recording[begin]);
                begin = end;
                continue;
            }

            lanes.Fork(_stream);
            _previousOneWave = false;
            Launch(recording[begin]);
            for (int i = begin + 1; i < end; i++)
            {
                _previousOneWave = false;
                Launch(recording[i], lanes.Stream(i - begin - 1));
            }

            lanes.Join(_stream);
            _previousOneWave = false;
            begin = end;
        }
    }

    private void Launch(CudaRecordedLaunch launch) => Launch(launch, _stream);

    private void Launch(CudaRecordedLaunch launch, nint stream)
    {
        var kernel = launch.Kernel;
        if (kernel.Library is { } gemm)
        {
            // Library GEMM: arguments x, w, y (device addresses), then the row count.
            var args = launch.Arguments;
            _previousOneWave = false;
            _dev.Blas!.Gemm(stream, gemm.Type, gemm.N, gemm.K, BitConverter.ToUInt32(args, 24),
                BitConverter.ToUInt64(args, 0), BitConverter.ToUInt64(args, 8), BitConverter.ToUInt64(args, 16));
            if (DiagnosticSwitches.SyncEachLaunch)
            {
                Synchronize(launch.Name);
            }

            return;
        }

        fixed (byte* arguments = launch.Arguments)
        {
            void* parameter = arguments;
            long blocks = (long)launch.GridX * launch.GridY * launch.GridZ;
            bool oneWave = kernel.ResidentBlocks > 0 && blocks <= kernel.ResidentBlocks;
            int sms = _dev.Properties.Multiprocessors;
            bool dependent = kernel.WaitsForPrevious && _dev.DependentLaunches && _previousOneWave && oneWave
                             && (kernel.HasIndependentPrologue || _previousBlocks + blocks <= sms
                                 || kernel.SharedPerBlock * ((blocks + sms - 1) / sms) <= Math.Max(_previousShared, DependentSharedFloor));
            _previousOneWave = oneWave;
            _previousShared = kernel.SharedPerBlock * kernel.BlocksPerMultiprocessor;
            _previousBlocks = blocks;
            int result = dependent
                ? LaunchDependent(kernel, launch, &parameter)
                : CudaApi.cuLaunchKernel(kernel.Function, launch.GridX, launch.GridY, launch.GridZ, kernel.ThreadsX, kernel.ThreadsY, kernel.ThreadsZ,
                    (uint)kernel.SharedBytes, stream, &parameter, null);
            if (result != 0)
            {
                throw new CudaException(result, $"CUDA launch of {launch.Name} (grid {launch.GridX}x{launch.GridY}x{launch.GridZ}) failed: {CudaApi.ErrorText(result)}");
            }
        }

        if (DiagnosticSwitches.SyncEachLaunch)
        {
            Synchronize(launch.Name);
        }
    }

    /// <summary>
    /// Launches <paramref name="kernel"/> as a programmatic dependent of the previous launch in the stream (<see cref="CudaDevice.DependentLaunches"/>):
    /// it may start while that launch's last blocks run and waits for it at its entry. A driver without <c>cuLaunchKernelEx</c> switches
    /// the device back to plain launches.
    /// </summary>
    private int LaunchDependent(CudaKernel kernel, CudaRecordedLaunch launch, void** parameter)
    {
        var attribute = new CuLaunchAttribute { Id = CudaApi.LaunchAttributeProgrammaticStreamSerialization, Value = 1 };
        var config = new CuLaunchConfig
        {
            GridX = launch.GridX,
            GridY = launch.GridY,
            GridZ = launch.GridZ,
            BlockX = kernel.ThreadsX,
            BlockY = kernel.ThreadsY,
            BlockZ = kernel.ThreadsZ,
            SharedMemoryBytes = (uint)kernel.SharedBytes,
            Stream = _stream,
            Attributes = &attribute,
            AttributeCount = 1,
        };
        try
        {
            return CudaApi.cuLaunchKernelEx(&config, kernel.Function, parameter, null);
        }
        catch (EntryPointNotFoundException)
        {
            _dev.DependentLaunches = false;
            return CudaApi.cuLaunchKernel(kernel.Function, launch.GridX, launch.GridY, launch.GridZ, kernel.ThreadsX, kernel.ThreadsY, kernel.ThreadsZ,
                (uint)kernel.SharedBytes, _stream, parameter, null);
        }
    }

    private void Synchronize(string what)
    {
        int result = CudaApi.cuStreamSynchronize(_stream);
        if (result != 0)
        {
            throw new CudaException(result, $"CUDA execution of {what} failed: {CudaApi.ErrorText(result)}");
        }
    }

    /// <summary>Runs each launch between its own pair of events and accumulates its GPU time in <see cref="ProfileData"/>.</summary>
    /// <summary>
    /// The launch name as profiles show it: without the node name (its second word), with the kernel's entry point, so launches of
    /// the same kind add up as in the Vulkan provider's profile.
    /// </summary>
    private static string ProfileName(string name, string entryPoint)
    {
        var parts = name.Split(' ');
        return $"{parts[0]}{(parts.Length > 2 ? " " + string.Join(" ", parts[2..]) : "")} {entryPoint}";
    }

    private void RunProfiled(CudaRecordedLaunch[] launches, string key)
    {
        double total = 0;
        foreach (var launch in launches)
        {
            double ms = Run(new ReadOnlySpan<CudaRecordedLaunch>(in launch));
            total += ms;
            string name = $"{launch.ProfileName} {key}";
            ProfileData[name] = ProfileData.TryGetValue(name, out var entry)
                ? (entry.Ms + ms, entry.Count + 1, Math.Min(entry.MinMs, ms))
                : (ms, 1, ms);
        }

        LastGpuMs = total;
    }

    /// <summary>Number of launches of a step with the given token count (all on one stream, so no barriers).</summary>
    /// <param name="tokens">Value of every symbol.</param>
    public (int Dispatches, int Barriers) Stats(long tokens)
    {
        var bindings = new Bindings(_program.Symbols.ToDictionary(s => s, _ => tokens));
        return (HazardTracker.Order(_program.Launches, bindings, reorder: false).Count, 0);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        // Claimed before waiting for the device: a second (concurrent) call must not free the buffers and host registrations again.
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        if (!_dev.IsDisposed)
        {
            try
            {
                WaitForCompletion();
            }
            catch (CudaException)
            {
                // The failure was the execution's; the buffers are freed regardless.
            }
        }

        if (_streamDone != 0)
        {
            // Copies queued by BeginStreaming without an execution may still be running.
            ReleaseDiskExperts();
            CudaApi.cuEventSynchronize(_streamDone);
            CudaApi.cuEventDestroy(_streamDone);
        }

        foreach (var (_, source, offset, _) in _registeredWeights)
        {
            _dev.UnregisterHost((byte*)source + offset);
        }

        _recorded.Clear();
        _lastExecuted.Clear();
        foreach (var (b, a) in _alloc)
        {
            if (_sharedArenas.TryGetValue(b, out var arena))
            {
                arena.Release();
            }
            else if (!_leases.ContainsKey(b) && !_borrowedStates.Contains(b))
            {
                _dev.Free(a);
            }
        }

        foreach (var lease in _leases.Values)
        {
            lease.Dispose();
        }

        _leases.Clear();
        _alloc.Clear();
        if (_dev.IsDisposed)
        {
            return;
        }

        _dev.Enter();
        foreach (string key in _graphs.Keys.ToList())
        {
            DestroyGraph(key);
        }

        if (_runtimeSlots != null)
        {
            _dev.FreeHost((byte*)_runtimeSlots, RuntimeSlots * sizeof(uint), _runtimeSlotsPinned);
        }

        _staging?.Dispose();

        if (_start != 0)
        {
            CudaApi.cuEventDestroy(_start);
        }

        if (_end != 0)
        {
            CudaApi.cuEventDestroy(_end);
        }

        if (_stream != 0)
        {
            CudaApi.cuStreamDestroy(_stream);
        }

        if (_streamCharged)
        {
            _dev.ReleaseDriverMemory(CudaDevice.StreamBytes);
        }
    }
}
