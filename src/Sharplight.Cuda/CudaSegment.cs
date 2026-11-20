using System.Diagnostics;
using System.Runtime.InteropServices;
using Sharplight.Cuda.Interop;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Shaders.Lowering;

namespace Sharplight.Cuda;

/// <summary>
/// A compiled segment on a CUDA device: owns the device buffers of its <see cref="KernelProgram"/> and launches its kernels on a stream of
/// its own, from a recording per distinct binding of the symbolic dimensions (kernels, arguments and grids resolved once). Launches on one
/// stream run in order, so no barriers are needed. Inputs and outputs are copied synchronously. An execution returns once its work is
/// submitted (the host can prepare the next step meanwhile); the segment waits for it before it reads an output, takes an input, copies
/// state or runs again, and before another segment of the device runs (<see cref="CudaDevice.Pending"/>). Profiled executions wait at once.
/// </summary>
public sealed unsafe class CudaSegment : IStreamedSegment, IPartialOutputSegment, IDeviceForwardingSegment
{
    /// <summary>Most steps kept recorded per segment (one per distinct symbol binding).</summary>
    private const int MaxRecordings = 64;

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
    /// <summary>With <see cref="CompileOptions.SparseExpertSlots"/>: the expert ids input (rewritten to slots) and the experts of the bank.</summary>
    private readonly (Value Ids, int Experts)? _sparse;
    /// <summary>The experts the last <see cref="SetInput"/> of the ids selected, in slot order (their weights are copied in at the next execution).</summary>
    private readonly List<int> _sparseSelected = [];
    /// <summary>Recorded on the upload stream after the streamed weights' copies (<see cref="BeginStreaming"/>).</summary>
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

    /// <summary>
    /// True when the stream's previous launch was a kernel whose grid the GPU holds at once (one wave). A kernel is launched as the
    /// programmatic dependent of the previous one only when both fit in one wave: a dependent's waiting blocks take SM slots that a
    /// kernel with further waves would need, and a multi-wave dependent's early blocks are placed while the previous kernel still
    /// holds the SMs (RTX PRO 4500 decode steps at 1 / 16 / 64 sequences, mean of two runs, benchmarks.md §58.3: every kernel early:
    /// Qwen3-8B-FP8 -0.3 / +1.5 / +1.9 %, gpt-oss-20b -1.5 / -1.2 / -0.9 %; after one-wave kernels: -0.2 / -1.0 / +0.9 % and
    /// -1.1 / +0.1 / 0 %; both one wave: -0.2 / -0.4 / -0.2 % and -0.7 / +0.2 / +0.1 %).
    /// </summary>
    private bool _previousOneWave;
    private readonly Queue<string> _recordOrder = new();
    private readonly nint _stream;
    private readonly nint _start;
    private readonly nint _end;
    /// <summary>Pinned host copy of the runtime scalar slots, uploaded at the start of every execution.</summary>
    private readonly uint* _runtimeSlots;
    /// <summary>Whether <see cref="_runtimeSlots"/> is page-locked (see <see cref="CudaDevice.AllocateHost"/>).</summary>
    private readonly bool _runtimeSlotsPinned;
    private bool _disposed;

    internal CudaSegment(CudaDevice dev, GraphSegment segment, KernelProgram program, CompileOptions options)
    {
        _dev = dev;
        _program = program;
        _options = options;
        Segment = segment;
        try
        {
            dev.Enter();
            nint stream;
            CudaApi.Check(CudaApi.cuStreamCreate(&stream, 0), "cuStreamCreate");
            _stream = stream;
            nint start;
            nint end;
            CudaApi.Check(CudaApi.cuEventCreate(&start, CudaApi.EventDefault), "cuEventCreate");
            _start = start;
            CudaApi.Check(CudaApi.cuEventCreate(&end, CudaApi.EventDefault), "cuEventCreate");
            _end = end;
            _runtimeSlots = (uint*)dev.AllocateHost(RuntimeSlots * sizeof(uint), out _runtimeSlotsPinned);
            new Span<uint>(_runtimeSlots, RuntimeSlots).Clear();

            if (options.StreamWeights && options.SparseExpertSlots > 0)
            {
                _sparse = SparseExpertIds(segment);
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
            return AllocateSparseWeight(b, sparse.Experts, sparseScope);
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
        if (b.Kind == BufferKind.Activations && _options.StateScope is { } arenaScope && SharedArena<CudaBuffer>.Enabled)
        {
            return ShareArena(arenaScope, b.Name, b);
        }

        // The streamed and sparse compilations of a graph's CPU expert segments run one at a time, each from its inputs set by the host to
        // its outputs read back (no device forwarding reaches them): they share their input and output buffers too.
        if (b.Kind is BufferKind.Input or BufferKind.Output && _options.StreamWeights && _options.StateScope is { } ioScope && SharedArena<CudaBuffer>.Enabled
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
    /// The expert ids input of a segment compiled with <see cref="CompileOptions.SparseExpertSlots"/> and the experts of its banks: every weight
    /// the segment reads must be the bank of a <c>MatMulId</c> (input 1, [experts, ...]) indexed by one ids input of the segment.
    /// </summary>
    /// <exception cref="NotSupportedException">The segment reads other weights or more than one ids value.</exception>
    private static (Value Ids, int Experts) SparseExpertIds(GraphSegment segment)
    {
        Value? ids = null;
        int experts = 0;
        foreach (var n in segment.Nodes)
        {
            for (int i = 0; i < n.Inputs.Length; i++)
            {
                if (n.Inputs[i].Weight is null)
                {
                    continue;
                }

                if (n.Op.Name != "MatMulId" || i != 1 || ids is not null && n.Inputs[2] != ids || experts > 0 && n.Inputs[1].Shape[0].Fixed != experts)
                {
                    throw new NotSupportedException($"Sparse expert streaming: {n.Op.Name} reads a weight that is not an expert bank of the segment's ids (output {n.Outputs[0].Name}).");
                }

                ids = n.Inputs[2];
                experts = (int)n.Inputs[1].Shape[0].Fixed;
            }
        }

        if (ids is null || !segment.Inputs.Contains(ids) || ids.Type != Tensors.DataType.I32)
        {
            throw new NotSupportedException("Sparse expert streaming: the segment has no expert banks indexed by an i32 input.");
        }

        return (ids, experts);
    }

    /// <summary>
    /// Allocates a weight buffer of a sparse segment: an expert bank holds <see cref="CompileOptions.SparseExpertSlots"/> experts (shared with
    /// the model's other sparse segments on the device, by bank ordinal); constant tables (decoding grids) are the segment's own.
    /// </summary>
    /// <exception cref="NotSupportedException">A bank is repacked or not memory-mapped.</exception>
    private long AllocateSparseWeight(BufferDecl b, int experts, object scope)
    {
        if (b.Weight is null)
        {
            _alloc[b] = _dev.Allocate(b.Bytes, Purpose(b));
            return b.Bytes;
        }

        nint source = b.Repack == WeightRepack.None && b.InitialData is null ? b.Weight.Source.TryGetPointer() : 0;
        if (source == 0 || b.Bytes % experts != 0 || b.Bytes != b.Weight.ByteSize)
        {
            throw new NotSupportedException($"Sparse expert streaming: {b.Name} is repacked or not memory-mapped.");
        }

        long expertBytes = b.Bytes / experts;
        string key = "sparse:" + _sparseWeights.Count;
        _sparseWeights.Add((b, source, expertBytes));
        return ShareArena(scope, key, b, _options.SparseExpertSlots * expertBytes);
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
            fixed (byte* data = b.InitialData)
            {
                CudaApi.Check(CudaApi.cuMemcpyHtoD(_alloc[b].Pointer, data, (nuint)b.InitialData!.Length), "cuMemcpyHtoD");
            }
        }

        const long Page = 4096;
        foreach (var (b, source, _) in _sparseWeights)
        {
            long start = ((source + Page - 1) & ~(Page - 1)) - source;
            long end = ((source + b.Bytes) & ~(Page - 1)) - source;
            if (end - start >= Page && _dev.TryRegisterHost((byte*)source + start, end - start))
            {
                _registeredWeights.Add((b, source, start, end - start));
            }
        }
    }

    /// <summary>
    /// Rewrites the expert ids of a sparse segment's input to the slots the selected experts are copied to (ids outside the bank, such as -1,
    /// select no expert and stay -1), remembering the experts in slot order.
    /// </summary>
    /// <exception cref="InvalidOperationException">The rows select more experts than the segment holds.</exception>
    private byte[] RemapSparseIds(ReadOnlySpan<byte> data, int experts)
    {
        var ids = MemoryMarshal.Cast<byte, int>(data);
        var remapped = new byte[data.Length];
        var slots = MemoryMarshal.Cast<byte, int>(remapped.AsSpan());
        _sparseSelected.Clear();
        for (int i = 0; i < ids.Length; i++)
        {
            int e = ids[i];
            if ((uint)e >= (uint)experts)
            {
                slots[i] = -1;
                continue;
            }

            int slot = _sparseSelected.IndexOf(e);
            if (slot < 0)
            {
                if (_sparseSelected.Count == _options.SparseExpertSlots)
                {
                    throw new InvalidOperationException($"The rows select more than the {_options.SparseExpertSlots} experts segment {Segment.Index} holds.");
                }

                slot = _sparseSelected.Count;
                _sparseSelected.Add(e);
            }

            slots[i] = slot;
        }

        return remapped;
    }

    /// <summary>Queues the copies of the selected experts (<see cref="RemapSparseIds"/>) of every bank into their slots on the upload stream.</summary>
    private void QueueSparseCopies()
    {
        foreach (var (b, source, expertBytes) in _sparseWeights)
        {
            ulong destination = _alloc[b].Pointer;
            // A copy must not span page-locked and pageable memory: the partial pages around the registered range go apart.
            var registered = _registeredWeights.FirstOrDefault(r => r.Buffer == b);
            long lockedStart = registered.Buffer is null ? 0 : registered.Offset;
            long lockedEnd = registered.Buffer is null ? 0 : registered.Offset + registered.Bytes;
            for (int slot = 0; slot < _sparseSelected.Count; slot++)
            {
                // Runs of consecutive experts go in one copy.
                int run = 1;
                while (slot + run < _sparseSelected.Count && _sparseSelected[slot + run] == _sparseSelected[slot] + run)
                {
                    run++;
                }

                long from = _sparseSelected[slot] * expertBytes;
                long to = from + run * expertBytes;
                ulong target = destination + (ulong)(slot * expertBytes);
                foreach (long cut in (ReadOnlySpan<long>)[lockedStart, lockedEnd, to])
                {
                    if (cut > from && from < to)
                    {
                        long end = Math.Min(cut, to);
                        CudaApi.Check(CudaApi.cuMemcpyHtoDAsync(target, (byte*)source + from, (nuint)(end - from), _dev.Stream), "cuMemcpyHtoDAsync");
                        target += (ulong)(end - from);
                        from = end;
                    }
                }

                slot += run - 1;
            }
        }
    }

    /// <summary>
    /// Page-locks the mapped source of every streamed weight stored as is (no repack), the whole pages inside it, so that
    /// <see cref="BeginStreaming"/> copies it asynchronously; weights the budget or the driver refuses go through staging memory.
    /// </summary>
    private void RegisterStreamedSources()
    {
        nint done;
        CudaApi.Check(CudaApi.cuEventCreate(&done, CudaApi.EventDisableTiming), "cuEventCreate");
        _streamDone = done;
        if (Environment.GetEnvironmentVariable("SHARPLIGHT_STREAM_REGISTER") is "0" or "false")
        {
            return;
        }

        const long Page = 4096;
        foreach (var b in _streamedWeights)
        {
            nint source = b.Repack == WeightRepack.None && b.InitialData is null ? b.Weight?.Source.TryGetPointer() ?? 0 : 0;
            if (source == 0)
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

    /// <inheritdoc />
    /// <remarks>
    /// Weights in page-locked mapped memory are queued as asynchronous copies on the upload stream (they overlap the work running on the
    /// other streams); the others pass through the device's staging buffers, which takes the calling thread.
    /// </remarks>
    public void BeginStreaming()
    {
        if (_sparse is not null && !_streaming)
        {
            _dev.Enter();
            lock (_dev.StreamLock)
            {
                QueueSparseCopies();
                CudaApi.Check(CudaApi.cuEventRecord(_streamDone, _dev.Stream), "cuEventRecord");
            }

            _streaming = true;
            return;
        }

        if (_streamedWeights.Count == 0 || _streaming || WeightsResident)
        {
            return;
        }

        var timer = Stopwatch.StartNew();
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
                    uploader.Upload(buffer, _alloc[buffer]);
                }
            }

            foreach (var (buffer, source, offset, bytes) in _registeredWeights)
            {
                ulong destination = _alloc[buffer].Pointer;
                long tail = buffer.Bytes - offset - bytes;
                // The partial pages around the registered range are copied from pageable memory, still on the upload stream.
                if (offset > 0)
                {
                    CudaApi.Check(CudaApi.cuMemcpyHtoDAsync(destination, (void*)source, (nuint)offset, _dev.Stream), "cuMemcpyHtoDAsync");
                }

                CudaApi.Check(CudaApi.cuMemcpyHtoDAsync(destination + (ulong)offset, (byte*)source + offset, (nuint)bytes, _dev.Stream), "cuMemcpyHtoDAsync");
                if (tail > 0)
                {
                    CudaApi.Check(CudaApi.cuMemcpyHtoDAsync(destination + (ulong)(offset + bytes), (byte*)source + offset + bytes, (nuint)tail, _dev.Stream), "cuMemcpyHtoDAsync");
                }
            }

            CudaApi.Check(CudaApi.cuEventRecord(_streamDone, _dev.Stream), "cuEventRecord");
        }

        _streaming = true;
        if (DiagnosticSwitches.LoadTiming)
        {
            long total = _streamedWeights.Sum(b => b.Bytes);
            Console.Error.WriteLine($"[stream] segment {Segment.Index}: {total / 1e9:F2} GB queued in {timer.Elapsed.TotalMilliseconds:F1} ms ({_registeredWeights.Sum(r => r.Bytes) / 1e9:F2} GB direct)");
        }
    }

    /// <inheritdoc />
    public bool WeightsResident { get; set; }

    /// <summary>
    /// Makes this segment's stream wait for the streamed weights' copies, starting them when <see cref="BeginStreaming"/> did not and the
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
            data = RemapSparseIds(data, sparse.Experts);
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
        fixed (byte* source = data)
        {
            CudaApi.Check(CudaApi.cuMemcpyHtoD(a.Pointer + (ulong)r.ByteOffset, source, (nuint)data.Length), "cuMemcpyHtoD");
        }
    }

    /// <inheritdoc />
    /// <remarks>The producer's output buffer is copied into the input buffer on the device at the start of each execution.</remarks>
    public bool TryForwardFrom(Value input, ICompiledSegment producer)
    {
        if (!DiagnosticSwitches.DeviceForwarding || producer is not CudaSegment source || source._dev != _dev
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
        _dev.Enter();
        fixed (byte* target = destination)
        {
            CudaApi.Check(CudaApi.cuMemcpyDtoH(target, _alloc[r.Buffer].Pointer + (ulong)(r.ByteOffset + byteOffset), (nuint)destination.Length), "cuMemcpyDtoH");
        }
    }

    // ---- execution ---------------------------------------------------------------------------
    /// <inheritdoc />
    public void Execute(RunParameters parameters)
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

        long started = Stopwatch.GetTimestamp();
        _dev.Enter();
        if (_program.ScalarSlots.Count > 0)
        {
            CudaApi.Check(CudaApi.cuMemcpyHtoDAsync(_alloc[_program.RuntimeBuffer].Pointer, _runtimeSlots, RuntimeSlots * sizeof(uint), _stream), "cuMemcpyHtoDAsync");
        }

        if (_streamedWeights.Count > 0 || _sparse is not null)
        {
            StreamWeights();
        }

        var bindings = new Bindings(RecordingSymbols(parameters));
        CopyForwardedInputs(bindings);
        var key = string.Join(",", _program.Symbols.Select(s => $"{s}={bindings[s]}"));
        long recordStart = Stopwatch.GetTimestamp();
        bool recorded = false;
        if (!_recorded.TryGetValue(key, out var recording))
        {
            recorded = true;
            CheckWithinCompiledBounds(bindings);
            EvictOldestRecording();
            // A shape's first run is launched launch by launch anyway: each launch goes out as soon as it is resolved, so the device
            // starts while the rest is still being recorded (mixed serving steps of a new row count record ~700 launches).
            recording = Record(_program.Launches, bindings, launch: !Profile);
            _recorded[key] = recording;
            _recordOrder.Enqueue(key);
            if (DiagnosticSwitches.Trace)
            {
                Console.Error.WriteLine($"[cuda] recorded {key}: {recording.Length} launches");
            }
        }

        long recordEnd = Stopwatch.GetTimestamp();
        _replayed = false;
        double recordMs = recorded ? Stopwatch.GetElapsedTime(recordStart, recordEnd).TotalMilliseconds : 0;
        if (Profile)
        {
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
            _pendingTimed = true;
            _pending = "the launches of segment " + Segment.Index;
        }
        else
        {
            RunStep(recording, key);
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
    /// Waits for the submitted execution, if any: checks it for errors and records its GPU time and the host's wait
    /// (<see cref="LastRunTimings"/>). Thread-safe (several threads may read outputs at once).
    /// </summary>
    private void WaitForCompletion()
    {
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
            CudaApi.Check(CudaApi.cuMemcpyDtoH(&count, _alloc[slot.Buffer].Pointer + (ulong)slot.ByteOffset, sizeof(int)), "cuMemcpyDtoH");
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

    /// <summary>Keeps the recording cache bounded (data-dependent symbols can take a new value on every run).</summary>
    private void EvictOldestRecording()
    {
        if (_recorded.Count < MaxRecordings)
        {
            return;
        }

        string key = _recordOrder.Dequeue();
        _recorded.Remove(key);
        DestroyGraph(key);
    }

    /// <summary>
    /// Runs a recorded step. With <see cref="DeviceFeatures.LaunchGraphs"/> a step recording that runs a second time is captured into
    /// a CUDA graph, and from then on replayed with one <c>cuGraphLaunch</c> instead of a host call per launch (decode steps repeat
    /// their recording: positions and lengths are runtime scalars uploaded before the launches). Steps that run once (prefill
    /// chunks of new lengths) are not captured, so they pay no instantiation.
    /// </summary>
    private void RunStep(CudaRecordedLaunch[] recording, string key)
    {
        if (!_dev.LaunchGraphs || DiagnosticSwitches.SyncEachLaunch || recording.Length < 2)
        {
            Submit(recording, "the launches of segment " + Segment.Index);
            return;
        }

        _graphs.TryGetValue(key, out var entry);
        if (entry.Exec == 0 && entry.Runs >= 1)
        {
            entry = Capture(recording, entry.Runs);
        }

        entry.Runs++;
        _graphs[key] = entry;
        if (entry.Exec == 0)
        {
            Submit(recording, "the launches of segment " + Segment.Index);
            return;
        }

        CudaApi.Check(CudaApi.cuEventRecord(_start, _stream), "cuEventRecord");
        CudaApi.Check(CudaApi.cuGraphLaunch(entry.Exec, _stream), "cuGraphLaunch");
        CudaApi.Check(CudaApi.cuEventRecord(_end, _stream), "cuEventRecord");
        _replayed = true;
        _pendingTimed = true;
        _pending = "the graph of segment " + Segment.Index;
    }

    /// <summary>Launches a recording between the timing events without waiting for it (<see cref="WaitForCompletion"/> does).</summary>
    private void Submit(ReadOnlySpan<CudaRecordedLaunch> launches, string what)
    {
        CudaApi.Check(CudaApi.cuEventRecord(_start, _stream), "cuEventRecord");
        _previousOneWave = false;   // copies may precede the first kernel: it is never a dependent
        foreach (var launch in launches)
        {
            Launch(launch);
        }

        CudaApi.Check(CudaApi.cuEventRecord(_end, _stream), "cuEventRecord");
        _pendingTimed = true;
        _pending = what;
    }

    /// <summary>Captures the launches of a recording into an instantiated graph (or none, when the driver refuses the capture).</summary>
    private (int Runs, nint Graph, nint Exec) Capture(CudaRecordedLaunch[] recording, int runs)
    {
        CudaApi.Check(CudaApi.cuStreamBeginCapture(_stream, CudaApi.CaptureModeThreadLocal), "cuStreamBeginCapture");
        _previousOneWave = false;   // the graph's first kernel follows whatever the stream ran before it
        nint graph = 0;
        try
        {
            foreach (var launch in recording)
            {
                Launch(launch);
            }
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
        if (_graphs.Remove(key, out var entry) && entry.Exec != 0)
        {
            CudaApi.cuGraphExecDestroy(entry.Exec);
            CudaApi.cuGraphDestroy(entry.Graph);
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

        // Program order; one stream needs no barriers, so the hazard tracking of the Vulkan recorder is skipped.
        foreach (var item in launches)
        {
            if (item.Select(bindings) is not { } variant)
            {
                continue;
            }

            var grid = variant.Grid(bindings);
            if (grid.X == 0 || grid.Y == 0 || grid.Z == 0)
            {
                continue;
            }

            var kernel = _dev.Kernel(variant.Kernel);
            recorded.Add(new CudaRecordedLaunch(item.Name, ProfileName(item.Name, variant.Kernel.EntryPoint), kernel, grid.X, grid.Y, grid.Z,
                PackArguments(item, bindings, kernel.TensorMaps)));
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

    private void Launch(CudaRecordedLaunch launch)
    {
        var kernel = launch.Kernel;
        if (kernel.Library is { } gemm)
        {
            // Library GEMM: arguments x, w, y (device addresses), then the row count.
            var args = launch.Arguments;
            _previousOneWave = false;
            _dev.Blas!.Gemm(_stream, gemm.Type, gemm.N, gemm.K, BitConverter.ToUInt32(args, 24),
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
            bool dependent = kernel.WaitsForPrevious && _dev.DependentLaunches && _previousOneWave && oneWave;
            _previousOneWave = oneWave;
            int result = dependent
                ? LaunchDependent(kernel, launch, &parameter)
                : CudaApi.cuLaunchKernel(kernel.Function, launch.GridX, launch.GridY, launch.GridZ, kernel.ThreadsX, kernel.ThreadsY, kernel.ThreadsZ,
                    (uint)kernel.SharedBytes, _stream, &parameter, null);
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
        if (_disposed)
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

        _disposed = true;
        if (_streamDone != 0)
        {
            // Copies queued by BeginStreaming without an execution may still be running.
            CudaApi.cuEventSynchronize(_streamDone);
            CudaApi.cuEventDestroy(_streamDone);
        }

        foreach (var (_, source, offset, _) in _registeredWeights)
        {
            _dev.UnregisterHost((byte*)source + offset);
        }

        _recorded.Clear();
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
    }
}
