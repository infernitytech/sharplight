using System.Diagnostics;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Shaders.Lowering;
using Vortice.Vulkan;

namespace Sharplight.Vulkan;

/// <summary>
/// A compiled segment on a Vulkan device: owns the buffers of its <see cref="KernelProgram"/> and
/// replays pre-recorded command buffers (one per distinct binding of symbolic dims, e.g. T=1 for
/// decode). Per-step values (n_past) live in a small host-visible runtime buffer, so the decode
/// command buffer is recorded once and re-submitted for every token.
/// </summary>
public sealed unsafe class VulkanSegment : IStreamedSegment, IPartialOutputSegment, IDeviceForwardingSegment
{
    private readonly VulkanDevice _dev;
    private readonly KernelProgram _program;
    private readonly Dictionary<BufferDecl, Allocation> _alloc = [];
    /// <summary>Weight buffers held through <see cref="CompileOptions.SharedWeights"/> (their allocations are freed by the lease).</summary>
    private readonly Dictionary<BufferDecl, SharedWeightLease<Allocation>> _leases = [];
    private readonly HashSet<BufferDecl> _borrowedStates = [];   // state buffers another segment of the same graph allocated (and frees)
    /// <summary>Activation buffers shared with the model's other segments on this device (see <see cref="SharedArena{T}"/>).</summary>
    private readonly Dictionary<BufferDecl, SharedArenaBuffer<Allocation>> _sharedArenas = [];
    /// <summary>With <see cref="CompileOptions.StreamWeights"/>: the weight buffers, copied in at every execution (see <see cref="StreamWeights"/>).</summary>
    private readonly List<BufferDecl> _streamedWeights = [];
    /// <summary>The copy of the streamed weights that <see cref="BeginStreaming"/> started on a worker thread, until an execution waits for it.</summary>
    private Task? _streaming;
    /// <summary>With <see cref="CompileOptions.SparseExpertSlots"/>: the expert ids input, the experts of the banks and the gathered tables.</summary>
    private readonly SparseExpertLayout? _sparse;
    /// <summary>The experts the last <see cref="SetInput"/> of the ids selected, in slot order (copied in at the next execution).</summary>
    private readonly SparseExpertSelection? _sparseSelection;
    /// <summary>With <see cref="CompileOptions.SparseExpertSlots"/>: the expert banks and tables, with the bytes of one expert.</summary>
    private readonly List<(BufferDecl Buffer, long ExpertBytes)> _sparseWeights = [];
    /// <summary>The mapped weight sources this segment imported (<see cref="VulkanDevice.HostImports"/>), released on dispose.</summary>
    private readonly List<nint> _imports = [];
    /// <summary>Host-visible mirrors of Input/Runtime (upload) and Output (readback) buffers. Kernels only
    /// touch device-local memory; the command buffer copies staging ↔ device at its start/end.</summary>
    private readonly Dictionary<BufferDecl, Allocation> _staging = [];
    private readonly Dictionary<string, VkCommandBuffer[]> _recorded = [];

    /// <summary>Barrier placement of each recorded launch set (see <see cref="HazardTracker.Order(IReadOnlyList{KernelLaunch}, Bindings, bool, Dictionary{string, ValueTuple{int, bool}[]}, int)"/>).</summary>
    private readonly Dictionary<string, (int Index, bool BarrierBefore)[]> _orders = [];
    /// <summary>Inputs copied from an earlier segment's device output instead of from staging (<see cref="TryForwardFrom"/>).</summary>
    private readonly Dictionary<BufferDecl, (Allocation Source, long SourceOffset)> _forwarded = [];
    /// <summary>When each recording was last submitted (<see cref="_useClock"/>), for least-recently-used eviction.</summary>
    private readonly Dictionary<string, long> _lastUse = [];
    private long _useClock;
    /// <summary>
    /// Most steps kept recorded per segment (one per distinct symbol binding). A serving engine's batched steps take many shapes
    /// (decode rows plus prompt chunks of any length, and every row count while a round drains): gpt-oss-20b at 64 streams
    /// recorded ~75 shapes per loadgen round at 2-20 ms each with 64 recordings kept first-in first-out.
    /// </summary>
    private const int MaxRecordings = 256;
    /// <summary>
    /// Most launches recorded into one command buffer. NVIDIA (Blackwell, driver 610) loses the device (Xid 13,
    /// SKEDCHECK36_DEPENDENCE_COUNTER_UNDERFLOW) once a command buffer holds more than about 65.8k dispatches, with or without
    /// barriers between them; graphs with that many launches (swin2sr: ~150k tiny concats) are split into several command
    /// buffers that are still submitted together.
    /// </summary>
    private const int MaxLaunchesPerCommandBuffer = 32768;
    private readonly CompileOptions _options;
    private readonly VkFence _fence;
    /// <summary>This segment's own command pool (pools are externally synchronized; segments of different models run on different threads).</summary>
    private readonly VkCommandPool _pool;
    /// <summary>Command pool on the device's transfer queue family (null handle without one), for the download command buffers.</summary>
    private readonly VkCommandPool _transferPool;
    /// <summary>Signalled by the compute submission, waited for by the download on the transfer queue.</summary>
    private readonly VkSemaphore _computeDone;
    /// <summary>Download command buffers on the transfer queue per recording key, for recordings that download on it (see <see cref="UsesTransferQueue"/>).</summary>
    private readonly Dictionary<string, VkCommandBuffer> _downloads = [];
    private readonly VkQueryPool _stepPool;
    private bool _disposed;

    /// <summary>GPU time (first to last command) of the most recent non-profiled <see cref="Execute(RunParameters)"/>, in ms (waits for it).</summary>
    public double LastGpuMs
    {
        get
        {
            WaitForCompletion();
            return _lastGpuMs;
        }
        private set => _lastGpuMs = value;
    }

    private double _lastGpuMs;
    /// <summary>Whether a submitted execution has not been waited for yet (<see cref="WaitForCompletion"/>).</summary>
    private volatile bool _pending;
    private readonly object _completion = new();
    private SegmentRunTimings _lastRunTimings;
    private SegmentRunTimings _pendingTimings;

    /// <inheritdoc />
    /// <remarks>
    /// True unless executions are profiled or split by a submission budget: an execution returns once its command buffers are submitted,
    /// and the segment waits for its fence before it reads an output, takes an input or runs again.
    /// </remarks>
    public bool ExecutesAsynchronously => !Profile && _dev.SubmissionBudgetMs <= 0 && _program.ProducedSymbols.Count == 0;

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

    /// <summary>Waits for the submitted execution, if any, and reads its timestamps. Thread-safe.</summary>
    internal void WaitForCompletion()
    {
        if (!_pending)
        {
            return;
        }

        lock (_completion)
        {
            if (!_pending)
            {
                return;
            }

            long start = Stopwatch.GetTimestamp();
            var fence = _fence;
            _dev.Api.vkWaitForFences(1, &fence, true, ulong.MaxValue).CheckResult();
            double waitMs = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            ulong* ts = stackalloc ulong[2];
            if (_dev.Timestamps && _dev.Api.vkGetQueryPoolResults(_stepPool, 0, 2, 16, ts, 8, VkQueryResultFlags.Bit64) == VkResult.Success)
            {
                _lastGpuMs = (ts[1] - ts[0]) * _dev.Phys.TimestampPeriod * 1e-6;
            }

            _lastRunTimings = _pendingTimings with { GpuMs = _lastGpuMs, WaitMs = waitMs };
            _pending = false;
        }
    }

    /// <inheritdoc />
    public GraphSegment Segment { get; }
    /// <inheritdoc />
    public long DeviceMemoryBytes { get; }

    /// <summary>
    /// Per-dispatch GPU timing. When on, every Execute records a fresh command buffer with
    /// timestamps and accumulates <see cref="ProfileData"/>. Defaults to on when <c>SHARPLIGHT_VK_PROFILE</c> is 1 or 2.
    /// </summary>
    public bool Profile { get; set; } = DiagnosticSwitches.Profile;

    /// <summary>Accumulated GPU time per launch kind (key: "&lt;op&gt; [shape] &lt;entry&gt; T=&lt;tokens&gt;").</summary>
    public Dictionary<string, (double Ms, int Count, double MinMs)> ProfileData { get; } = [];
    /// <summary>Clears <see cref="ProfileData"/>.</summary>
    public void ResetProfile() => ProfileData.Clear();

    /// <summary>Formats <see cref="ProfileData"/> as a table, heaviest first.</summary>
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

    internal VulkanSegment(VulkanDevice dev, GraphSegment segment, KernelProgram program, CompileOptions options)
    {
        _dev = dev;
        _program = program;
        _options = options;
        Segment = segment;

        _pool = dev.CreateCommandPool();
        if (dev.TransferFamily >= 0)
        {
            var tpci = new VkCommandPoolCreateInfo { queueFamilyIndex = (uint)dev.TransferFamily, flags = VkCommandPoolCreateFlags.ResetCommandBuffer };
            VkCommandPool transferPool;
            dev.Api.vkCreateCommandPool(&tpci, null, &transferPool).CheckResult();
            _transferPool = transferPool;
            var sci = new VkSemaphoreCreateInfo();
            VkSemaphore semaphore;
            dev.Api.vkCreateSemaphore(&sci, null, &semaphore).CheckResult();
            _computeDone = semaphore;
        }

        var fci = new VkFenceCreateInfo();
        VkFence f;
        dev.Api.vkCreateFence(&fci, null, &f).CheckResult();
        _fence = f;
        var qci = new VkQueryPoolCreateInfo { queryType = VkQueryType.Timestamp, queryCount = 2 };
        VkQueryPool qp;
        _dev.Api.vkCreateQueryPool(&qci, null, &qp).CheckResult();
        _stepPool = qp;

        long bytes = 0;
        int streamedInputs = 0;
        int streamedOutputs = 0;
        try
        {
            if (options.StreamWeights && options.SparseExpertSlots > 0)
            {
                if (dev.HostImports is null)
                {
                    throw new NotSupportedException("Sparse expert streaming needs imported host memory (VK_EXT_external_memory_host).");
                }

                _sparse = SparseExpertLayout.Of(segment);
                _sparseSelection = new SparseExpertSelection(_sparse.Experts, options.SparseExpertSlots);
                // Import the banks' sources before allocating anything: drivers may refuse some memory (file mappings), and a sparse
                // compilation that would copy through staging is not used.
                foreach (var b in program.Buffers.Where(b => b.Kind == BufferKind.Weight && b.Weight is not null && !_sparse.Tables.Contains(b.Weight)))
                {
                    if (!TryImportSource(b))
                    {
                        throw new NotSupportedException($"Sparse expert streaming: the device does not import the mapped source of {b.Name}.");
                    }
                }
            }

            foreach (var b in program.Buffers)
            {
                ulong size = (ulong)Math.Max(16, b.Bytes);
                if (b.Kind == BufferKind.Weight && b.Weight is not null && _sparse is { } sparse && options.StateScope is { } sparseScope)
                {
                    bytes += AllocateSparseWeight(b, sparse, sparseScope);
                    continue;
                }

                // Streamed weights: the streamed segments of one graph on this device share the memory, by weight ordinal.
                if (b.Kind == BufferKind.Weight && options.StreamWeights && _sparse is null && options.StateScope is { } streamScope)
                {
                    bytes += ShareArena(dev, streamScope, $"stream{options.StreamBuffer}:" + _streamedWeights.Count, b, size);
                    _streamedWeights.Add(b);
                    TryImportSource(b);
                    continue;
                }

                if (b.Kind == BufferKind.Weight && b.Weight is { } w && options.SharedWeights is { } shared)
                {
                    // Weights shared with the other compiled models of this source on the device (e.g. a draft head's LM head).
                    var lease = shared.Acquire(dev, WeightBufferKey.For(w, LayoutName(b)), b.Bytes,
                        () => dev.Memory.Allocate(size, MemoryUsage.Device, Purpose(b)), dev.Memory.Free);
                    _leases[b] = lease;
                    _alloc[b] = lease.Buffer;
                    bytes += lease.MustFill ? b.Bytes : 0;
                    continue;
                }

                // The segments of one graph on this device run one after another and share their transient buffers.
                if (b.Kind == BufferKind.Activations && options.StateScope is { } arenaScope && SharedArena<Allocation>.Enabled)
                {
                    bytes += ShareArena(dev, arenaScope, b.Name, b, size);
                    continue;
                }

                // The streamed compilations of a graph's CPU expert segments run one at a time, each from its inputs set by the host to its
                // outputs read back: their device-side input and output buffers are shared too (by ordinal: the layers' value names differ;
                // not inputs holding graph inputs, which the caller sets for every segment before a run); each keeps its own staging memory.
                if (b.Kind is BufferKind.Input or BufferKind.Output && options.StreamWeights && options.StateScope is { } ioScope && SharedArena<Allocation>.Enabled
                    && (b.Kind == BufferKind.Output || program.InputRefs.All(kv => kv.Value.Buffer != b || kv.Key.Kind == ValueKind.Intermediate)))
                {
                    bytes += ShareArena(dev, ioScope, $"streamed-{b.Kind}:{(b.Kind == BufferKind.Input ? streamedInputs++ : streamedOutputs++)}", b, size);
                    _staging[b] = b.Kind == BufferKind.Input
                        ? dev.Memory.Allocate(size, MemoryUsage.Upload, "upload staging of " + Purpose(b))
                        : dev.Memory.Allocate(size, MemoryUsage.Readback, "readback staging of " + Purpose(b));
                    continue;
                }

                // The segments of one graph on this device share each state (its nodes may be split by other devices' nodes).
                var sharedStates = b.Kind == BufferKind.State && b.State is not null && options.StateScope is { } scope
                    ? dev.SharedStates.GetOrCreateValue(scope) : null;
                if (sharedStates is not null && sharedStates.TryGetValue(b.State!, out var existing))
                {
                    _alloc[b] = existing;
                    _borrowedStates.Add(b);
                    continue;
                }

                _alloc[b] = dev.Memory.Allocate(size, MemoryUsage.Device, Purpose(b));
                sharedStates?.Add(b.State!, _alloc[b]);
                bytes += b.Bytes;
                if (b.Kind is BufferKind.Input or BufferKind.Runtime)
                {
                    _staging[b] = dev.Memory.Allocate(size, MemoryUsage.Upload, "upload staging of " + Purpose(b));
                }
                else if (b.Kind == BufferKind.Output)
                {
                    _staging[b] = dev.Memory.Allocate(size, MemoryUsage.Readback, "readback staging of " + Purpose(b));
                }
            }
            DeviceMemoryBytes = bytes;
            var swUp = Stopwatch.StartNew();
            if (_streamedWeights.Count == 0)
            {
                UploadWeights();
            }
            if (DiagnosticSwitches.LoadTiming)
            {
                Console.Error.WriteLine($"[load] upload segment {segment.Index}: {swUp.ElapsedMilliseconds} ms, {program.Buffers.Where(b => b.Kind == BufferKind.Weight).Sum(b => b.Bytes) / 1e9:F2} GB");
            }

            RunLaunches(program.ResetLaunches);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>
    /// Allocates a weight buffer of a sparse segment and imports its mapped source: an expert bank holds
    /// <see cref="CompileOptions.SparseExpertSlots"/> experts (shared with the model's other sparse segments on the device, by bank ordinal);
    /// a gathered per-expert table keeps all its rows, zeroed once: gathers clamp the ids to the table, so the ids of -1 (no expert) read
    /// the rows past the slots, which stay zero. Returns the bytes newly allocated (the banks' sources are imported
    /// before).
    /// </summary>
    /// <exception cref="NotSupportedException">The weight is repacked or not memory-mapped.</exception>
    private long AllocateSparseWeight(BufferDecl b, SparseExpertLayout sparse, object scope)
    {
        nint source = b.Repack == WeightRepack.None && b.InitialData is null ? b.Weight!.Source.TryGetPointer() : 0;
        if (source == 0 || b.Bytes % sparse.Experts != 0 || b.Bytes != b.Weight!.ByteSize)
        {
            throw new NotSupportedException($"Sparse expert streaming: {b.Name} is repacked or not memory-mapped.");
        }

        long expertBytes = b.Bytes / sparse.Experts;
        bool table = sparse.Tables.Contains(b.Weight);
        ulong size = (ulong)Math.Max(16, table ? b.Bytes : _options.SparseExpertSlots * expertBytes);
        bool allocated = ShareArena(_dev, scope, "sparse:" + _sparseWeights.Count, b, size) > 0;
        _sparseWeights.Add((b, expertBytes));
        if (table && allocated)
        {
            var target = _alloc[b];
            _dev.Immediate(cb => _dev.Api.vkCmdFillBuffer(cb, target.Buffer, target.Offset, size & ~3UL, 0));
        }

        if (table)
        {
            TryImportSource(b);
        }

        return allocated ? (long)size : 0;
    }

    /// <summary>Imports the mapped source of a weight stored as is (<see cref="VulkanDevice.HostImports"/>); false when it is not imported.</summary>
    private bool TryImportSource(BufferDecl b)
    {
        nint source = b.Repack == WeightRepack.None && b.InitialData is null ? b.Weight?.Source.TryGetPointer() ?? 0 : 0;
        if (source == 0 || _dev.HostImports is not { } imports || !imports.TryImport((byte*)source, b.Bytes))
        {
            return false;
        }

        _imports.Add(source);
        return true;
    }

    /// <summary>
    /// Queues the copies of the selected experts (<see cref="SparseExpertSelection.Selected"/>) of every bank into their slots, from the
    /// imported pages, on the device queue (the execution submitted next runs after them).
    /// </summary>
    private void CopySelectedExperts()
    {
        var selected = _sparseSelection!.Selected;
        lock (_dev.StreamLock)
        {
            var uploader = _dev.StreamUploader(1);
            foreach (var (b, expertBytes) in _sparseWeights)
            {
                for (int slot = 0; slot < selected.Count; slot++)
                {
                    // Runs of consecutive experts go in one copy.
                    int run = 1;
                    while (slot + run < selected.Count && selected[slot + run] == selected[slot] + run)
                    {
                        run++;
                    }

                    uploader.UploadRange(b, selected[slot] * expertBytes, run * expertBytes, _alloc[b], (ulong)(slot * expertBytes));
                    slot += run - 1;
                }
            }

            uploader.Submit();
        }
    }

    /// <summary>What a buffer holds, for out-of-memory messages.</summary>
    private static string Purpose(BufferDecl buffer) => $"{buffer.Kind.ToString().ToLowerInvariant()} buffer {buffer.Name}";

    private VkCommandBuffer AllocCb()
    {
        var ai = new VkCommandBufferAllocateInfo { commandPool = _pool, level = VkCommandBufferLevel.Primary, commandBufferCount = 1 };
        VkCommandBuffer cb;
        _dev.Api.vkAllocateCommandBuffers(&ai, &cb).CheckResult();
        return cb;
    }

    private void FreeCb(VkCommandBuffer cb) => _dev.Api.vkFreeCommandBuffers(_pool, 1, &cb);

    /// <summary>Frees the transfer-queue download of recording <paramref name="key"/>, if it has one.</summary>
    private void FreeDownload(string key)
    {
        if (_downloads.Remove(key, out var cb))
        {
            _dev.Api.vkFreeCommandBuffers(_transferPool, 1, &cb);
        }
    }

    /// <summary>Frees every transfer-queue download.</summary>
    private void FreeDownloads()
    {
        foreach (var key in _downloads.Keys.ToList())
        {
            FreeDownload(key);
        }
    }

    private void FreeCbs(VkCommandBuffer[] cbs)
    {
        foreach (var cb in cbs)
        {
            FreeCb(cb);
        }
    }

    // ---- weights ---------------------------------------------------------------------------
    private void UploadWeights()
    {
        // Shared weights another compilation holds are uploaded by that one. A cancelled or failed upload leaves this segment's
        // filling leases unfilled; disposing the segment (the constructor does on any exception) marks them failed, so other
        // holders waiting for them throw instead of hanging, and the next acquirer allocates and fills a fresh buffer.
        // The banks and tables of a sparse compilation are copied in at every execution.
        var weights = _program.Buffers.Where(b => b.Kind == BufferKind.Weight && (!_leases.TryGetValue(b, out var l) || l.MustFill)
            && _sparseWeights.All(w => w.Buffer != b)).ToList();
        var ct = _options.CancellationToken;
        if (weights.Count > 0)
        {
            long largestRow = weights.Where(b => b.Repack != WeightRepack.None).Select(b => WeightRepacker.RowBytes(b.Repack, b.RepackRowElements)).DefaultIfEmpty(1).Max();
            using var uploader = new WeightUploader(_dev, _pool, weights.Sum(b => b.Bytes), largestRow);
            var materialized = WeightCopier.MaterializeInBackground(weights, ct);
            var progress = new LoadProgressCounter(_options.Progress, LoadStage.Uploading, weights.Sum(b => b.Bytes), _dev.Info.Name);
            try
            {
                foreach (var buffer in weights)
                {
                    // A cancelled upload still completes the copies in flight before the staging buffers are released.
                    if (ct.IsCancellationRequested)
                    {
                        break;
                    }

                    uploader.Upload(buffer, _alloc[buffer]);
                    progress.Add(buffer.Bytes);
                }

                uploader.Complete();
                ct.ThrowIfCancellationRequested();
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

    /// <summary>Device layout of a weight buffer in <see cref="WeightBufferKey"/>: empty for the bytes as stored, else the repack.</summary>
    private static string LayoutName(BufferDecl b) => b.Repack == WeightRepack.None ? "" : $"vulkan:{b.Repack}:{b.RepackRowElements}";

    /// <summary>Takes the buffer of <paramref name="b"/> from the arena <paramref name="key"/> of <paramref name="scope"/> on the device; returns the bytes newly allocated.</summary>
    private long ShareArena(VulkanDevice dev, object scope, string key, BufferDecl b, ulong size)
    {
        var arenas = dev.SharedArenas.GetOrCreateValue(scope);
        if (!arenas.TryGetValue(key, out var arena))
        {
            arenas[key] = arena = new SharedArena<Allocation>();
        }

        var sharedArena = arena.Acquire((long)size, n => dev.Memory.Allocate((ulong)n, MemoryUsage.Device, Purpose(b)), dev.Memory.Free, out bool allocated);
        _sharedArenas[b] = sharedArena;
        _alloc[b] = sharedArena.Buffer;
        return allocated ? b.Bytes : 0;
    }

    /// <inheritdoc />
    /// <remarks>The copy runs through the device's staging buffers on a worker thread, while the caller runs the segment before this one.</remarks>
    public void BeginStreaming()
    {
        if (_streamedWeights.Count > 0 && _streaming is null && !WeightsResident)
        {
            _streaming = Task.Run(CopyStreamedWeights);
        }
    }

    /// <inheritdoc />
    public bool WeightsResident { get; set; }

    /// <summary>
    /// Waits for the copy <see cref="BeginStreaming"/> started, or copies the streamed weights now unless they are <see cref="WeightsResident"/>.
    /// </summary>
    private void StreamWeights()
    {
        var streaming = _streaming;
        _streaming = null;
        if (streaming is null && WeightsResident)
        {
            return;
        }

        if (streaming is null)
        {
            CopyStreamedWeights();
        }
        else
        {
            streaming.GetAwaiter().GetResult();
        }
    }

    /// <summary>Copies the streamed weights into their shared buffers (<see cref="CompileOptions.StreamWeights"/>) and waits for the copies.</summary>
    private void CopyStreamedWeights()
    {
        long total = _streamedWeights.Sum(b => b.Bytes);
        long largestRow = _streamedWeights.Where(b => b.Repack != WeightRepack.None).Select(b => WeightRepacker.RowBytes(b.Repack, b.RepackRowElements)).DefaultIfEmpty(1).Max();
        var timer = Stopwatch.StartNew();
        lock (_dev.StreamLock)
        {
            var uploader = _dev.StreamUploader(largestRow);
            foreach (var buffer in _streamedWeights)
            {
                uploader.Upload(buffer, _alloc[buffer]);
            }

            uploader.Complete();
        }

        if (DiagnosticSwitches.LoadTiming)
        {
            Console.Error.WriteLine($"[stream] segment {Segment.Index}: {total / 1e9:F2} GB in {timer.Elapsed.TotalMilliseconds:F1} ms");
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

        _dev.ThrowIfDisposed();
        WaitForCompletion();
        if (_sparse is { } sparse && value == sparse.Ids)
        {
            data = _sparseSelection!.Remap(data);
        }

        var a = _staging[r.Buffer];
        if ((ulong)data.Length > a.Size - (ulong)r.ByteOffset)
        {
            throw new ArgumentException($"Input {value.Name}: {data.Length} bytes exceeds the compiled maximum ({r.Buffer.Bytes}). Increase CompileOptions.MaxTokens.");
        }

        data.CopyTo(new Span<byte>(a.Mapped + r.ByteOffset, data.Length));
    }

    /// <inheritdoc />
    /// <remarks>
    /// The input's device buffer is then filled by a device-to-device copy from the producer's output buffer at the start of the
    /// command buffer. Both segments must run on the same device; the run submits them one after the other.
    /// </remarks>
    public bool TryForwardFrom(Value input, ICompiledSegment producer)
    {
        if (!DiagnosticSwitches.DeviceForwarding || producer is not VulkanSegment source || source._dev != _dev
            || !_program.InputRefs.TryGetValue(input, out var r) || !source._program.OutputRefs.TryGetValue(input, out var o))
        {
            return false;
        }

        _forwarded[r.Buffer] = (source._alloc[o.Buffer], o.ByteOffset - r.ByteOffset);
        foreach (var cbs in _recorded.Values)
        {
            FreeCbs(cbs);
        }

        _recorded.Clear();
        _lastUse.Clear();
        FreeDownloads();
        return true;
    }

    /// <inheritdoc />
    public void GetOutput(Value value, Span<byte> destination) => GetOutput(value, 0, destination);

    /// <inheritdoc />
    public void GetOutput(Value value, long byteOffset, Span<byte> destination)
    {
        _dev.ThrowIfDisposed();
        WaitForCompletion();
        var r = _program.OutputRefs[value];
        var a = _staging[r.Buffer];
        byte* source = a.Mapped + r.ByteOffset + byteOffset;
        if (destination.Length < ParallelReadMinBytes)
        {
            new ReadOnlySpan<byte>(source, destination.Length).CopyTo(destination);
            return;
        }

        fixed (byte* target = destination)
        {
            ParallelCopy(source, target, destination.Length);
        }
    }

    /// <summary>Outputs from this size are read from the staging memory by several threads (<see cref="ParallelCopy"/>).</summary>
    private const int ParallelReadMinBytes = 4 << 20;

    /// <summary>Bytes each thread of a parallel read copies at least.</summary>
    private const int ParallelReadChunkBytes = 1 << 20;

    /// <summary>
    /// Copies <paramref name="length"/> bytes of mapped staging memory in parallel chunks. One thread read the host-cached staging
    /// memory the GPU had just written at about 4 GB/s (EPYC host, NVIDIA driver: segformer-b0's 79 MB output at batch 8 took
    /// 19 ms); several threads together reach the memory bandwidth (2.4 ms).
    /// </summary>
    private static void ParallelCopy(byte* source, byte* target, int length)
    {
        int chunks = Math.Min(Math.Min(Environment.ProcessorCount, 16), length / ParallelReadChunkBytes);
        long per = ((long)length + chunks - 1) / chunks;
        nint src = (nint)source;
        nint dst = (nint)target;
        Parallel.For(0, chunks, i =>
        {
            long start = i * per;
            long count = Math.Min(per, length - start);
            Buffer.MemoryCopy((byte*)(src + (nint)start), (byte*)(dst + (nint)start), count, count);
        });
    }

    // ---- execution ---------------------------------------------------------------------------
    /// <inheritdoc />
    public void Execute(RunParameters parameters) => Execute(parameters, s_noSkips);

    private static readonly HashSet<Value> s_noSkips = [];

    /// <inheritdoc />
    /// <remarks>Each distinct set of skipped outputs gets its own recorded command buffer (without their staging copies).</remarks>
    public void Execute(RunParameters parameters, IReadOnlySet<Value> skipDownloads)
    {
        _dev.ThrowIfDisposed();
        WaitForCompletion();
        if (_sharedArenas.Count > 0)
        {
            // Another segment of the device may still run on the same shared buffers.
            if (_dev.PendingShared is { } previous && previous != this)
            {
                previous.WaitForCompletion();
            }

            _dev.PendingShared = this;
        }

        if (_streamedWeights.Count > 0)
        {
            StreamWeights();
        }

        if (_sparse is not null)
        {
            CopySelectedExperts();
        }

        var rt = _staging[_program.RuntimeBuffer];
        var slots = new Span<uint>(rt.Mapped, 64);
        foreach (var (name, slot) in _program.ScalarSlots)
        {
            slots[slot] = parameters.Scalars.TryGetValue(name, out var v) ? (uint)v
                : throw new ArgumentException($"Runtime scalar '{name}' not provided.");
        }

        var bindings = new Bindings(RecordingSymbols(parameters));
        ExecuteBound(bindings, skipDownloads);
        PublishProducedSymbols(parameters);
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

    /// <summary>Reads back the data-dependent counts computed by the execution and binds them for the later segments and the caller.</summary>
    private void PublishProducedSymbols(RunParameters parameters)
    {
        foreach (var (symbol, slot) in _program.ProducedSymbols)
        {
            int count = *(int*)(_staging[slot.Buffer].Mapped + slot.ByteOffset);
            parameters.Symbols[symbol] = Math.Clamp(count, 0, _options.Bound(symbol));
        }
    }

    private void ExecuteBound(Bindings bindings, IReadOnlySet<Value> skipDownloads)
    {
        if (Profile)
        {
            ExecuteProfiled(bindings, skipDownloads);
            return;
        }
        var key = RecordingKey(bindings, skipDownloads);
        if (_dev.SubmissionBudgetMs > 0)
        {
            ExecuteBudgeted(key, bindings, skipDownloads);
            return;
        }
        long recordStart = Stopwatch.GetTimestamp();
        bool recorded = !_recorded.ContainsKey(key);
        var cbs = Recording(key, bindings, skipDownloads);
        double recordMs = recorded ? Stopwatch.GetElapsedTime(recordStart).TotalMilliseconds : 0;
        long submitStart = Stopwatch.GetTimestamp();
        _lastUse[key] = ++_useClock;
        var fence = _fence;
        _dev.Api.vkResetFences(1, &fence).CheckResult();
        if (_downloads.TryGetValue(key, out var download))
        {
            _dev.SubmitWithDownload(cbs, _computeDone, download, fence);
        }
        else
        {
            _dev.Submit(cbs, fence);
        }

        _pendingTimings = new SegmentRunTimings(0, recordMs, Stopwatch.GetElapsedTime(submitStart).TotalMilliseconds, 0, recorded, true);
        _pending = true;
        if (_program.ProducedSymbols.Count > 0)
        {
            WaitForCompletion();   // the counts are read back now
        }
    }

    /// <summary>The key of the recording for these bindings and skipped downloads.</summary>
    private string RecordingKey(Bindings bindings, IReadOnlySet<Value> skipDownloads)
    {
        var key = string.Join(",", _program.Symbols.Select(s => $"{s}={bindings[s]}"));
        if (skipDownloads.Count > 0)
        {
            key += " skip:" + string.Join(",", skipDownloads.Select(v => v.Id).Order());
        }

        return key;
    }

    /// <summary>The recorded command buffers of <paramref name="key"/>, recording them first when they are not cached.</summary>
    private VkCommandBuffer[] Recording(string key, Bindings bindings, IReadOnlySet<Value> skipDownloads)
    {
        if (_recorded.TryGetValue(key, out var cbs))
        {
            return cbs;
        }

        CheckWithinCompiledBounds(bindings);
        EvictOldestRecording();
        long recordStart = System.Diagnostics.Stopwatch.GetTimestamp();
        bool transfer = UsesTransferQueue(bindings, skipDownloads);
        cbs = RecordStep(key, bindings, transfer ? AllOutputs : skipDownloads);
        if (transfer)
        {
            _downloads[key] = RecordDownload(bindings, skipDownloads);
        }

        _recorded[key] = cbs;
        if (DiagnosticSwitches.Trace)
        {
            Console.Error.WriteLine($"[vk] recorded {key} in {System.Diagnostics.Stopwatch.GetElapsedTime(recordStart).TotalMilliseconds:F1} ms ({_recorded.Count} recordings)");
        }

        return cbs;
    }

    /// <inheritdoc />
    /// <remarks>Records the step's command buffers (kept like those of executed steps, least recently used first out).</remarks>
    public void Prepare(RunParameters parameters, IReadOnlySet<Value> skipDownloads)
    {
        _dev.ThrowIfDisposed();
        if (Profile || _dev.SubmissionBudgetMs > 0)
        {
            return;
        }

        var bindings = new Bindings(RecordingSymbols(parameters));
        var key = RecordingKey(bindings, skipDownloads);
        if (!_recorded.ContainsKey(key))
        {
            Recording(key, bindings, skipDownloads);
            _lastUse[key] = ++_useClock;
        }
    }

    /// <summary>
    /// Records one whole step (timestamps, input copies, every launch, output copies) into new command buffers, one per
    /// <see cref="MaxLaunchesPerCommandBuffer"/> launches, to be submitted together in order.
    /// </summary>
    private VkCommandBuffer[] RecordStep(string key, Bindings bindings, IReadOnlySet<Value> skipDownloads)
    {
        var recordTimer = Stopwatch.StartNew();
        long pipelineTicksBefore = _dev.PipelineTicks;
        int pipelinesBefore = _dev.PipelinesCreated;
        var launches = _program.Launches;
        int count = Math.Max(1, (launches.Count + MaxLaunchesPerCommandBuffer - 1) / MaxLaunchesPerCommandBuffer);
        var cbs = new VkCommandBuffer[count];
        for (int i = 0; i < count; i++)
        {
            var cb = AllocCb();
            cbs[i] = cb;
            var beginInfo = new VkCommandBufferBeginInfo();
            _dev.Api.vkBeginCommandBuffer(cb, &beginInfo).CheckResult();
            if (i == 0)
            {
                if (_dev.Timestamps)
                {
                    _dev.Api.vkCmdResetQueryPool(cb, _stepPool, 0, 2);
                    _dev.Api.vkCmdWriteTimestamp(cb, VkPipelineStageFlags.TopOfPipe, _stepPool, 0);
                }

                RecordInputCopies(cb, bindings);
            }
            else
            {
                // The previous command buffer's launches must finish before this one's read their results.
                Barrier(cb, VkPipelineStageFlags.ComputeShader, VkAccessFlags.ShaderWrite,
                    VkPipelineStageFlags.ComputeShader, VkAccessFlags.ShaderRead | VkAccessFlags.ShaderWrite);
            }

            int start = i * MaxLaunchesPerCommandBuffer;
            Record(cb, Slice(launches, start, MaxLaunchesPerCommandBuffer), bindings, firstIndex: start);
            if (i == count - 1)
            {
                RecordOutputCopies(cb, bindings, skipDownloads);
                if (_dev.Timestamps)
                {
                    _dev.Api.vkCmdWriteTimestamp(cb, VkPipelineStageFlags.BottomOfPipe, _stepPool, 1);
                }
            }

            _dev.Api.vkEndCommandBuffer(cb).CheckResult();
        }

        if (DiagnosticSwitches.LoadTiming)
        {
            double pipelineMs = (double)(_dev.PipelineTicks - pipelineTicksBefore) / Stopwatch.Frequency * 1000;
            Console.Error.WriteLine($"[load] record {key}: {recordTimer.ElapsedMilliseconds} ms in {count} command buffer(s), of which {_dev.PipelinesCreated - pipelinesBefore} pipelines {pipelineMs:F0} ms");
        }

        return cbs;
    }

    private static IReadOnlyList<KernelLaunch> Slice(IReadOnlyList<KernelLaunch> launches, int start, int count)
        => start == 0 && launches.Count <= count ? launches : launches.Skip(start).Take(count).ToList();

    /// <summary>
    /// Keeps the recording cache bounded: data-dependent symbols (e.g. a detection count) can take a new value on every run, and
    /// each distinct binding is recorded once. The least recently submitted recording is freed when the cache is full.
    /// </summary>
    private void EvictOldestRecording()
    {
        if (_recorded.Count < MaxRecordings)
        {
            return;
        }

        var oldest = _lastUse.MinBy(u => u.Value).Key;
        _lastUse.Remove(oldest);
        FreeCbs(_recorded[oldest]);
        _recorded.Remove(oldest);
        FreeDownload(oldest);
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

    private void ExecuteProfiled(Bindings bindings, IReadOnlySet<Value> skipDownloads)
    {
        var names = new string[_program.Launches.Count];
        var ms = MeasureLaunches(bindings, skipDownloads, _dev.SubmissionBudgetMs > 0 ? CalibrationChunk : MaxLaunchesPerCommandBuffer, (i, l, v) =>
        {
            var parts = l.Name.Split(' ');
            names[i] = DiagnosticSwitches.ProfileDetail
                ? $"{l.Name} {v.Kernel.EntryPoint}"
                : $"{parts[0]}{(parts.Length > 2 ? " " + string.Join(" ", parts[2..]) : "")} {v.Kernel.EntryPoint} T={string.Join(",", _program.Symbols.Select(s => bindings[s]))}";
        });
        for (int i = 0; i < names.Length; i++)
        {
            if (names[i] is null)
            {
                continue;
            }

            ProfileData[names[i]] = ProfileData.TryGetValue(names[i], out var entry)
                ? (entry.Ms + ms[i], entry.Count + 1, Math.Min(entry.MinMs, ms[i]))
                : (ms[i], 1, ms[i]);
        }
    }

    /// <summary>
    /// Runs the whole program (input copies, every launch, output copies) with a GPU timestamp after each dispatch, in submissions
    /// of at most <paramref name="chunk"/> launches, and returns the GPU time of each launch in ms (0 for skipped ones).
    /// </summary>
    private double[] MeasureLaunches(Bindings bindings, IReadOnlySet<Value> skipDownloads, int chunk, Action<int, KernelLaunch, KernelVariant>? onLaunch = null)
    {
        if (!_dev.Timestamps)
        {
            return MeasureLaunchesOnHost(bindings, skipDownloads, chunk, onLaunch);
        }

        var api = _dev.Api;
        var launches = _program.Launches;
        var result = new double[launches.Count];
        double tick = _dev.Phys.TimestampPeriod * 1e-6;
        _dev.Immediate(cb => RecordInputCopies(cb, bindings));
        for (int start = 0; start < launches.Count; start += chunk)
        {
            int count = Math.Min(chunk, launches.Count - start);
            var slice = launches.Skip(start).Take(count).ToList();
            if (DiagnosticSwitches.Trace)
            {
                Console.Error.WriteLine($"[vk] launches {start}..{start + count - 1}: {string.Join(" | ", slice.Select(l => l.Name))}");
            }

            uint n = (uint)count + 1;
            var qci = new VkQueryPoolCreateInfo { queryType = VkQueryType.Timestamp, queryCount = n };
            VkQueryPool created;
            api.vkCreateQueryPool(&qci, null, &created).CheckResult();
            var pool = created;
            try
            {
                int s0 = start;
                _dev.Immediate(cb =>
                {
                    api.vkCmdResetQueryPool(cb, pool, 0, n);
                    Record(cb, slice, bindings, (i, l, v) =>
                    {
                        if (i < 0)
                        {
                            api.vkCmdWriteTimestamp(cb, VkPipelineStageFlags.ComputeShader, pool, 0);
                            return;
                        }
                        api.vkCmdWriteTimestamp(cb, VkPipelineStageFlags.ComputeShader, pool, (uint)i + 1);
                        if (v is not null)
                        {
                            onLaunch?.Invoke(s0 + i, l, v);
                        }
                    }, firstIndex: s0);
                });
                var ts = new ulong[n];
                fixed (ulong* t = ts)
                {
                    api.vkGetQueryPoolResults(pool, 0, n, (nuint)(n * 8), t, 8, VkQueryResultFlags.Bit64 | VkQueryResultFlags.Wait).CheckResult();
                }

                for (int i = 0; i < count; i++)
                {
                    result[start + i] = (ts[i + 1] - ts[i]) * tick;
                }

                if (DiagnosticSwitches.Trace)
                {
                    for (int i = 0; i < count; i++)
                    {
                        if (result[start + i] > 50)
                        {
                            Console.Error.WriteLine($"[vk]   {slice[i].Name}: {result[start + i]:F0} ms");
                        }
                    }
                }
            }
            finally
            {
                api.vkDestroyQueryPool(pool, null);
            }
        }
        _dev.Immediate(cb => RecordOutputCopies(cb, bindings, skipDownloads));
        return result;
    }

    /// <summary>
    /// <see cref="MeasureLaunches"/> on a queue without timestamps: each chunk is timed on the host from submission to completion and
    /// its time spread evenly over its launches (with chunks of one launch, as in calibration, that is the launch's time plus the
    /// submission overhead).
    /// </summary>
    private double[] MeasureLaunchesOnHost(Bindings bindings, IReadOnlySet<Value> skipDownloads, int chunk, Action<int, KernelLaunch, KernelVariant>? onLaunch)
    {
        var launches = _program.Launches;
        var result = new double[launches.Count];
        _dev.Immediate(cb => RecordInputCopies(cb, bindings));
        for (int start = 0; start < launches.Count; start += chunk)
        {
            int count = Math.Min(chunk, launches.Count - start);
            var slice = launches.Skip(start).Take(count).ToList();
            int s0 = start;
            long t0 = Stopwatch.GetTimestamp();
            _dev.Immediate(cb => Record(cb, slice, bindings, (i, l, v) =>
            {
                if (i >= 0 && v is not null)
                {
                    onLaunch?.Invoke(s0 + i, l, v);
                }
            }, firstIndex: s0));
            double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
            for (int i = 0; i < count; i++)
            {
                result[start + i] = ms / count;
            }
        }

        _dev.Immediate(cb => RecordOutputCopies(cb, bindings, skipDownloads));
        return result;
    }

    // ---- budgeted execution (devices where one long submission is unsafe, see DeviceInfo.SubmissionBudgetMs) ----------------
    private const int CalibrationChunk = 1;   // one launch per measured submission: a single launch is the only unit whose cost is not yet known

    private sealed class Chunk
    {
        public int Start;
        public int End;
        public double EstMs;
        public VkCommandBuffer Cb;
    }

    private sealed class Plan
    {
        public double[] Est = [];
        public List<Chunk> Chunks = [];
    }

    private readonly Dictionary<string, Plan> _plans = [];

    /// <summary>
    /// First execution of a binding set: run it in small measured submissions and derive the plan. Later executions replay the
    /// plan, one submission (waited on) per chunk of roughly <see cref="VulkanDevice.SubmissionBudgetMs"/>. A chunk that turns
    /// out to run much longer than estimated (e.g. attention over a longer past) is re-estimated and split further.
    /// </summary>
    private void ExecuteBudgeted(string key, Bindings bindings, IReadOnlySet<Value> skipDownloads)
    {
        CheckWithinCompiledBounds(bindings);

        var sw = Stopwatch.StartNew();
        if (!_plans.TryGetValue(key, out var plan))
        {
            var est = MeasureLaunches(bindings, skipDownloads, CalibrationChunk);
            _plans[key] = BuildPlan(est, bindings, skipDownloads);
            LastGpuMs = est.Sum();
            return;
        }
        double total = 0;
        bool replan = false;
        foreach (var c in plan.Chunks)
        {
            var fence = _fence;
            _dev.Api.vkResetFences(1, &fence).CheckResult();
            var cb = c.Cb;
            long t0 = Stopwatch.GetTimestamp();
            _dev.Submit(cb, fence);
            _dev.Api.vkWaitForFences(1, &fence, true, ulong.MaxValue).CheckResult();
            double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
            total += ms;
            if (ms > 2 * _dev.SubmissionBudgetMs && c.End - c.Start > 1)
            {
                double f = ms / Math.Max(c.EstMs, 1e-3);
                for (int i = c.Start; i < c.End; i++)
                {
                    plan.Est[i] = plan.Est[i] > 0 ? plan.Est[i] * f : ms / (c.End - c.Start);
                }

                replan = true;
            }
        }
        LastGpuMs = total;
        if (replan)
        {
            foreach (var c in plan.Chunks)
            {
                FreeCb(c.Cb);
            }

            _plans[key] = BuildPlan(plan.Est, bindings, skipDownloads);
        }
    }

    private Plan BuildPlan(double[] est, Bindings bindings, IReadOnlySet<Value> skipDownloads)
    {
        var launches = _program.Launches;
        var plan = new Plan { Est = est };
        double budget = _dev.SubmissionBudgetMs;
        int start = 0;
        double sum = 0;
        for (int i = 0; i <= launches.Count; i++)
        {
            bool end = i == launches.Count;
            if (end || (i > start && (sum + est[i] > budget || i - start >= MaxLaunchesPerCommandBuffer)))
            {
                if (i > start)
                {
                    plan.Chunks.Add(new Chunk { Start = start, End = i, EstMs = sum });
                }

                start = i;
                sum = 0;
            }
            if (!end)
            {
                sum += est[i];
            }
        }
        if (plan.Chunks.Count == 0)
        {
            plan.Chunks.Add(new Chunk { Start = 0, End = 0 });
        }

        for (int ci = 0; ci < plan.Chunks.Count; ci++)
        {
            var c = plan.Chunks[ci];
            var cb = AllocCb();
            var bi = new VkCommandBufferBeginInfo();
            _dev.Api.vkBeginCommandBuffer(cb, &bi).CheckResult();
            if (ci == 0)
            {
                RecordInputCopies(cb, bindings);
            }
            else
            {
                Barrier(cb, VkPipelineStageFlags.ComputeShader, VkAccessFlags.ShaderWrite,
                VkPipelineStageFlags.ComputeShader, VkAccessFlags.ShaderRead | VkAccessFlags.ShaderWrite);
            }

            Record(cb, launches.Skip(c.Start).Take(c.End - c.Start).ToList(), bindings);
            if (ci == plan.Chunks.Count - 1)
            {
                RecordOutputCopies(cb, bindings, skipDownloads);
            }

            _dev.Api.vkEndCommandBuffer(cb).CheckResult();
            c.Cb = cb;
        }
        return plan;
    }

    /// <inheritdoc />
    public void ResetState()
    {
        WaitForCompletion();
        RunLaunches(_program.ResetLaunches);
    }

    /// <inheritdoc/>
    public bool CanCopyStateRows => true;

    /// <inheritdoc/>
    public void CopyStateRows(IReadOnlyCollection<string> states, int from, int to, int count)
    {
        _dev.ThrowIfDisposed();
        WaitForCompletion();
        var regions = StateRowCopies.Regions(_program, _borrowedStates, states, from, to, count, b => (long)_alloc[b].Size);
        if (regions.Count == 0)
        {
            return;
        }

        _dev.Immediate(cb =>
        {
            // After the kernels of earlier submissions, before those of later ones.
            Barrier(cb, VkPipelineStageFlags.AllCommands, VkAccessFlags.MemoryWrite, VkPipelineStageFlags.Transfer, VkAccessFlags.TransferRead | VkAccessFlags.TransferWrite);
            foreach (var (b, src, dst, bytes) in regions)
            {
                var a = _alloc[b];
                var region = new VkBufferCopy { srcOffset = a.Offset + (ulong)src, dstOffset = a.Offset + (ulong)dst, size = (ulong)bytes };
                _dev.Api.vkCmdCopyBuffer(cb, a.Buffer, a.Buffer, 1, &region);
            }

            Barrier(cb, VkPipelineStageFlags.Transfer, VkAccessFlags.TransferWrite, VkPipelineStageFlags.AllCommands, VkAccessFlags.MemoryRead | VkAccessFlags.MemoryWrite);
        });
    }

    private void RunLaunches(IReadOnlyList<KernelLaunch> launches)
    {
        if (launches.Count == 0)
        {
            return;
        }

        _dev.Immediate(cb => Record(cb, launches, new Bindings(new Dictionary<string, long>())));
    }

    /// <param name="cb">Command buffer being recorded.</param>
    /// <param name="launches">Launches to record, at most <see cref="MaxLaunchesPerCommandBuffer"/>.</param>
    /// <param name="bindings">Symbol values.</param>
    /// <param name="afterDispatch">Called with -1 before the first launch and with each launch's index in <paramref name="launches"/> after it.</param>
    /// <param name="firstIndex">Program index of the first launch, for <see cref="DiagnosticSwitches.MaxLaunches"/> and launch listings.</param>
    private void Record(VkCommandBuffer cb, IReadOnlyList<KernelLaunch> launches, Bindings bindings,
        Action<int, KernelLaunch, KernelVariant>? afterDispatch = null, int firstIndex = 0)
    {
        if (launches.Count > MaxLaunchesPerCommandBuffer)
        {
            throw new InvalidOperationException($"{launches.Count} launches exceed {MaxLaunchesPerCommandBuffer} per command buffer.");
        }

        afterDispatch?.Invoke(-1, null!, null!);
        var api = _dev.Api;
        byte* pc = stackalloc byte[(int)_dev.PushConstantSize];
        VkPipeline bound = VkPipeline.Null;

        // Profiling serializes every launch (one timestamp delta per kernel) and keeps program order, as does bisecting.
        bool reorder = afterDispatch is null && DiagnosticSwitches.MaxLaunches == int.MaxValue && DiagnosticSwitches.ReorderLaunches;
        int skippedUpTo = 0;
        foreach (var (launch, index, variant, barrierBefore) in HazardTracker.Order(launches, bindings, reorder, _orders, firstIndex))
        {
            if (firstIndex + index >= DiagnosticSwitches.MaxLaunches)
            {
                break;
            }

            // Conditional launches not needed for these bindings (e.g. a fused small-batch kernel at large T).
            for (; afterDispatch is not null && skippedUpTo < index; skippedUpTo++)
            {
                afterDispatch(skippedUpTo, launches[skippedUpTo], null!);
            }

            skippedUpTo = Math.Max(skippedUpTo, index + 1);
            bool hazard = afterDispatch is not null // profiling: serialize so each timestamp delta is one kernel
                       || barrierBefore;
            if (hazard && !DiagnosticSwitches.NoBarriers)
            {
                Barrier(cb, VkPipelineStageFlags.ComputeShader, VkAccessFlags.ShaderWrite,
                    VkPipelineStageFlags.ComputeShader, VkAccessFlags.ShaderRead | VkAccessFlags.ShaderWrite);
            }

            var pipe = _dev.Pipeline(variant.Kernel);
            if (pipe != bound)
            {
                api.vkCmdBindPipeline(cb, VkPipelineBindPoint.Compute, pipe);
                bound = pipe;
            }

            int off = 0;
            foreach (var arg in launch.Args.Where(a => a is PointerArg))
            {
                var p = (PointerArg)arg;
                *(ulong*)(pc + off) = _alloc[p.Buffer].Address + (ulong)p.ByteOffset;
                off += 8;
            }
            foreach (var arg in launch.Args.Where(a => a is not PointerArg))
            {
                switch (arg)
                {
                    case UIntArg u: *(uint*)(pc + off) = u.Value; break;
                    case FloatArg f: *(float*)(pc + off) = f.Value; break;
                    case SymbolArg s: *(uint*)(pc + off) = (uint)(bindings[s.Symbol] * s.Multiplier); break;
                }
                off += 4;
            }
            if (off > _dev.PushConstantSize)
            {
                throw new InvalidOperationException($"{launch.Name}: {off} bytes of arguments exceed push constant space.");
            }

            api.vkCmdPushConstants(cb, _dev.PipelineLayout, VkShaderStageFlags.Compute, 0, (uint)((off + 3) & ~3), pc);
            var g = variant.Grid(bindings);
            if (DiagnosticSwitches.ListLaunches)
            {
                Console.Error.WriteLine($"[launch {firstIndex + index}]{(hazard ? " (barrier)" : "")} {launch.Name} {variant.Kernel.CacheKey} grid {g.X}x{g.Y}x{g.Z}");
            }

            api.vkCmdDispatch(cb, g.X, g.Y, g.Z);
            afterDispatch?.Invoke(index, launch, variant);
        }

        for (; afterDispatch is not null && skippedUpTo < launches.Count && firstIndex + skippedUpTo < DiagnosticSwitches.MaxLaunches; skippedUpTo++)
        {
            afterDispatch(skippedUpTo, launches[skippedUpTo], null!);
        }
        // Make results visible to the host.
        Barrier(cb, VkPipelineStageFlags.ComputeShader, VkAccessFlags.ShaderWrite, VkPipelineStageFlags.Host, VkAccessFlags.HostRead);
    }

    /// <summary>Bytes of a staged tensor actually used under the given symbol bindings.</summary>
    private static ulong UsedBytes(TensorRef r, Bindings b)
    {
        long elems = r.MaxElements;
        if (r.Symbol is not null && r.Shape.Length > 0 && r.Shape[0] > 0)
        {
            elems = elems / r.Shape[0] * Math.Min(r.Shape[0], b[r.Symbol]);
        }

        return (ulong)r.Type.ByteSize(elems);
    }

    private void Copy(VkCommandBuffer cb, Allocation src, Allocation dst, ulong offset, ulong size)
    {
        if (size == 0)
        {
            return;
        }

        var region = new VkBufferCopy { srcOffset = src.Offset + offset, dstOffset = dst.Offset + offset, size = Math.Min(size, Math.Min(src.Size, dst.Size) - offset) };
        _dev.Api.vkCmdCopyBuffer(cb, src.Buffer, dst.Buffer, 1, &region);
    }

    private void RecordInputCopies(VkCommandBuffer cb, Bindings b)
    {
        Copy(cb, _staging[_program.RuntimeBuffer], _alloc[_program.RuntimeBuffer], 0, 256);
        if (_forwarded.Count > 0)
        {
            // The producers' kernels wrote the forwarded values in earlier submissions.
            Barrier(cb, VkPipelineStageFlags.ComputeShader, VkAccessFlags.ShaderWrite, VkPipelineStageFlags.Transfer, VkAccessFlags.TransferRead);
        }

        foreach (var r in _program.InputRefs.Values)
        {
            if (_forwarded.TryGetValue(r.Buffer, out var f))
            {
                ulong size = UsedBytes(r, b);
                if (size > 0)
                {
                    var dst = _alloc[r.Buffer];
                    var region = new VkBufferCopy { srcOffset = f.Source.Offset + (ulong)(r.ByteOffset + f.SourceOffset), dstOffset = dst.Offset + (ulong)r.ByteOffset, size = size };
                    _dev.Api.vkCmdCopyBuffer(cb, f.Source.Buffer, dst.Buffer, 1, &region);
                }

                continue;
            }

            Copy(cb, _staging[r.Buffer], _alloc[r.Buffer], (ulong)r.ByteOffset, UsedBytes(r, b));
        }

        Barrier(cb, VkPipelineStageFlags.Transfer, VkAccessFlags.TransferWrite,
            VkPipelineStageFlags.ComputeShader, VkAccessFlags.ShaderRead | VkAccessFlags.ShaderWrite);
    }

    /// <summary>Output bytes from which a recording downloads its outputs on the transfer queue (<see cref="VulkanDevice.TransferFamily"/>).</summary>
    private const ulong TransferQueueMinBytes = 1 << 20;

    /// <summary>The outputs as a skip set: the compute command buffers of a recording that downloads on the transfer queue copy none.</summary>
    private HashSet<Value> AllOutputs => field ??= [.. _program.OutputRefs.Keys];

    /// <summary>
    /// True when a recording downloads its outputs on the transfer queue: the device has one and the outputs total at least
    /// <see cref="TransferQueueMinBytes"/> (below that, the second submission costs more than the slower copy). Data-dependent
    /// counts are read back with the outputs.
    /// </summary>
    private bool UsesTransferQueue(Bindings b, IReadOnlySet<Value> skipDownloads)
    {
        if (_dev.TransferFamily < 0)
        {
            return false;
        }

        ulong bytes = 0;
        foreach (var (value, r) in _program.OutputRefs)
        {
            bytes += skipDownloads.Contains(value) ? 0 : UsedBytes(r, b);
        }

        return bytes >= TransferQueueMinBytes;
    }

    /// <summary>
    /// The transfer-queue command buffer that copies the outputs (except <paramref name="skipDownloads"/>) and the data-dependent
    /// counts to their host staging buffers, after the compute submission has signalled <see cref="_computeDone"/>.
    /// </summary>
    private VkCommandBuffer RecordDownload(Bindings b, IReadOnlySet<Value> skipDownloads)
    {
        var ai = new VkCommandBufferAllocateInfo { commandPool = _transferPool, level = VkCommandBufferLevel.Primary, commandBufferCount = 1 };
        VkCommandBuffer cb;
        _dev.Api.vkAllocateCommandBuffers(&ai, &cb).CheckResult();
        var beginInfo = new VkCommandBufferBeginInfo();
        _dev.Api.vkBeginCommandBuffer(cb, &beginInfo).CheckResult();
        foreach (var (value, r) in _program.OutputRefs)
        {
            if (!skipDownloads.Contains(value))
            {
                Copy(cb, _alloc[r.Buffer], _staging[r.Buffer], (ulong)r.ByteOffset, UsedBytes(r, b));
            }
        }

        foreach (var slot in _program.ProducedSymbols.Values)
        {
            Copy(cb, _alloc[slot.Buffer], _staging[slot.Buffer], (ulong)slot.ByteOffset, sizeof(int));
        }

        Barrier(cb, VkPipelineStageFlags.Transfer, VkAccessFlags.TransferWrite, VkPipelineStageFlags.Host, VkAccessFlags.HostRead);
        _dev.Api.vkEndCommandBuffer(cb).CheckResult();
        return cb;
    }

    private void RecordOutputCopies(VkCommandBuffer cb, Bindings b, IReadOnlySet<Value> skipDownloads)
    {
        Barrier(cb, VkPipelineStageFlags.ComputeShader, VkAccessFlags.ShaderWrite, VkPipelineStageFlags.Transfer, VkAccessFlags.TransferRead);
        foreach (var (value, r) in _program.OutputRefs)
        {
            if (skipDownloads.Contains(value))
            {
                continue;
            }

            Copy(cb, _alloc[r.Buffer], _staging[r.Buffer], (ulong)r.ByteOffset, UsedBytes(r, b));
        }

        foreach (var slot in _program.ProducedSymbols.Values)
        {
            Copy(cb, _alloc[slot.Buffer], _staging[slot.Buffer], (ulong)slot.ByteOffset, sizeof(int));
        }

        Barrier(cb, VkPipelineStageFlags.Transfer, VkAccessFlags.TransferWrite, VkPipelineStageFlags.Host, VkAccessFlags.HostRead);
    }

    private void Barrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkAccessFlags srcAccess, VkPipelineStageFlags dst, VkAccessFlags dstAccess)
    {
        var mb = new VkMemoryBarrier { srcAccessMask = srcAccess, dstAccessMask = dstAccess };
        _dev.Api.vkCmdPipelineBarrier(cb, src, dst, 0, 1, &mb, 0, null, 0, null);
    }

    /// <summary>Number of dispatches and barriers in the command buffer for the given token count.</summary>
    public (int Dispatches, int Barriers) Stats(long tokens)
    {
        var s = Schedule(tokens);
        return (s.Count, s.Count(x => x.BarrierBefore));
    }

    /// <summary>Launch order with the barriers <see cref="Execute(RunParameters)"/> places (for diagnostics).</summary>
    public List<(string Name, bool BarrierBefore)> Schedule(long tokens = 1)
    {
        var bindings = new Bindings(_program.Symbols.ToDictionary(s => s, _ => tokens));
        return [.. HazardTracker.Order(_program.Launches, bindings, DiagnosticSwitches.ReorderLaunches).Select(o => (o.Launch.Name, o.BarrierBefore))];
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
            WaitForCompletion();
        }

        _disposed = true;
        try
        {
            if (_dev.PendingShared == this)
            {
                _dev.PendingShared = null;
            }

            // A copy BeginStreaming started without an execution may still write the shared weight buffers.
            _streaming?.Wait();
        }
        catch (AggregateException)
        {
            // Its failure belongs to the execution that never came.
        }

        if (_dev.IsDisposed)
        {
            // The device was disposed first (out of order): destroying it released this segment's command buffers, pools, fences
            // and memory, and calling into the destroyed VkDevice would crash. Only the shared-weight holds are given back.
            foreach (var lease in _leases.Values)
            {
                lease.Dispose();
            }

            _leases.Clear();
            _alloc.Clear();
            _staging.Clear();
            return;
        }

        foreach (nint source in _imports)
        {
            _dev.HostImports!.Release((byte*)source);
        }

        _imports.Clear();
        // Every Execute waits for its fence, so no work of this segment is in flight; no device-wide wait (it would
        // stall, and need the queue lock against, the other models sharing the device).
        foreach (var cbs in _recorded.Values)
        {
            FreeCbs(cbs);
        }

        _recorded.Clear();
        FreeDownloads();
        foreach (var p in _plans.Values)
        {
            foreach (var c in p.Chunks)
            {
                FreeCb(c.Cb);
            }
        }

        _plans.Clear();
        foreach (var (b, a) in _alloc)
        {
            if (_sharedArenas.TryGetValue(b, out var arena))
            {
                arena.Release();
            }
            else if (!_leases.ContainsKey(b) && !_borrowedStates.Contains(b))
            {
                _dev.Memory.Free(a);
            }
        }

        foreach (var lease in _leases.Values)
        {
            lease.Dispose();
        }

        _leases.Clear();
        _alloc.Clear();
        foreach (var a in _staging.Values)
        {
            _dev.Memory.Free(a);
        }

        _staging.Clear();
        _dev.Api.vkDestroyFence(_fence, null);
        _dev.Api.vkDestroyQueryPool(_stepPool, null);
        _dev.Api.vkDestroyCommandPool(_pool, null);
        if (_transferPool != VkCommandPool.Null)
        {
            _dev.Api.vkDestroyCommandPool(_transferPool, null);
            _dev.Api.vkDestroySemaphore(_computeDone, null);
        }

    }
}
