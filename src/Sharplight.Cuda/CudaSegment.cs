using System.Diagnostics;
using Sharplight.Cuda.Interop;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Shaders.Lowering;

namespace Sharplight.Cuda;

/// <summary>
/// A compiled segment on a CUDA device: owns the device buffers of its <see cref="KernelProgram"/> and launches its kernels on a stream of
/// its own, from a recording per distinct binding of the symbolic dimensions (kernels, arguments and grids resolved once). Launches on one
/// stream run in order, so no barriers are needed. Inputs and outputs are copied synchronously; an execution waits for its stream.
/// </summary>
public sealed unsafe class CudaSegment : ICompiledSegment, IPartialOutputSegment, IDeviceForwardingSegment
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
    /// <summary>Inputs copied from an earlier segment's output buffer instead of from <see cref="SetInput"/> (<see cref="TryForwardFrom"/>).</summary>
    private readonly Dictionary<BufferDecl, (CudaBuffer Source, long SourceOffset)> _forwarded = [];
    private readonly Dictionary<string, CudaRecordedLaunch[]> _recorded = [];
    private readonly Queue<string> _recordOrder = new();
    private readonly nint _stream;
    private readonly nint _start;
    private readonly nint _end;
    /// <summary>Pinned host copy of the runtime scalar slots, uploaded at the start of every execution.</summary>
    private readonly uint* _runtimeSlots;
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
            void* slots;
            CudaApi.Check(CudaApi.cuMemAllocHost(&slots, RuntimeSlots * sizeof(uint)), "cuMemAllocHost");
            _runtimeSlots = (uint*)slots;
            new Span<uint>(_runtimeSlots, RuntimeSlots).Clear();

            long bytes = 0;
            foreach (var b in program.Buffers)
            {
                bytes += AllocateBuffer(b);
            }

            DeviceMemoryBytes = bytes;
            var uploadTimer = Stopwatch.StartNew();
            UploadWeights();
            if (DiagnosticSwitches.LoadTiming)
            {
                Console.Error.WriteLine($"[load] upload segment {segment.Index}: {uploadTimer.ElapsedMilliseconds} ms, {program.Buffers.Where(b => b.Kind == BufferKind.Weight).Sum(b => b.Bytes) / 1e9:F2} GB");
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

    /// <summary>GPU time (first to last launch) of the most recent <see cref="Execute(RunParameters)"/>, in ms.</summary>
    public double LastGpuMs { get; private set; }

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

    /// <summary>Allocates (or borrows) the buffer of <paramref name="b"/>; returns the bytes this segment newly holds.</summary>
    private long AllocateBuffer(BufferDecl b)
    {
        if (b.Kind == BufferKind.Weight && b.Weight is { } w && _options.SharedWeights is { } shared)
        {
            // Weights shared with the other compiled models of this source on the device (e.g. a draft head's LM head).
            var lease = shared.Acquire(_dev, WeightBufferKey.For(w, LayoutName(b)), b.Bytes, () => _dev.Allocate(b.Bytes), _dev.Free);
            _leases[b] = lease;
            _alloc[b] = lease.Buffer;
            return lease.MustFill ? b.Bytes : 0;
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

        _alloc[b] = _dev.Allocate(b.Bytes);
        sharedStates?.Add(b.State!, _alloc[b]);
        return b.Bytes;
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
                using var uploader = new CudaWeightUploader(_dev);
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

        _dev.Enter();
        if (_program.ScalarSlots.Count > 0)
        {
            CudaApi.Check(CudaApi.cuMemcpyHtoDAsync(_alloc[_program.RuntimeBuffer].Pointer, _runtimeSlots, RuntimeSlots * sizeof(uint), _stream), "cuMemcpyHtoDAsync");
        }

        var bindings = new Bindings(RecordingSymbols(parameters));
        CopyForwardedInputs(bindings);
        var key = string.Join(",", _program.Symbols.Select(s => $"{s}={bindings[s]}"));
        if (!_recorded.TryGetValue(key, out var recording))
        {
            CheckWithinCompiledBounds(bindings);
            EvictOldestRecording();
            recording = Record(_program.Launches, bindings);
            _recorded[key] = recording;
            _recordOrder.Enqueue(key);
            if (DiagnosticSwitches.Trace)
            {
                Console.Error.WriteLine($"[cuda] recorded {key}: {recording.Length} launches");
            }
        }

        if (Profile)
        {
            RunProfiled(recording, key);
        }
        else
        {
            LastGpuMs = Run(recording);
        }

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

        _recorded.Remove(_recordOrder.Dequeue());
    }

    /// <inheritdoc />
    public void ResetState()
    {
        _dev.Enter();
        RunLaunches(_program.ResetLaunches);
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
    private CudaRecordedLaunch[] Record(IReadOnlyList<KernelLaunch> launches, Bindings bindings)
    {
        var recorded = new List<CudaRecordedLaunch>(launches.Count);
        foreach (var (launch, _, variant, _) in HazardTracker.Order(launches, bindings, reorder: false))
        {
            var grid = variant.Grid(bindings);
            if (grid.X == 0 || grid.Y == 0 || grid.Z == 0)
            {
                continue;
            }

            recorded.Add(new CudaRecordedLaunch(launch.Name, ProfileName(launch.Name, variant.Kernel.EntryPoint), _dev.Kernel(variant.Kernel), grid.X, grid.Y, grid.Z, PackArguments(launch, bindings)));
        }

        return [.. recorded];
    }

    /// <summary>The kernel's parameter struct: every pointer (8 bytes) first, then the 32-bit scalars, as the kernels declare them.</summary>
    private byte[] PackArguments(KernelLaunch launch, Bindings bindings)
    {
        int pointers = launch.Args.Count(a => a is PointerArg);
        int bytes = pointers * 8 + (launch.Args.Count - pointers) * 4;
        var packed = new byte[Math.Max(16, (bytes + 15) & ~15)];
        fixed (byte* p = packed)
        {
            int offset = 0;
            foreach (var arg in launch.Args)
            {
                if (arg is PointerArg pointer)
                {
                    *(ulong*)(p + offset) = _alloc[pointer.Buffer].Pointer + (ulong)pointer.ByteOffset;
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
        }

        return packed;
    }

    /// <summary>Launches <paramref name="launches"/> on the segment's stream, waits for them and returns the GPU time in ms.</summary>
    private double Run(ReadOnlySpan<CudaRecordedLaunch> launches)
    {
        CudaApi.Check(CudaApi.cuEventRecord(_start, _stream), "cuEventRecord");
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
            int result = CudaApi.cuLaunchKernel(kernel.Function, launch.GridX, launch.GridY, launch.GridZ, kernel.ThreadsX, kernel.ThreadsY, kernel.ThreadsZ,
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

        _disposed = true;
        // Every execution waits for its stream, so no work of this segment is in flight.
        _recorded.Clear();
        foreach (var (b, a) in _alloc)
        {
            if (!_leases.ContainsKey(b) && !_borrowedStates.Contains(b))
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
        if (_runtimeSlots != null)
        {
            CudaApi.cuMemFreeHost(_runtimeSlots);
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
