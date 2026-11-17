using System.Diagnostics;
using Sharplight.Graph;
using Sharplight.Metal.Interop;
using Sharplight.Runtime;
using Sharplight.Shaders.Lowering;

namespace Sharplight.Metal;

/// <summary>
/// A compiled segment on a Metal device: owns the shared-storage buffers of its <see cref="KernelProgram"/> and encodes its launches
/// into a command buffer per execution, from a recording per distinct binding of the symbolic dimensions (pipelines, arguments, grids
/// and barrier placement resolved once). On unified memory there is no staging: inputs, runtime scalars and weights are written into
/// the buffers the kernels read, and outputs are read from the buffers they write.
/// </summary>
public sealed unsafe class MetalSegment : ICompiledSegment, IPartialOutputSegment, IDeviceForwardingSegment
{
    /// <summary>Most steps kept recorded per segment (one per distinct symbol binding).</summary>
    private const int MaxRecordings = 64;

    /// <summary>Most launches encoded into one command buffer; longer steps are committed as several command buffers in order.</summary>
    private const int MaxLaunchesPerCommandBuffer = 8192;

    /// <summary>Bytes of weight data copied per call (rows of repacked weights stay whole).</summary>
    private const long WeightChunkBytes = 256L << 20;

    private readonly MetalDevice _dev;
    private readonly KernelProgram _program;
    private readonly CompileOptions _options;
    private readonly Dictionary<BufferDecl, MetalBuffer> _alloc = [];
    /// <summary>Weight buffers held through <see cref="CompileOptions.SharedWeights"/> (their buffers are freed by the lease).</summary>
    private readonly Dictionary<BufferDecl, SharedWeightLease<MetalBuffer>> _leases = [];
    /// <summary>State buffers another segment of the same graph allocated (and frees).</summary>
    private readonly HashSet<BufferDecl> _borrowedStates = [];
    /// <summary>Inputs copied from an earlier segment's output buffer instead of from <see cref="SetInput"/> (<see cref="TryForwardFrom"/>).</summary>
    private readonly Dictionary<BufferDecl, (MetalBuffer Source, long SourceOffset)> _forwarded = [];
    private readonly Dictionary<string, MetalRecordedLaunch[]> _recorded = [];
    private readonly Queue<string> _recordOrder = new();
    /// <summary>Every buffer the kernels may address, made resident for each encoder (pointer arguments are indirect references).</summary>
    private nint[] _resources = [];
    private bool _disposed;

    internal MetalSegment(MetalDevice dev, GraphSegment segment, KernelProgram program, CompileOptions options)
    {
        _dev = dev;
        _program = program;
        _options = options;
        Segment = segment;
        try
        {
            long bytes = 0;
            foreach (var b in program.Buffers)
            {
                bytes += AllocateBuffer(b);
            }

            DeviceMemoryBytes = bytes;
            _resources = [.. _alloc.Values.Select(a => a.Handle).Distinct()];
            var uploadTimer = Stopwatch.StartNew();
            UploadWeights();
            if (DiagnosticSwitches.LoadTiming)
            {
                Console.Error.WriteLine($"[load] upload segment {segment.Index}: {uploadTimer.ElapsedMilliseconds} ms, {program.Buffers.Where(b => b.Kind == BufferKind.Weight).Sum(b => b.Bytes) / 1e9:F2} GB");
            }

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

    /// <summary>GPU time (first to last command) of the most recent <see cref="Execute(RunParameters)"/>, in ms.</summary>
    public double LastGpuMs { get; private set; }

    /// <summary>Per-launch GPU timing: every launch runs in its own command buffer. Defaults to on with <c>SHARPLIGHT_MTL_PROFILE=1</c>.</summary>
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
            var lease = shared.Acquire(_dev, WeightBufferKey.For(w, LayoutName(b)), b.Bytes, () => AllocateWeight(b), _dev.Free);
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

        _alloc[b] = b.Kind == BufferKind.Weight ? AllocateWeight(b) : _dev.Allocate(b.Bytes);
        sharedStates?.Add(b.State!, _alloc[b]);
        return _alloc[b].WrapsHostMemory ? 0 : b.Bytes;
    }

    /// <summary>
    /// The buffer of a weight: the mapped file pages themselves when the weight is stored as the kernels read it (no repack) and its
    /// source is memory-mapped, otherwise a new buffer the upload fills.
    /// </summary>
    private MetalBuffer AllocateWeight(BufferDecl b)
    {
        if (DiagnosticSwitches.NoCopyWeights && b.Repack == WeightRepack.None && b.InitialData is null && b.Weight is { } w
            && w.ByteSize == b.Bytes && w.Source?.TryGetPointer() is { } pointer && pointer != 0)
        {
            var wrapped = w.Source.TryGetMapping(out nint mappingBase, out long mappingLength)
                ? _dev.TryWrapMapping(pointer, b.Bytes, mappingBase, mappingLength)
                : _dev.TryWrap(pointer, b.Bytes);
            if (wrapped is not null)
            {
                return wrapped;
            }
        }

        return _dev.Allocate(b.Bytes);
    }

    /// <summary>Device layout of a weight buffer in <see cref="WeightBufferKey"/>: empty for the bytes as stored, else the repack.</summary>
    private static string LayoutName(BufferDecl b) => b.Repack == WeightRepack.None ? "" : $"metal:{b.Repack}:{b.RepackRowElements}";

    /// <summary>Copies the weights and constants into their buffers (unified memory: no staging, no GPU copy).</summary>
    private void UploadWeights()
    {
        var weights = _program.Buffers.Where(b => b.Kind == BufferKind.Weight && !_alloc[b].WrapsHostMemory
            && (!_leases.TryGetValue(b, out var l) || l.MustFill)).ToList();
        var ct = _options.CancellationToken;
        if (weights.Count > 0)
        {
            var materialized = WeightCopier.MaterializeInBackground(weights, ct);
            var progress = new LoadProgressCounter(_options.Progress, LoadStage.Uploading, weights.Sum(b => b.Bytes), _dev.Info.Name);
            try
            {
                foreach (var buffer in weights)
                {
                    ct.ThrowIfCancellationRequested();
                    CopyWeight(buffer, _alloc[buffer]);
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

    private static void CopyWeight(BufferDecl buffer, MetalBuffer destination)
    {
        long total = buffer.Weight?.ByteSize ?? buffer.InitialData!.Length;
        var source = buffer.Weight?.Source;
        nint pointer = source?.TryGetPointer() ?? 0;
        long rowBytes = buffer.Repack != WeightRepack.None ? WeightRepacker.RowBytes(buffer.Repack, buffer.RepackRowElements) : 1;
        long chunkBytes = Math.Max(rowBytes, WeightChunkBytes - WeightChunkBytes % rowBytes);
        for (long done = 0; done < total; done += chunkBytes)
        {
            long chunk = Math.Min(chunkBytes, total - done);
            WeightCopier.CopyChunk(buffer, source, pointer, done, chunk, destination.Contents + done, rowBytes);
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

        data.CopyTo(new Span<byte>(a.Contents + r.ByteOffset, data.Length));
    }

    /// <inheritdoc />
    /// <remarks>The producer's output buffer is copied into the input buffer by the CPU at the start of each execution (unified memory).</remarks>
    public bool TryForwardFrom(Value input, ICompiledSegment producer)
    {
        if (!DiagnosticSwitches.DeviceForwarding || producer is not MetalSegment source || source._dev != _dev
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
        var r = _program.OutputRefs[value];
        new ReadOnlySpan<byte>(_alloc[r.Buffer].Contents + r.ByteOffset + byteOffset, destination.Length).CopyTo(destination);
    }

    // ---- execution ---------------------------------------------------------------------------
    /// <inheritdoc />
    public void Execute(RunParameters parameters)
    {
        var slots = new Span<uint>(_alloc[_program.RuntimeBuffer].Contents, 64);
        foreach (var (name, slot) in _program.ScalarSlots)
        {
            slots[slot] = parameters.Scalars.TryGetValue(name, out var v) ? (uint)v
                : throw new ArgumentException($"Runtime scalar '{name}' not provided.");
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
                Console.Error.WriteLine($"[mtl] recorded {key}: {recording.Length} launches, {recording.Count(l => l.BarrierBefore)} barriers");
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
            int count = *(int*)(_alloc[slot.Buffer].Contents + slot.ByteOffset);
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

            long size = UsedBytes(r, bindings);
            var destination = _alloc[r.Buffer];
            Buffer.MemoryCopy(f.Source.Contents + r.ByteOffset + f.SourceOffset, destination.Contents + r.ByteOffset, destination.Bytes - r.ByteOffset, size);
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
    public void ResetState() => RunLaunches(_program.ResetLaunches);

    private void RunLaunches(IReadOnlyList<KernelLaunch> launches)
    {
        if (launches.Count == 0)
        {
            return;
        }

        Run(Record(launches, new Bindings(new Dictionary<string, long>())));
    }

    /// <summary>Resolves the launches for <paramref name="bindings"/>: variant, pipeline, packed arguments, grid and barriers.</summary>
    private MetalRecordedLaunch[] Record(IReadOnlyList<KernelLaunch> launches, Bindings bindings)
    {
        var recorded = new List<MetalRecordedLaunch>(launches.Count);
        foreach (var (launch, _, variant, barrierBefore) in HazardTracker.Order(launches, bindings, reorder: !DiagnosticSwitches.Serial))
        {
            var grid = variant.Grid(bindings);
            if (grid.X == 0 || grid.Y == 0 || grid.Z == 0)
            {
                continue;
            }

            var pipeline = _dev.Pipeline(variant.Kernel);
            recorded.Add(new MetalRecordedLaunch(launch.Name, pipeline.Handle, new MTLSize(grid.X, grid.Y, grid.Z), pipeline.Threads,
                PackArguments(launch, bindings), barrierBefore));
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
                    *(ulong*)(p + offset) = _alloc[pointer.Buffer].GpuAddress + (ulong)pointer.ByteOffset;
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

    /// <summary>Encodes and runs <paramref name="launches"/>, waits for them and returns the GPU time in ms.</summary>
    private double Run(ReadOnlySpan<MetalRecordedLaunch> launches)
    {
        using var pool = AutoreleasePool.Push();
        nint first = 0;
        nint last = 0;
        int start = 0;
        do
        {
            int count = Math.Min(MaxLaunchesPerCommandBuffer, launches.Length - start);
            last = Encode(launches.Slice(start, count));
            first = first == 0 ? last : first;
            ObjCRuntime.SendVoid(last, Selectors.Commit);
            start += count;
        }
        while (start < launches.Length);

        ObjCRuntime.SendVoid(last, Selectors.WaitUntilCompleted);
        CheckCompleted(last);
        CheckCompleted(first);
        return (ObjCRuntime.SendDouble(last, Selectors.GpuEndTime) - ObjCRuntime.SendDouble(first, Selectors.GpuStartTime)) * 1000;
    }

    /// <summary>Runs each launch in its own command buffer and accumulates its GPU time in <see cref="ProfileData"/>.</summary>
    private void RunProfiled(MetalRecordedLaunch[] launches, string key)
    {
        double total = 0;
        foreach (var launch in launches)
        {
            double ms = Run(new ReadOnlySpan<MetalRecordedLaunch>(in launch));
            total += ms;
            string name = $"{launch.Name} {key}";
            ProfileData[name] = ProfileData.TryGetValue(name, out var entry)
                ? (entry.Ms + ms, entry.Count + 1, Math.Min(entry.MinMs, ms))
                : (ms, 1, ms);
        }

        LastGpuMs = total;
    }

    /// <summary>Encodes <paramref name="launches"/> into a new (autoreleased) command buffer, not yet committed.</summary>
    private nint Encode(ReadOnlySpan<MetalRecordedLaunch> launches)
    {
        nint commandBuffer = ObjCRuntime.Send(_dev.Queue, Selectors.CommandBufferWithUnretainedReferences);
        nint encoder = ObjCRuntime.SendIndex(commandBuffer, Selectors.ComputeCommandEncoderWithDispatchType,
            DiagnosticSwitches.Serial ? MetalConstants.DispatchSerial : MetalConstants.DispatchConcurrent);
        fixed (nint* resources = _resources)
        {
            ObjCRuntime.SendVoid(encoder, Selectors.UseResources, resources, (nuint)_resources.Length, MetalConstants.UsageReadWrite);
        }

        nint bound = 0;
        bool firstLaunch = true;
        foreach (var launch in launches)
        {
            if (launch.BarrierBefore && !firstLaunch && !DiagnosticSwitches.Serial)
            {
                ObjCRuntime.SendVoidIndex(encoder, Selectors.MemoryBarrierWithScope, MetalConstants.BarrierScopeBuffers);
            }

            firstLaunch = false;
            if (launch.Pipeline != bound)
            {
                ObjCRuntime.SendVoid(encoder, Selectors.SetComputePipelineState, launch.Pipeline);
                bound = launch.Pipeline;
            }

            fixed (byte* arguments = launch.Arguments)
            {
                ObjCRuntime.SendVoid(encoder, Selectors.SetBytes, arguments, (nuint)launch.Arguments.Length, 0);
            }

            ObjCRuntime.SendVoid(encoder, Selectors.DispatchThreadgroups, launch.Threadgroups, launch.Threads);
        }

        ObjCRuntime.SendVoid(encoder, Selectors.EndEncoding);
        return commandBuffer;
    }

    private static void CheckCompleted(nint commandBuffer)
    {
        ulong status = ObjCRuntime.SendUInt64(commandBuffer, Selectors.Status);
        if (status == MetalConstants.CommandBufferError)
        {
            throw new InvalidOperationException("Metal command buffer failed: " + Foundation.ErrorText(ObjCRuntime.Send(commandBuffer, Selectors.Error), "unknown error"));
        }
    }

    /// <summary>Number of dispatches and barriers of a step with the given token count.</summary>
    /// <param name="tokens">Value of every symbol.</param>
    public (int Dispatches, int Barriers) Stats(long tokens)
    {
        var bindings = new Bindings(_program.Symbols.ToDictionary(s => s, _ => tokens));
        var order = HazardTracker.Order(_program.Launches, bindings, reorder: !DiagnosticSwitches.Serial);
        return (order.Count, order.Count(o => o.BarrierBefore));
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        // Every execution waits for its command buffers, so no work of this segment is in flight.
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
        _resources = [];
    }
}
