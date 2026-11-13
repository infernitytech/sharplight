using System.Diagnostics;
using System.Runtime.InteropServices;
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
public sealed unsafe class VulkanSegment : ICompiledSegment
{
    readonly VulkanDevice _dev;
    readonly KernelProgram _program;
    readonly Dictionary<BufferDecl, Allocation> _alloc = [];
    /// <summary>Host-visible mirrors of Input/Runtime (upload) and Output (readback) buffers. Kernels only
    /// touch device-local memory; the command buffer copies staging ↔ device at its start/end.</summary>
    readonly Dictionary<BufferDecl, Allocation> _staging = [];
    readonly Dictionary<string, VkCommandBuffer> _recorded = [];
    readonly CompileOptions _options;
    readonly VkFence _fence;
    readonly VkQueryPool _stepPool;
    bool _disposed;

    /// <summary>GPU time (first to last command) of the most recent non-profiled <see cref="Execute"/>, in ms.</summary>
    public double LastGpuMs { get; private set; }

    public GraphSegment Segment { get; }
    public KernelProgram Program => _program;
    public long DeviceMemoryBytes { get; }

    /// <summary>
    /// Per-dispatch GPU timing. When on, every Execute records a fresh command buffer with
    /// timestamps and accumulates <see cref="ProfileData"/>. Defaults to SHARPLIGHT_VK_PROFILE=1.
    /// </summary>
    public bool Profile { get; set; } = Environment.GetEnvironmentVariable("SHARPLIGHT_VK_PROFILE") is "1" or "2";

    /// <summary>Accumulated GPU time per launch kind (key: "&lt;op&gt; [shape] &lt;entry&gt; T=&lt;tokens&gt;").</summary>
    public Dictionary<string, (double Ms, int Count, double MinMs)> ProfileData { get; } = [];
    static readonly bool s_profileDetail = Environment.GetEnvironmentVariable("SHARPLIGHT_VK_PROFILE") == "2";
    public void ResetProfile() => ProfileData.Clear();

    /// <summary>Formats <see cref="ProfileData"/> as a table, heaviest first.</summary>
    public string ProfileReport(int top = 30)
    {
        double total = ProfileData.Values.Sum(v => v.Ms);
        var sb = new System.Text.StringBuilder($"GPU time {total:F2} ms\n");
        foreach (var (k, v) in ProfileData.OrderByDescending(kv => kv.Value.Ms).Take(top))
            sb.AppendLine($"{v.Ms,10:F3} ms {100 * v.Ms / total,5:F1}% {v.Count,6}x  min {v.MinMs * 1000,8:F1} us  {k}");
        return sb.ToString();
    }

    internal VulkanSegment(VulkanDevice dev, GraphSegment segment, KernelProgram program, CompileOptions options)
    {
        _dev = dev;
        _program = program;
        _options = options;
        Segment = segment;

        var fci = new VkFenceCreateInfo();
        VkFence f;
        dev.Api.vkCreateFence(&fci, null, &f).CheckResult();
        _fence = f;
        var qci = new VkQueryPoolCreateInfo { queryType = VkQueryType.Timestamp, queryCount = 2 };
        VkQueryPool qp;
        _dev.Api.vkCreateQueryPool(&qci, null, &qp).CheckResult();
        _stepPool = qp;

        long bytes = 0;
        try
        {
            foreach (var b in program.Buffers)
            {
                ulong size = (ulong)Math.Max(16, b.Bytes);
                _alloc[b] = dev.Memory.Allocate(size, MemoryUsage.Device);
                bytes += b.Bytes;
                if (b.Kind is BufferKind.Input or BufferKind.Runtime) _staging[b] = dev.Memory.Allocate(size, MemoryUsage.Upload);
                else if (b.Kind == BufferKind.Output) _staging[b] = dev.Memory.Allocate(size, MemoryUsage.Readback);
            }
            DeviceMemoryBytes = bytes;
            var swUp = Stopwatch.StartNew();
            UploadWeights();
            if (VulkanDevice.s_loadTiming) Console.Error.WriteLine($"[load] upload segment {segment.Index}: {swUp.ElapsedMilliseconds} ms, {program.Buffers.Where(b => b.Kind == BufferKind.Weight).Sum(b => b.Bytes) / 1e9:F2} GB");
            RunLaunches(program.ResetLaunches);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    // ---- weights ---------------------------------------------------------------------------
    void UploadWeights()
    {
        var weights = _program.Buffers.Where(b => b.Kind == BufferKind.Weight).ToList();
        if (weights.Count == 0) return;
        const ulong StagingSize = 128UL << 20;
        var staging = new[] { _dev.Memory.Allocate(StagingSize, MemoryUsage.Upload), _dev.Memory.Allocate(StagingSize, MemoryUsage.Upload) };
        var cbs = new[] { _dev.AllocateCommandBuffer(), _dev.AllocateCommandBuffer() };
        var fences = new VkFence[2];
        for (int i = 0; i < 2; i++)
        {
            var fci = new VkFenceCreateInfo { flags = VkFenceCreateFlags.Signaled };
            VkFence f;
            _dev.Api.vkCreateFence(&fci, null, &f).CheckResult();
            fences[i] = f;
        }
        try
        {
            int slot = 0;
            ulong fill = 0;
            var pending = new List<(VkBuffer Dst, ulong DstOff, ulong SrcOff, ulong Size)>();

            void Flush()
            {
                if (pending.Count == 0) return;
                var cb = cbs[slot];
                var bi = new VkCommandBufferBeginInfo { flags = VkCommandBufferUsageFlags.OneTimeSubmit };
                _dev.Api.vkBeginCommandBuffer(cb, &bi).CheckResult();
                foreach (var (dst, dstOff, srcOff, size) in pending)
                {
                    var region = new VkBufferCopy { srcOffset = staging[slot].Offset + srcOff, dstOffset = dstOff, size = size };
                    _dev.Api.vkCmdCopyBuffer(cb, staging[slot].Buffer, dst, 1, &region);
                }
                _dev.Api.vkEndCommandBuffer(cb).CheckResult();
                var f = fences[slot];
                _dev.Api.vkResetFences(1, &f).CheckResult();
                var si = new VkSubmitInfo { commandBufferCount = 1, pCommandBuffers = &cb };
                _dev.Api.vkQueueSubmit(_dev.Queue, 1, &si, f).CheckResult();
                pending.Clear();
                slot ^= 1;
                fill = 0;
                var nf = fences[slot];
                _dev.Api.vkWaitForFences(1, &nf, true, ulong.MaxValue).CheckResult();
            }

            foreach (var b in weights)
            {
                var dst = _alloc[b];
                long total = b.Weight?.ByteSize ?? b.InitialData!.Length;
                var src = b.Weight?.Source;
                nint ptr = src?.TryGetPointer() ?? 0;
                long done = 0;
                // Repacked weights are transformed in whole rows.
                long rowBytes = b.Repack != WeightRepack.None ? WeightRepacker.RowBytes(b.Repack, b.RepackRowElements) : 1;
                while (done < total)
                {
                    if (StagingSize - fill < (ulong)Math.Min(rowBytes, total - done)) Flush();
                    ulong chunk = Math.Min(StagingSize - fill, (ulong)(total - done));
                    chunk -= chunk % (ulong)rowBytes;
                    var span = new Span<byte>(staging[slot].Mapped + fill, (int)chunk);
                    if (b.Repack != WeightRepack.None && ptr != 0)
                        RepackParallel(b, (byte*)ptr + done, staging[slot].Mapped + fill, (long)chunk, rowBytes);
                    else if (b.Repack != WeightRepack.None)
                    {
                        var tmp = System.Buffers.ArrayPool<byte>.Shared.Rent((int)chunk);
                        try
                        {
                            if (ptr != 0) new ReadOnlySpan<byte>((byte*)ptr + done, (int)chunk).CopyTo(tmp);
                            else if (b.InitialData is not null) b.InitialData.AsSpan((int)done, (int)chunk).CopyTo(tmp);
                            else src!.Read(done, tmp.AsSpan(0, (int)chunk));
                            RepackParallel(b, tmp, staging[slot].Mapped + fill, (long)chunk, rowBytes);
                        }
                        finally { System.Buffers.ArrayPool<byte>.Shared.Return(tmp); }
                    }
                    else if (b.InitialData is not null) b.InitialData.AsSpan((int)done, (int)chunk).CopyTo(span);
                    else if (ptr != 0) CopyParallel((byte*)ptr + done, staging[slot].Mapped + fill, (long)chunk);
                    else src!.Read(done, span);
                    pending.Add((dst.Buffer, dst.Offset + (ulong)done, fill, chunk));
                    fill += (chunk + 255) & ~255UL;
                    if (fill > StagingSize) fill = StagingSize;
                    done += (long)chunk;
                }
            }
            Flush();
            for (int i = 0; i < 2; i++) { var f = fences[i]; _dev.Api.vkWaitForFences(1, &f, true, ulong.MaxValue); }
        }
        finally
        {
            for (int i = 0; i < 2; i++)
            {
                _dev.Api.vkDestroyFence(fences[i], null);
                _dev.FreeCommandBuffer(cbs[i]);
                _dev.Memory.Free(staging[i]);
            }
        }
    }

    /// <summary>Multi-threaded copy: weight sources are usually mmapped files, where a single thread is
    /// bound by page-fault handling rather than memory bandwidth.</summary>
    static void CopyParallel(byte* src, byte* dst, long bytes)
    {
        const long Piece = 2 << 20;
        if (bytes <= Piece) { Buffer.MemoryCopy(src, dst, bytes, bytes); return; }
        nint s = (nint)src, d = (nint)dst;
        Parallel.For(0, (bytes + Piece - 1) / Piece, i =>
        {
            long o = i * Piece, n = Math.Min(Piece, bytes - o);
            Buffer.MemoryCopy((byte*)s + o, (byte*)d + o, n, n);
        });
    }

    static void RepackParallel(BufferDecl b, byte* src, byte* dst, long bytes, long rowBytes)
    {
        long rows = bytes / rowBytes;
        const long RowsPerTask = 256;
        nint s = (nint)src, d = (nint)dst;
        Parallel.For(0, (rows + RowsPerTask - 1) / RowsPerTask, t =>
        {
            long r0 = t * RowsPerTask, n = Math.Min(RowsPerTask, rows - r0);
            WeightRepacker.Apply(b.Repack, b.RepackRowElements,
                new ReadOnlySpan<byte>((byte*)s + r0 * rowBytes, (int)(n * rowBytes)),
                new Span<byte>((byte*)d + r0 * rowBytes, (int)(n * rowBytes)));
        });
    }

    static void RepackParallel(BufferDecl b, byte[] src, byte* dst, long bytes, long rowBytes)
    {
        long rows = bytes / rowBytes;
        const long RowsPerTask = 256;
        nint d = (nint)dst;
        Parallel.For(0, (rows + RowsPerTask - 1) / RowsPerTask, t =>
        {
            long r0 = t * RowsPerTask, n = Math.Min(RowsPerTask, rows - r0);
            WeightRepacker.Apply(b.Repack, b.RepackRowElements,
                src.AsSpan((int)(r0 * rowBytes), (int)(n * rowBytes)),
                new Span<byte>((byte*)d + r0 * rowBytes, (int)(n * rowBytes)));
        });
    }

    // ---- I/O ---------------------------------------------------------------------------------
    public void SetInput(Value value, ReadOnlySpan<byte> data)
    {
        if (!_program.InputRefs.TryGetValue(value, out var r)) return;
        var a = _staging[r.Buffer];
        if ((ulong)data.Length > a.Size - (ulong)r.ByteOffset)
            throw new ArgumentException($"Input {value.Name}: {data.Length} bytes exceeds the compiled maximum ({r.Buffer.Bytes}). Increase CompileOptions.MaxTokens.");
        data.CopyTo(new Span<byte>(a.Mapped + r.ByteOffset, data.Length));
    }

    public void GetOutput(Value value, Span<byte> destination)
    {
        var r = _program.OutputRefs[value];
        var a = _staging[r.Buffer];
        new ReadOnlySpan<byte>(a.Mapped + r.ByteOffset, destination.Length).CopyTo(destination);
    }

    /// <summary>Direct read-only view of an output (valid until the next Execute).</summary>
    public ReadOnlySpan<byte> OutputSpan(Value value, int bytes)
    {
        var r = _program.OutputRefs[value];
        return new ReadOnlySpan<byte>(_staging[r.Buffer].Mapped + r.ByteOffset, bytes);
    }

    // ---- execution ---------------------------------------------------------------------------
    public void Execute(RunParameters parameters)
    {
        var rt = _staging[_program.RuntimeBuffer];
        var slots = new Span<uint>(rt.Mapped, 64);
        foreach (var (name, slot) in _program.ScalarSlots)
            slots[slot] = parameters.Scalars.TryGetValue(name, out var v) ? (uint)v
                : throw new ArgumentException($"Runtime scalar '{name}' not provided.");

        var bindings = new Bindings(parameters.Symbols);
        if (Profile) { ExecuteProfiled(bindings); return; }
        var key = string.Join(",", _program.Symbols.Select(s => $"{s}={bindings[s]}"));
        if (!_recorded.TryGetValue(key, out var cb))
        {
            foreach (var s in _program.Symbols)
                if (bindings[s] > _options.Bound(s))
                    throw new ArgumentException($"Symbol {s}={bindings[s]} exceeds the compiled bound.");
            var swRec = Stopwatch.StartNew();
            long pt0 = _dev.PipelineTicks; int pc0 = _dev.PipelinesCreated;
            cb = _dev.AllocateCommandBuffer();
            var bi = new VkCommandBufferBeginInfo();
            _dev.Api.vkBeginCommandBuffer(cb, &bi).CheckResult();
            _dev.Api.vkCmdResetQueryPool(cb, _stepPool, 0, 2);
            _dev.Api.vkCmdWriteTimestamp(cb, VkPipelineStageFlags.TopOfPipe, _stepPool, 0);
            RecordInputCopies(cb, bindings);
            Record(cb, _program.Launches, bindings);
            RecordOutputCopies(cb, bindings);
            _dev.Api.vkCmdWriteTimestamp(cb, VkPipelineStageFlags.BottomOfPipe, _stepPool, 1);
            _dev.Api.vkEndCommandBuffer(cb).CheckResult();
            _recorded[key] = cb;
            if (VulkanDevice.s_loadTiming) Console.Error.WriteLine($"[load] record {key}: {swRec.ElapsedMilliseconds} ms, of which {_dev.PipelinesCreated - pc0} pipelines {(double)(_dev.PipelineTicks - pt0) / Stopwatch.Frequency * 1000:F0} ms");
        }
        var fence = _fence;
        _dev.Api.vkResetFences(1, &fence).CheckResult();
        var si = new VkSubmitInfo { commandBufferCount = 1, pCommandBuffers = &cb };
        _dev.Api.vkQueueSubmit(_dev.Queue, 1, &si, fence).CheckResult();
        _dev.Api.vkWaitForFences(1, &fence, true, ulong.MaxValue).CheckResult();
        ulong* ts = stackalloc ulong[2];
        if (_dev.Api.vkGetQueryPoolResults(_stepPool, 0, 2, 16, ts, 8, VkQueryResultFlags.Bit64) == VkResult.Success)
            LastGpuMs = (ts[1] - ts[0]) * _dev.Phys.TimestampPeriod * 1e-6;
    }

    void ExecuteProfiled(Bindings bindings)
    {
        var api = _dev.Api;
        uint n = (uint)_program.Launches.Count + 1;
        var qci = new VkQueryPoolCreateInfo { queryType = VkQueryType.Timestamp, queryCount = n };
        VkQueryPool created;
        api.vkCreateQueryPool(&qci, null, &created).CheckResult();
        var pool = created;
        try
        {
            var names = new string[_program.Launches.Count];
            _dev.Immediate(cb =>
            {
                api.vkCmdResetQueryPool(cb, pool, 0, n);
                RecordInputCopies(cb, bindings);
                Record(cb, _program.Launches, bindings, (i, l, v) =>
                {
                    if (i < 0) { api.vkCmdWriteTimestamp(cb, VkPipelineStageFlags.ComputeShader, pool, 0); return; }
                    api.vkCmdWriteTimestamp(cb, VkPipelineStageFlags.ComputeShader, pool, (uint)i + 1);
                    if (v is null) return;
                    var parts = l.Name.Split(' ');
                    names[i] = s_profileDetail
                        ? $"{l.Name} {v.Kernel.EntryPoint}"
                        : $"{parts[0]}{(parts.Length > 2 ? " " + string.Join(" ", parts[2..]) : "")} {v.Kernel.EntryPoint} T={string.Join(",", _program.Symbols.Select(s => bindings[s]))}";
                });
                RecordOutputCopies(cb, bindings);
            });
            var ts = new ulong[n];
            fixed (ulong* t = ts)
                api.vkGetQueryPoolResults(pool, 0, n, (nuint)(n * 8), t, 8, VkQueryResultFlags.Bit64 | VkQueryResultFlags.Wait).CheckResult();
            double tick = _dev.Phys.TimestampPeriod * 1e-6;
            for (int i = 0; i < names.Length; i++)
            {
                if (names[i] is null) continue;
                double ms = (ts[i + 1] - ts[i]) * tick;
                ProfileData[names[i]] = ProfileData.TryGetValue(names[i], out var e) ? (e.Ms + ms, e.Count + 1, Math.Min(e.MinMs, ms)) : (ms, 1, ms);
            }
        }
        finally { api.vkDestroyQueryPool(pool, null); }
    }

    public void ResetState() => RunLaunches(_program.ResetLaunches);

    void RunLaunches(IReadOnlyList<KernelLaunch> launches)
    {
        if (launches.Count == 0) return;
        _dev.Immediate(cb => Record(cb, launches, new Bindings(new Dictionary<string, long>())));
    }

    // Diagnostics only: SHARPLIGHT_VK_NO_BARRIERS=1 drops inter-dispatch barriers (wrong results) to measure their cost.
    static readonly bool s_noBarriers = Environment.GetEnvironmentVariable("SHARPLIGHT_VK_NO_BARRIERS") == "1";

    void Record(VkCommandBuffer cb, IReadOnlyList<KernelLaunch> launches, Bindings bindings,
        Action<int, KernelLaunch, KernelVariant>? afterDispatch = null)
    {
        afterDispatch?.Invoke(-1, null!, null!);
        int index = 0;
        var api = _dev.Api;
        var pendingWrites = new List<BufferRange>();
        var pendingReads = new List<BufferRange>();
        byte* pc = stackalloc byte[(int)_dev.PushConstantSize];
        VkPipeline bound = VkPipeline.Null;

        foreach (var launch in launches)
        {
            var variant = launch.Select(bindings);
            if (variant is null)
            {
                // Conditional launch not needed for these bindings (e.g. a fused small-batch kernel at large T).
                afterDispatch?.Invoke(index, launch, null!);
                index++;
                continue;
            }
            bool hazard = afterDispatch is not null // profiling: serialize so each timestamp delta is one kernel
                       || launch.Reads.Any(r => pendingWrites.Any(w => w.Overlaps(r)))
                       || launch.Writes.Any(w => pendingWrites.Any(p => p.Overlaps(w)) || pendingReads.Any(p => p.Overlaps(w)));
            if (hazard && !s_noBarriers)
            {
                Barrier(cb, VkPipelineStageFlags.ComputeShader, VkAccessFlags.ShaderWrite,
                    VkPipelineStageFlags.ComputeShader, VkAccessFlags.ShaderRead | VkAccessFlags.ShaderWrite);
                pendingWrites.Clear();
                pendingReads.Clear();
            }
            pendingWrites.AddRange(launch.Writes);
            pendingReads.AddRange(launch.Reads);

            var pipe = _dev.Pipeline(variant.Kernel);
            if (pipe != bound) { api.vkCmdBindPipeline(cb, VkPipelineBindPoint.Compute, pipe); bound = pipe; }

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
            if (off > _dev.PushConstantSize) throw new InvalidOperationException($"{launch.Name}: {off} bytes of arguments exceed push constant space.");
            api.vkCmdPushConstants(cb, _dev.PipelineLayout, VkShaderStageFlags.Compute, 0, (uint)((off + 3) & ~3), pc);
            var g = variant.Grid(bindings);
            api.vkCmdDispatch(cb, g.X, g.Y, g.Z);
            afterDispatch?.Invoke(index, launch, variant);
            index++;
        }
        // Make results visible to the host.
        Barrier(cb, VkPipelineStageFlags.ComputeShader, VkAccessFlags.ShaderWrite, VkPipelineStageFlags.Host, VkAccessFlags.HostRead);
    }

    /// <summary>Bytes of a staged tensor actually used under the given symbol bindings.</summary>
    static ulong UsedBytes(TensorRef r, Bindings b)
    {
        long elems = r.MaxElements;
        if (r.Symbol is not null && r.Shape.Length > 0 && r.Shape[0] > 0) elems = elems / r.Shape[0] * Math.Min(r.Shape[0], b[r.Symbol]);
        return (ulong)r.Type.ByteSize(elems);
    }

    void Copy(VkCommandBuffer cb, Allocation src, Allocation dst, ulong offset, ulong size)
    {
        if (size == 0) return;
        var region = new VkBufferCopy { srcOffset = src.Offset + offset, dstOffset = dst.Offset + offset, size = Math.Min(size, Math.Min(src.Size, dst.Size) - offset) };
        _dev.Api.vkCmdCopyBuffer(cb, src.Buffer, dst.Buffer, 1, &region);
    }

    void RecordInputCopies(VkCommandBuffer cb, Bindings b)
    {
        Copy(cb, _staging[_program.RuntimeBuffer], _alloc[_program.RuntimeBuffer], 0, 256);
        foreach (var r in _program.InputRefs.Values)
            Copy(cb, _staging[r.Buffer], _alloc[r.Buffer], (ulong)r.ByteOffset, UsedBytes(r, b));
        Barrier(cb, VkPipelineStageFlags.Transfer, VkAccessFlags.TransferWrite,
            VkPipelineStageFlags.ComputeShader, VkAccessFlags.ShaderRead | VkAccessFlags.ShaderWrite);
    }

    void RecordOutputCopies(VkCommandBuffer cb, Bindings b)
    {
        Barrier(cb, VkPipelineStageFlags.ComputeShader, VkAccessFlags.ShaderWrite, VkPipelineStageFlags.Transfer, VkAccessFlags.TransferRead);
        foreach (var r in _program.OutputRefs.Values)
            Copy(cb, _alloc[r.Buffer], _staging[r.Buffer], (ulong)r.ByteOffset, UsedBytes(r, b));
        Barrier(cb, VkPipelineStageFlags.Transfer, VkAccessFlags.TransferWrite, VkPipelineStageFlags.Host, VkAccessFlags.HostRead);
    }

    void Barrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkAccessFlags srcAccess, VkPipelineStageFlags dst, VkAccessFlags dstAccess)
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

    /// <summary>Launch order with the barriers <see cref="Execute"/> places (for diagnostics).</summary>
    public List<(string Name, bool BarrierBefore)> Schedule(long tokens = 1)
    {
        var list = new List<(string, bool)>();
        var pw = new List<BufferRange>(); var pr = new List<BufferRange>();
        var bindings = new Bindings(_program.Symbols.ToDictionary(s => s, _ => tokens));
        foreach (var l in _program.Launches)
        {
            if (l.Select(bindings) is null) continue;
            bool b = l.Reads.Any(r => pw.Any(w => w.Overlaps(r))) || l.Writes.Any(w => pw.Any(p => p.Overlaps(w)) || pr.Any(p => p.Overlaps(w)));
            if (b) { pw.Clear(); pr.Clear(); }
            pw.AddRange(l.Writes); pr.AddRange(l.Reads);
            list.Add((l.Name, b));
        }
        return list;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _dev.Api.vkDeviceWaitIdle();
        foreach (var cb in _recorded.Values) _dev.FreeCommandBuffer(cb);
        _recorded.Clear();
        foreach (var a in _alloc.Values) _dev.Memory.Free(a);
        _alloc.Clear();
        foreach (var a in _staging.Values) _dev.Memory.Free(a);
        _staging.Clear();
        _dev.Api.vkDestroyFence(_fence, null);
        _dev.Api.vkDestroyQueryPool(_stepPool, null);
    }
}
