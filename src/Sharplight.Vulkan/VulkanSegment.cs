using System.Diagnostics;
using Sharplight.Graph;
using Sharplight.Models;
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
public sealed unsafe class VulkanSegment : ICompiledSegment, IPartialOutputSegment
{
    private readonly VulkanDevice _dev;
    private readonly KernelProgram _program;
    private readonly Dictionary<BufferDecl, Allocation> _alloc = [];
    /// <summary>Host-visible mirrors of Input/Runtime (upload) and Output (readback) buffers. Kernels only
    /// touch device-local memory; the command buffer copies staging ↔ device at its start/end.</summary>
    private readonly Dictionary<BufferDecl, Allocation> _staging = [];
    private readonly Dictionary<string, VkCommandBuffer> _recorded = [];
    private readonly CompileOptions _options;
    private readonly VkFence _fence;
    /// <summary>This segment's own command pool (pools are externally synchronized; segments of different models run on different threads).</summary>
    private readonly VkCommandPool _pool;
    private readonly VkQueryPool _stepPool;
    private bool _disposed;

    /// <summary>GPU time (first to last command) of the most recent non-profiled <see cref="Execute"/>, in ms.</summary>
    public double LastGpuMs { get; private set; }

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
                if (b.Kind is BufferKind.Input or BufferKind.Runtime)
                {
                    _staging[b] = dev.Memory.Allocate(size, MemoryUsage.Upload);
                }
                else if (b.Kind == BufferKind.Output)
                {
                    _staging[b] = dev.Memory.Allocate(size, MemoryUsage.Readback);
                }
            }
            DeviceMemoryBytes = bytes;
            var swUp = Stopwatch.StartNew();
            UploadWeights();
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

    private VkCommandBuffer AllocCb()
    {
        var ai = new VkCommandBufferAllocateInfo { commandPool = _pool, level = VkCommandBufferLevel.Primary, commandBufferCount = 1 };
        VkCommandBuffer cb;
        _dev.Api.vkAllocateCommandBuffers(&ai, &cb).CheckResult();
        return cb;
    }

    private void FreeCb(VkCommandBuffer cb) => _dev.Api.vkFreeCommandBuffers(_pool, 1, &cb);

    // ---- weights ---------------------------------------------------------------------------
    private void UploadWeights()
    {
        var weights = _program.Buffers.Where(b => b.Kind == BufferKind.Weight).ToList();
        if (weights.Count == 0)
        {
            return;
        }

        const ulong StagingSize = 128UL << 20;
        var staging = new[] { _dev.Memory.Allocate(StagingSize, MemoryUsage.Upload), _dev.Memory.Allocate(StagingSize, MemoryUsage.Upload) };
        var cbs = new[] { AllocCb(), AllocCb() };
        var fences = new VkFence[2];
        var materialized = MaterializeInBackground(weights);
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
                if (pending.Count == 0)
                {
                    return;
                }

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
                _dev.Submit(cb, f);
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
                    if (StagingSize - fill < (ulong)Math.Min(rowBytes, total - done))
                    {
                        Flush();
                    }

                    ulong chunk = Math.Min(StagingSize - fill, (ulong)(total - done));
                    chunk -= chunk % (ulong)rowBytes;
                    CopyWeightChunk(b, src, ptr, done, (long)chunk, staging[slot].Mapped + fill, rowBytes);

                    pending.Add((dst.Buffer, dst.Offset + (ulong)done, fill, chunk));
                    fill += (chunk + 255) & ~255UL;
                    if (fill > StagingSize)
                    {
                        fill = StagingSize;
                    }

                    done += (long)chunk;
                }
            }
            Flush();
            for (int i = 0; i < 2; i++)
            {
                var f = fences[i];
                _dev.Api.vkWaitForFences(1, &f, true, ulong.MaxValue);
            }
        }
        finally
        {
            materialized.Wait();
            for (int i = 0; i < 2; i++)
            {
                _dev.Api.vkDestroyFence(fences[i], null);
                FreeCb(cbs[i]);
                _dev.Memory.Free(staging[i]);
            }
        }
    }

    /// <summary>
    /// Produces the weights that are computed on first read (layout conversions, folded constants) on several threads while the
    /// copy loop uploads the ones before them; mapped weights need no preparation.
    /// </summary>
    private static Task MaterializeInBackground(List<BufferDecl> weights)
    {
        const int Parallelism = 4;
        var lazy = weights.Select(b => b.Weight?.Source).OfType<IWeightSource>().Where(source => source.TryGetPointer() == 0).ToList();
        return Task.Run(() => Parallel.ForEach(lazy, new ParallelOptions { MaxDegreeOfParallelism = Parallelism }, source =>
        {
            try
            {
                source.Materialize();
            }
            catch (Exception)
            {
                // The producer's exception is cached and rethrown by the copy loop's read of this weight.
            }
        }));
    }

    /// <summary>
    /// Copies <paramref name="chunk"/> bytes of a weight (or constant) starting at <paramref name="done"/> into staging
    /// memory, repacking rows when the buffer asks for it. The source is a mapped pointer when it has one, otherwise
    /// the weight source or initial data is read through a span.
    /// </summary>
    private static void CopyWeightChunk(BufferDecl b, IWeightSource? src, nint ptr, long done, long chunk, byte* destination, long rowBytes)
    {
        var span = new Span<byte>(destination, (int)chunk);
        if (b.Repack != WeightRepack.None && ptr != 0)
        {
            RepackParallel(b, (byte*)ptr + done, destination, chunk, rowBytes);
        }
        else if (b.Repack != WeightRepack.None)
        {
            var tmp = System.Buffers.ArrayPool<byte>.Shared.Rent((int)chunk);
            try
            {
                if (b.InitialData is not null)
                {
                    b.InitialData.AsSpan((int)done, (int)chunk).CopyTo(tmp);
                }
                else
                {
                    src!.Read(done, tmp.AsSpan(0, (int)chunk));
                }

                RepackParallel(b, tmp, destination, chunk, rowBytes);
            }
            finally
            {
                System.Buffers.ArrayPool<byte>.Shared.Return(tmp);
            }
        }
        else if (b.InitialData is not null)
        {
            b.InitialData.AsSpan((int)done, (int)chunk).CopyTo(span);
        }
        else if (ptr != 0)
        {
            CopyParallel((byte*)ptr + done, destination, chunk);
        }
        else
        {
            src!.Read(done, span);
        }
    }

    /// <summary>Multi-threaded copy: weight sources are usually mmapped files, where a single thread is
    /// bound by page-fault handling rather than memory bandwidth.</summary>
    private static void CopyParallel(byte* src, byte* dst, long bytes)
    {
        const long Piece = 2 << 20;
        if (bytes <= Piece)
        {
            Buffer.MemoryCopy(src, dst, bytes, bytes);
            return;
        }
        nint s = (nint)src, d = (nint)dst;
        Parallel.For(0, (bytes + Piece - 1) / Piece, i =>
        {
            long o = i * Piece;
            long n = Math.Min(Piece, bytes - o);
            Buffer.MemoryCopy((byte*)s + o, (byte*)d + o, n, n);
        });
    }

    private static void RepackParallel(BufferDecl b, byte* src, byte* dst, long bytes, long rowBytes)
    {
        long rows = bytes / rowBytes;
        const long RowsPerTask = 256;
        nint s = (nint)src, d = (nint)dst;
        Parallel.For(0, (rows + RowsPerTask - 1) / RowsPerTask, t =>
        {
            long r0 = t * RowsPerTask;
            long n = Math.Min(RowsPerTask, rows - r0);
            WeightRepacker.Apply(b.Repack, b.RepackRowElements,
                new ReadOnlySpan<byte>((byte*)s + r0 * rowBytes, (int)(n * rowBytes)),
                new Span<byte>((byte*)d + r0 * rowBytes, (int)(n * rowBytes)));
        });
    }

    private static void RepackParallel(BufferDecl b, byte[] src, byte* dst, long bytes, long rowBytes)
    {
        long rows = bytes / rowBytes;
        const long RowsPerTask = 256;
        nint d = (nint)dst;
        Parallel.For(0, (rows + RowsPerTask - 1) / RowsPerTask, t =>
        {
            long r0 = t * RowsPerTask;
            long n = Math.Min(RowsPerTask, rows - r0);
            WeightRepacker.Apply(b.Repack, b.RepackRowElements,
                src.AsSpan((int)(r0 * rowBytes), (int)(n * rowBytes)),
                new Span<byte>((byte*)d + r0 * rowBytes, (int)(n * rowBytes)));
        });
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

        var a = _staging[r.Buffer];
        if ((ulong)data.Length > a.Size - (ulong)r.ByteOffset)
        {
            throw new ArgumentException($"Input {value.Name}: {data.Length} bytes exceeds the compiled maximum ({r.Buffer.Bytes}). Increase CompileOptions.MaxTokens.");
        }

        data.CopyTo(new Span<byte>(a.Mapped + r.ByteOffset, data.Length));
    }

    /// <inheritdoc />
    public void GetOutput(Value value, Span<byte> destination) => GetOutput(value, 0, destination);

    /// <inheritdoc />
    public void GetOutput(Value value, long byteOffset, Span<byte> destination)
    {
        var r = _program.OutputRefs[value];
        var a = _staging[r.Buffer];
        new ReadOnlySpan<byte>(a.Mapped + r.ByteOffset + byteOffset, destination.Length).CopyTo(destination);
    }

    // ---- execution ---------------------------------------------------------------------------
    /// <inheritdoc />
    public void Execute(RunParameters parameters)
    {
        var rt = _staging[_program.RuntimeBuffer];
        var slots = new Span<uint>(rt.Mapped, 64);
        foreach (var (name, slot) in _program.ScalarSlots)
        {
            slots[slot] = parameters.Scalars.TryGetValue(name, out var v) ? (uint)v
                : throw new ArgumentException($"Runtime scalar '{name}' not provided.");
        }

        var bindings = new Bindings(parameters.Symbols);
        if (Profile)
        {
            ExecuteProfiled(bindings);
            return;
        }
        var key = string.Join(",", _program.Symbols.Select(s => $"{s}={bindings[s]}"));
        if (_dev.SubmissionBudgetMs > 0)
        {
            ExecuteBudgeted(key, bindings);
            return;
        }
        if (!_recorded.TryGetValue(key, out var cb))
        {
            CheckWithinCompiledBounds(bindings);

            var swRec = Stopwatch.StartNew();
            long pt0 = _dev.PipelineTicks;
            int pc0 = _dev.PipelinesCreated;
            cb = AllocCb();
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
            if (DiagnosticSwitches.LoadTiming)
            {
                Console.Error.WriteLine($"[load] record {key}: {swRec.ElapsedMilliseconds} ms, of which {_dev.PipelinesCreated - pc0} pipelines {(double)(_dev.PipelineTicks - pt0) / Stopwatch.Frequency * 1000:F0} ms");
            }
        }
        var fence = _fence;
        _dev.Api.vkResetFences(1, &fence).CheckResult();
        _dev.Submit(cb, fence);
        _dev.Api.vkWaitForFences(1, &fence, true, ulong.MaxValue).CheckResult();
        ulong* ts = stackalloc ulong[2];
        if (_dev.Api.vkGetQueryPoolResults(_stepPool, 0, 2, 16, ts, 8, VkQueryResultFlags.Bit64) == VkResult.Success)
        {
            LastGpuMs = (ts[1] - ts[0]) * _dev.Phys.TimestampPeriod * 1e-6;
        }
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

    private void ExecuteProfiled(Bindings bindings)
    {
        var names = new string[_program.Launches.Count];
        var ms = MeasureLaunches(bindings, _dev.SubmissionBudgetMs > 0 ? CalibrationChunk : int.MaxValue, (i, l, v) =>
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
    private double[] MeasureLaunches(Bindings bindings, int chunk, Action<int, KernelLaunch, KernelVariant>? onLaunch = null)
    {
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
                    });
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
        _dev.Immediate(cb => RecordOutputCopies(cb, bindings));
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
    private void ExecuteBudgeted(string key, Bindings bindings)
    {
        CheckWithinCompiledBounds(bindings);

        var sw = Stopwatch.StartNew();
        if (!_plans.TryGetValue(key, out var plan))
        {
            var est = MeasureLaunches(bindings, CalibrationChunk);
            _plans[key] = BuildPlan(est, bindings);
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

            _plans[key] = BuildPlan(plan.Est, bindings);
        }
    }

    private Plan BuildPlan(double[] est, Bindings bindings)
    {
        var launches = _program.Launches;
        var plan = new Plan { Est = est };
        double budget = _dev.SubmissionBudgetMs;
        int start = 0;
        double sum = 0;
        for (int i = 0; i <= launches.Count; i++)
        {
            bool end = i == launches.Count;
            if (end || (i > start && sum + est[i] > budget))
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
                RecordOutputCopies(cb, bindings);
            }

            _dev.Api.vkEndCommandBuffer(cb).CheckResult();
            c.Cb = cb;
        }
        return plan;
    }

    /// <inheritdoc />
    public void ResetState() => RunLaunches(_program.ResetLaunches);

    private void RunLaunches(IReadOnlyList<KernelLaunch> launches)
    {
        if (launches.Count == 0)
        {
            return;
        }

        _dev.Immediate(cb => Record(cb, launches, new Bindings(new Dictionary<string, long>())));
    }

    private void Record(VkCommandBuffer cb, IReadOnlyList<KernelLaunch> launches, Bindings bindings,
        Action<int, KernelLaunch, KernelVariant>? afterDispatch = null)
    {
        afterDispatch?.Invoke(-1, null!, null!);
        int index = 0;
        var api = _dev.Api;
        var hazards = new HazardTracker();
        byte* pc = stackalloc byte[(int)_dev.PushConstantSize];
        VkPipeline bound = VkPipeline.Null;

        foreach (var launch in launches)
        {
            if (index >= DiagnosticSwitches.MaxLaunches)
            {
                break;
            }

            var variant = launch.Select(bindings);
            if (variant is null)
            {
                // Conditional launch not needed for these bindings (e.g. a fused small-batch kernel at large T).
                afterDispatch?.Invoke(index, launch, null!);
                index++;
                continue;
            }
            bool hazard = afterDispatch is not null // profiling: serialize so each timestamp delta is one kernel
                       || hazards.Conflicts(launch);
            if (hazard && !DiagnosticSwitches.NoBarriers)
            {
                Barrier(cb, VkPipelineStageFlags.ComputeShader, VkAccessFlags.ShaderWrite,
                    VkPipelineStageFlags.ComputeShader, VkAccessFlags.ShaderRead | VkAccessFlags.ShaderWrite);
                hazards.Clear();
            }

            hazards.Add(launch);

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
                Console.Error.WriteLine($"[launch {index}] {launch.Name} {variant.Kernel.CacheKey} grid {g.X}x{g.Y}x{g.Z}");
            }

            api.vkCmdDispatch(cb, g.X, g.Y, g.Z);
            afterDispatch?.Invoke(index, launch, variant);
            index++;
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
        foreach (var r in _program.InputRefs.Values)
        {
            Copy(cb, _staging[r.Buffer], _alloc[r.Buffer], (ulong)r.ByteOffset, UsedBytes(r, b));
        }

        Barrier(cb, VkPipelineStageFlags.Transfer, VkAccessFlags.TransferWrite,
            VkPipelineStageFlags.ComputeShader, VkAccessFlags.ShaderRead | VkAccessFlags.ShaderWrite);
    }

    private void RecordOutputCopies(VkCommandBuffer cb, Bindings b)
    {
        Barrier(cb, VkPipelineStageFlags.ComputeShader, VkAccessFlags.ShaderWrite, VkPipelineStageFlags.Transfer, VkAccessFlags.TransferRead);
        foreach (var r in _program.OutputRefs.Values)
        {
            Copy(cb, _alloc[r.Buffer], _staging[r.Buffer], (ulong)r.ByteOffset, UsedBytes(r, b));
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

    /// <summary>Launch order with the barriers <see cref="Execute"/> places (for diagnostics).</summary>
    public List<(string Name, bool BarrierBefore)> Schedule(long tokens = 1)
    {
        var list = new List<(string, bool)>();
        var hazards = new HazardTracker();
        var bindings = new Bindings(_program.Symbols.ToDictionary(s => s, _ => tokens));
        foreach (var l in _program.Launches)
        {
            if (l.Select(bindings) is null)
            {
                continue;
            }

            bool barrier = hazards.Conflicts(l);
            if (barrier)
            {
                hazards.Clear();
            }

            hazards.Add(l);
            list.Add((l.Name, barrier));
        }
        return list;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        // Every Execute waits for its fence, so no work of this segment is in flight; no device-wide wait (it would
        // stall, and need the queue lock against, the other models sharing the device).
        foreach (var cb in _recorded.Values)
        {
            FreeCb(cb);
        }

        _recorded.Clear();
        foreach (var p in _plans.Values)
        {
            foreach (var c in p.Chunks)
            {
                FreeCb(c.Cb);
            }
        }

        _plans.Clear();
        foreach (var a in _alloc.Values)
        {
            _dev.Memory.Free(a);
        }

        _alloc.Clear();
        foreach (var a in _staging.Values)
        {
            _dev.Memory.Free(a);
        }

        _staging.Clear();
        _dev.Api.vkDestroyFence(_fence, null);
        _dev.Api.vkDestroyQueryPool(_stepPool, null);
        _dev.Api.vkDestroyCommandPool(_pool, null);
    }
}
