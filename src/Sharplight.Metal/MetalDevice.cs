using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using Sharplight.Graph;
using Sharplight.Metal.Interop;
using Sharplight.Runtime;
using Sharplight.Shaders;
using Sharplight.Shaders.Lowering;

namespace Sharplight.Metal;

/// <summary>An opened Metal device: the GPU's command queue, buffers and compute pipelines.</summary>
internal sealed unsafe class MetalDevice : IComputeDevice
{
    private readonly ConcurrentDictionary<string, Lazy<MetalPipeline>> _pipelines = new();
    /// <summary>Records compilations that come without a model-level cache (graphs built in code) into the artifact store.</summary>
    private readonly CompilationArtifacts _ownArtifacts = new();
    /// <summary>Model-level cache key of the device stage's binary archive.</summary>
    internal const string BinaryArchiveKey = "metal/binary_archive";

    /// <summary>Artifact kind of the device stage.</summary>
    internal const string BinaryArchiveKind = "binary-archive";

    /// <summary>Version of the kernel argument layout, part of the device stage's identity.</summary>
    private const string KernelAbiVersion = "1";

    private readonly bool _kernelBinaries;
    private readonly Lock _archiveLock = new();
    private MetalBinaryArchive? _archive;

    /// <summary>Pipelines taken from the device stage's archive (diagnostics, tests).</summary>
    internal int PipelinesFromArchive;

    /// <summary>Stored device code that was rejected (damaged, other GPU or OS build).</summary>
    internal int RejectedDeviceCode;

    /// <summary>Wrapped file mappings by start address (see <see cref="TryWrapMapping"/>).</summary>
    private readonly Dictionary<nint, MappingEntry> _mappings = [];
    private readonly long _budget;
    /// <summary>The caps of <see cref="DeviceOptions.MemoryLimits"/> and the environment (Metal memory is unified: no BAR window, no pinned staging).</summary>
    private readonly DeviceMemoryLimits _limits;
    private long _used;
    private bool _disposed;

    internal MetalDevice(MetalDeviceProperties properties, DeviceInfo info, DeviceOptions options)
    {
        Properties = properties;
        Info = info;
        Options = options;
        EnabledFeatures = DeviceFeatures.Effective(info.Features, options.DisabledFeatures);
        ArtifactStore = ArtifactStore.For(options);
        Code = new MetalCodeGenerator(properties.MaxSharedMemory, EnabledFeatures,
            options.ResolveSubmissionBudgetMs(info.SubmissionBudgetMs),
            options.ResolveConcurrentWorkgroups(info.ConcurrentWorkgroups), 0);
        _kernelBinaries = !DeviceFeatures.DisabledByEnvironment.Contains(DeviceFeatures.KernelBinaries) && !options.DisabledFeatures.Contains(DeviceFeatures.KernelBinaries);
        ArtifactTargets = _kernelBinaries ? [Code.CreateCodeTarget(info.Provider), CreateArchiveTarget()] : [Code.CreateCodeTarget(info.Provider)];
        _limits = DeviceMemoryLimits.For(options);
        _budget = options.MemoryBudget ?? DeviceMemoryLimits.Apply(info.TotalMemory, _limits.DeviceMemory);
        Queue = ObjCRuntime.Send(properties.Handle, Selectors.NewCommandQueue);
        if (Queue == 0)
        {
            throw new InvalidOperationException($"Metal device {info.Name}: no command queue.");
        }
    }

    internal MetalDeviceProperties Properties { get; }

    /// <summary>The retained <c>MTLCommandQueue</c>; command queues are thread-safe, so segments of different models share it.</summary>
    internal nint Queue { get; }

    /// <summary>Lowering and MSL generation for this device.</summary>
    internal MetalCodeGenerator Code { get; }

    public DeviceInfo Info { get; }

    public DeviceOptions Options { get; }

    /// <summary>Detected features minus the ones disabled for this device.</summary>
    public IReadOnlySet<string> EnabledFeatures { get; }

    /// <summary>
    /// Code tier: MSL kernels (<c>kernel/*</c>), content addressed. Binary tier (unless <see cref="DeviceFeatures.KernelBinaries"/> is
    /// disabled): the device stage's binary archive (<c>metal/binary_archive</c>), matched on GPU name, OS build and kernel ABI.
    /// </summary>
    public IReadOnlyList<ArtifactTarget> ArtifactTargets { get; }

    public ArtifactStore? ArtifactStore { get; }

    public long AvailableMemory => GetMemoryInfo().Available;

    /// <inheritdoc />
    public bool SharesWeights => true;

    /// <summary>The submission budget in ms (0 = unlimited).</summary>
    public double SubmissionBudgetMs => Code.SubmissionBudgetMs;

    /// <summary>The working-set recommendation of the GPU as the total, the device's own allocations as the usage.</summary>
    public DeviceMemoryInfo GetMemoryInfo()
    {
        long ours = Interlocked.Read(ref _used);
        long allocated = (long)ObjCRuntime.SendUInt64(Properties.Handle, Selectors.CurrentAllocatedSize);
        return new DeviceMemoryInfo(DeviceMemoryLimits.Apply(Info.TotalMemory, _limits.DeviceMemory), _budget, Math.Max(ours, allocated), ours, false);
    }

    public OpSupport Supports(Node node) => GraphLowerer.Supports(node);

    public DeviceProfile CreateProfile()
    {
        var properties = new Dictionary<string, string>(Info.Capabilities)
        {
            ["registry_id"] = Properties.RegistryId.ToString("x", CultureInfo.InvariantCulture),
            ["os_version"] = Info.Driver,
        };
        return Code.CreateProfile(Info, properties);
    }

    /// <summary>
    /// State buffers per compiled graph (<see cref="CompileOptions.StateScope"/>): the first segment of the graph that uses a state allocates
    /// (and frees) its buffer, the later ones reuse it.
    /// </summary>
    internal System.Runtime.CompilerServices.ConditionalWeakTable<object, Dictionary<Models.StateDefinition, MetalBuffer>> SharedStates { get; } = new();

    internal long PipelineTicks;
    internal int PipelinesCreated;

    /// <summary>
    /// Device features whose kernels are compiled again without them when Metal rejects one of their pipelines (see
    /// <see cref="Compile"/>): the 8x8 matrix kernels and the integer-dot matvecs, which have portable SIMT fallbacks.
    /// </summary>
    private static readonly string[] s_fallbackFeatures = [DeviceFeatures.MatrixF16Small, DeviceFeatures.IntDot];

    /// <inheritdoc />
    /// <remarks>
    /// When Metal fails to create a pipeline (after one retry, see <see cref="CreatePipeline"/>) and the segment used one of the
    /// optional matrix or integer-dot features, the segment is lowered again without them, so the portable SIMT kernels replace the
    /// failing ones (counted in <see cref="PipelineFallbacks"/>) instead of failing the model.
    /// </remarks>
    public ICompiledSegment Compile(GraphSegment segment, CompileOptions options)
    {
        try
        {
            return CompileProgram(segment, options);
        }
        catch (KernelCompilationException e) when (s_fallbackFeatures.Any(f => options.FeaturesFor(this).Contains(f)))
        {
            // Always reported: a fallback means a slower kernel and a kernel Metal rejects (CI logs show which).
            int fallbacks = Interlocked.Increment(ref PipelineFallbacks);
            Console.Error.WriteLine($"[mtl] fallback {fallbacks}, segment {segment.Index}: {e.Message.Split('\n')[0]}; compiling it again without {string.Join(", ", s_fallbackFeatures)}");

            return CompileProgram(segment, options.WithDisabledFeatures(s_fallbackFeatures));
        }
    }

    /// <summary>Segments compiled again without the optional features after a pipeline failure (diagnostics, tests).</summary>
    internal int PipelineFallbacks;

    /// <summary>
    /// Test hook: kernels for which pipeline creation gets deliberately invalid MSL, to exercise the retry and the fallback of
    /// <see cref="Compile"/> on a real Metal compiler error.
    /// </summary>
    internal Func<CompiledKernel, bool>? InvalidSourceFor { get; set; }

    private ICompiledSegment CompileProgram(GraphSegment segment, CompileOptions options)
    {
        var cache = options.CacheFor?.Invoke(this) ?? _ownArtifacts.For(this);
        EnsureArchive(cache);
        var program = Code.Lower(segment, this, options, cache, out var kernels);
        options.CancellationToken.ThrowIfCancellationRequested();

        // Slang keeps compiling in the background while the weights are copied; pipelines are created as the kernels complete.
        var compiling = new LoadProgressCounter(options.Progress, LoadStage.Compiling, kernels.Count, Info.Name, kernels.Count(k => k.FromCache));
        var pipelines = Task.Run(() => CreatePipelines(kernels, compiling, options.CancellationToken));
        MetalSegment compiled;
        try
        {
            compiled = new MetalSegment(this, segment, program, options);
        }
        catch
        {
            WaitQuietly(pipelines);
            throw;
        }

        var waitTimer = Stopwatch.StartNew();
        try
        {
            pipelines.GetAwaiter().GetResult();
        }
        catch
        {
            compiled.Dispose();
            throw;
        }

        if (DiagnosticSwitches.LoadTiming)
        {
            Console.Error.WriteLine($"[load] kernels and pipelines segment {segment.Index}: waited {waitTimer.ElapsedMilliseconds} ms for {kernels.Count} kernels; slang {SlangCompiler.Statistics}; {PipelinesCreated} pipelines ({PipelinesFromArchive} from the binary archive), {(double)PipelineTicks / Stopwatch.Frequency * 1000:F0} ms summed");
        }

        return compiled;
    }

    /// <summary>Creates the pipelines of <paramref name="kernels"/> in parallel, waiting for kernels that are still compiling.</summary>
    /// <exception cref="KernelCompilationException">Slang rejected one of the kernels.</exception>
    private void CreatePipelines(IEnumerable<CompiledKernel> kernels, LoadProgressCounter progress, CancellationToken ct)
    {
        try
        {
            Parallel.ForEach(kernels, new ParallelOptions { CancellationToken = ct }, kernel =>
            {
                Pipeline(kernel);
                progress.Add();
            });
        }
        catch (AggregateException e)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw();
        }
    }

    private static void WaitQuietly(Task task)
    {
        try
        {
            task.GetAwaiter().GetResult();
        }
        catch (Exception)
        {
            // The caller is already failing; the pipeline error would only hide the original one.
        }
    }

    /// <summary>The pipeline of <paramref name="kernel"/>, created on first use (thread-safe).</summary>
    internal MetalPipeline Pipeline(CompiledKernel kernel)
        => _pipelines.GetOrAdd(kernel.CacheKey, _ => new Lazy<MetalPipeline>(() => CreatePipeline(kernel))).Value;

    private MetalPipeline CreatePipeline(CompiledKernel kernel)
    {
        long start = Stopwatch.GetTimestamp();
        var source = MetalKernelSource.Parse(kernel);
        if (DiagnosticSwitches.DumpSourceDirectory is { } dump)
        {
            Directory.CreateDirectory(dump);
            File.WriteAllText(Path.Combine(dump, $"{Path.GetFileNameWithoutExtension(kernel.Name)}_{kernel.EntryPoint}_{kernel.CacheKey[..12]}.metal"), source.Source);
        }

        if (InvalidSourceFor?.Invoke(kernel) == true)
        {
            source = source.WithSource("kernel void broken( {");
        }

        using var pool = AutoreleasePool.Push();
        // Metal's compiler runs in a service process; after it crashes on one kernel, the next requests of the process can fail
        // until it has restarted, so a failed kernel is tried once more before the error counts.
        KernelCompilationException? failure = null;
        for (int attempt = 0; attempt < PipelineAttempts; attempt++)
        {
            if (attempt > 0)
            {
                Thread.Sleep(PipelineRetryDelayMs);
            }

            try
            {
                return CreatePipelineOnce(kernel, source, start);
            }
            catch (KernelCompilationException e)
            {
                failure = e;
            }
        }

        throw failure!;
    }

    /// <summary>Attempts at creating one pipeline, and the pause before the second.</summary>
    private const int PipelineAttempts = 2;
    private const int PipelineRetryDelayMs = 250;

    private MetalPipeline CreatePipelineOnce(CompiledKernel kernel, MetalKernelSource source, long start)
    {
        var (library, function) = CompileFunction(source, $"{kernel.Name}:{kernel.EntryPoint} ({kernel.CacheKey})");
        try
        {
            nint state = _archive is { } archive ? FromArchive(archive, function) : 0;
            bool fromArchive = state != 0;
            if (!fromArchive)
            {
                nint error = 0;
                state = ObjCRuntime.Send(Properties.Handle, Selectors.NewComputePipelineStateWithFunction, function, &error);
                if (state == 0)
                {
                    throw new KernelCompilationException($"Metal could not create the pipeline of {kernel.Name}:{kernel.EntryPoint}: {Foundation.ErrorText(error, "no pipeline")}");
                }
            }

            ulong maxThreads = ObjCRuntime.SendUInt64(state, Selectors.MaxTotalThreadsPerThreadgroup);
            if (source.Threads > maxThreads)
            {
                Foundation.Release(state);
                throw new KernelCompilationException($"Kernel {kernel.Name}:{kernel.EntryPoint} needs {source.Threads} threads per threadgroup; the pipeline allows {maxThreads}.");
            }

            Interlocked.Add(ref PipelineTicks, Stopwatch.GetTimestamp() - start);
            Interlocked.Increment(ref PipelinesCreated);
            if (fromArchive)
            {
                Interlocked.Increment(ref PipelinesFromArchive);
            }

            if (DiagnosticSwitches.Trace)
            {
                Console.Error.WriteLine($"[mtl] pipeline {kernel.Name}:{kernel.EntryPoint} {kernel.CacheKey[..12]} threads {source.ThreadsX}x{source.ThreadsY}x{source.ThreadsZ} (max {maxThreads}){(fromArchive ? " from the binary archive" : "")} in {Stopwatch.GetElapsedTime(start).TotalMilliseconds:F1} ms");
            }

            return new MetalPipeline(state, new MTLSize(source.ThreadsX, source.ThreadsY, source.ThreadsZ));
        }
        finally
        {
            Foundation.Release(function);
            Foundation.Release(library);
        }
    }

    /// <summary>Compiles the MSL of a kernel (fast math off) and returns the retained library and kernel function.</summary>
    /// <exception cref="KernelCompilationException">Metal rejected the source.</exception>
    private (nint Library, nint Function) CompileFunction(MetalKernelSource source, string what)
    {
        using var pool = AutoreleasePool.Push();
        nint options = Foundation.New("MTLCompileOptions");
        try
        {
            ObjCRuntime.SendVoidBool(options, Selectors.SetFastMathEnabled, DiagnosticSwitches.FastMath ? (byte)1 : (byte)0);
            nint error = 0;
            nint library = ObjCRuntime.Send(Properties.Handle, Selectors.NewLibraryWithSource, Foundation.String(source.Source), options, &error);
            if (library == 0)
            {
                throw new KernelCompilationException($"Metal rejected kernel {what}: {Foundation.ErrorText(error, "no library")}");
            }

            nint function = ObjCRuntime.Send(library, Selectors.NewFunctionWithName, Foundation.String(source.Function));
            if (function == 0)
            {
                Foundation.Release(library);
                throw new KernelCompilationException($"Kernel {what}: no function {source.Function} in the compiled library.");
            }

            return (library, function);
        }
        finally
        {
            Foundation.Release(options);
        }
    }

    /// <summary>The pipeline of <paramref name="function"/> from the device stage's archive, or 0 when the archive does not have it.</summary>
    private nint FromArchive(MetalBinaryArchive archive, nint function)
    {
        using var pool = AutoreleasePool.Push();
        nint descriptor = Foundation.New("MTLComputePipelineDescriptor");
        try
        {
            ObjCRuntime.SendVoid(descriptor, Selectors.SetComputeFunction, function);
            ObjCRuntime.SendVoid(descriptor, Selectors.SetBinaryArchives, Foundation.ArrayOf(archive.Handle));
            nint error = 0;
            return ObjCRuntime.Send(Properties.Handle, Selectors.NewComputePipelineStateWithDescriptor, descriptor, MetalConstants.FailOnBinaryArchiveMiss, null, &error);
        }
        finally
        {
            Foundation.Release(descriptor);
        }
    }

    /// <summary>
    /// Loads the binary archive of the model-level cache (the device stage, <see cref="BinaryArchiveKey"/>) once per device, unless
    /// <see cref="DeviceFeatures.KernelBinaries"/> is disabled. A damaged or foreign archive is counted in
    /// <see cref="RejectedDeviceCode"/> and ignored (pipelines then compile from MSL).
    /// </summary>
    private void EnsureArchive(ICompilationCache? cache)
    {
        if (!_kernelBinaries || cache is null)
        {
            return;
        }

        lock (_archiveLock)
        {
            if (_archive is not null || !cache.TryGet(BinaryArchiveKey, out var data) || data.IsEmpty)
            {
                return;
            }

            _archive = MetalBinaryArchive.Load(Properties.Handle, data.Span);
            if (_archive is null)
            {
                Interlocked.Increment(ref RejectedDeviceCode);
            }
        }
    }

    /// <summary>
    /// The device stage: one binary archive with the pipeline of every kernel of <paramref name="portable"/> (MSL), compiled for this
    /// GPU and OS build (<see cref="MetalProvider.CompileDeviceSpecificAsync"/>).
    /// </summary>
    internal DeviceCodeSet CompileDeviceCode(PortableCodeSet portable, CancellationToken cancellationToken)
    {
        var kernels = portable.Entries.Where(e => e.Key.StartsWith(SlangCompiler.KernelKeyPrefix, StringComparison.Ordinal)).ToList();
        if (kernels.Count == 0 || !_kernelBinaries)
        {
            return DeviceCodeSet.Empty(Info.Provider, Info.Name);
        }

        using var archive = MetalBinaryArchive.CreateEmpty(Properties.Handle);
        foreach (var (key, code) in kernels)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var kernel = new CompiledKernel(key, "main", ShaderTarget.Metal, code.ToArray(), key[SlangCompiler.KernelKeyPrefix.Length..]);
            using var pool = AutoreleasePool.Push();
            var (library, function) = CompileFunction(MetalKernelSource.Parse(kernel), key);
            try
            {
                archive.Add(function);
            }
            finally
            {
                Foundation.Release(function);
                Foundation.Release(library);
            }
        }

        var target = ArtifactTargets.Single(t => t.Kind == BinaryArchiveKind);
        return new DeviceCodeSet(Info.Provider, Info.Name, target.Descriptor,
            new Dictionary<string, ReadOnlyMemory<byte>> { [BinaryArchiveKey] = archive.Serialize() });
    }

    /// <summary>The binary tier: the device stage's archive, valid for the GPU model and OS build it was compiled on.</summary>
    private ArtifactTarget CreateArchiveTarget() => new(
        new Dictionary<string, string>
        {
            [ArtifactDescriptorKeys.Provider] = Info.Provider,
            [ArtifactDescriptorKeys.Tier] = ArtifactTiers.Binary,
            [ArtifactDescriptorKeys.Kind] = BinaryArchiveKind,
            [ArtifactDescriptorKeys.DeviceName] = Info.Name,
            [ArtifactDescriptorKeys.Driver] = Info.Driver,
            [ArtifactDescriptorKeys.KernelAbi] = KernelAbiVersion,
        },
        [ArtifactDescriptorKeys.DeviceName, ArtifactDescriptorKeys.Driver, ArtifactDescriptorKeys.KernelAbi],
        [BinaryArchiveKey]);

    /// <summary>Allocates a zero-filled shared buffer of at least 16 bytes.</summary>
    /// <param name="bytes">The size.</param>
    /// <param name="purpose">What the buffer holds, for the error message.</param>
    /// <exception cref="DeviceOutOfMemoryException">
    /// Metal could not allocate the buffer, it exceeds the device's largest buffer (or <see cref="DeviceMemoryLimits.MaxAllocation"/>), or
    /// it would exceed <see cref="DeviceMemoryLimits.DeviceMemory"/>.
    /// </exception>
    internal MetalBuffer Allocate(long bytes, string? purpose = null)
    {
        long size = Math.Max(16, bytes);
        long largest = DeviceMemoryLimits.Apply(Properties.MaxBufferBytes > 0 ? Properties.MaxBufferBytes : long.MaxValue, _limits.MaxAllocation);
        string details = $"Device: {Info.TotalMemory / 1048576} MiB, Sharplight {Interlocked.Read(ref _used) / 1048576} MiB, largest buffer {largest / 1048576} MiB."
            + (_limits.IsEmpty ? "" : $" Limits: {_limits}.");
        if (size > largest)
        {
            throw new DeviceOutOfMemoryException(Info.Name, purpose ?? "a buffer", size, "The buffer exceeds the device's largest buffer. " + details);
        }

        if (_limits.DeviceMemory is { } cap && Interlocked.Read(ref _used) + size > cap)
        {
            throw new DeviceOutOfMemoryException(Info.Name, purpose ?? "a buffer", size, "The device memory limit is reached. " + details);
        }

        nint handle = ObjCRuntime.Send(Properties.Handle, Selectors.NewBufferWithLength, (nuint)size, MetalConstants.SharedUntracked);
        if (handle == 0)
        {
            throw new DeviceOutOfMemoryException(Info.Name, purpose ?? "a buffer", size, details);
        }

        Interlocked.Add(ref _used, size);
        return new MetalBuffer(handle, size);
    }

    /// <summary>
    /// A view of <paramref name="bytes"/> at <paramref name="pointer"/> inside the file mapping [<paramref name="mappingBase"/>,
    /// + <paramref name="mappingLength"/>): the whole mapping is wrapped once per device (one <c>newBufferWithBytesNoCopy</c>, its
    /// pages touched in parallel first so that wrapping does not fault them in one by one) and shared by every weight in it,
    /// reference counted. Returns null when Metal refuses the memory (the caller copies instead).
    /// </summary>
    internal MetalBuffer? TryWrapMapping(nint pointer, long bytes, nint mappingBase, long mappingLength)
    {
        lock (_mappings)
        {
            if (!_mappings.TryGetValue(mappingBase, out var entry))
            {
                long page = Environment.SystemPageSize;
                long start = mappingBase & ~(page - 1);
                long end = (mappingBase + mappingLength + page - 1) & ~(page - 1);
                Prefault((byte*)mappingBase, mappingLength, page);
                nint handle = ObjCRuntime.Send(Properties.Handle, Selectors.NewBufferWithBytesNoCopy, (void*)start, (nuint)(end - start), MetalConstants.SharedUntracked, 0);
                if (handle == 0)
                {
                    return null;
                }

                entry = new MappingEntry(handle, (nint)start);
                _mappings[mappingBase] = entry;
            }

            entry.References++;
            return new MetalBuffer(entry.Handle, bytes, pointer - entry.Start, wrapsHostMemory: true, mapping: mappingBase);
        }
    }

    /// <summary>Reads one byte per page of [<paramref name="data"/>, + <paramref name="length"/>) on several threads.</summary>
    private static void Prefault(byte* data, long length, long page)
    {
        const long PagesPerTask = 1024;
        nint start = (nint)data;
        long pages = (length + page - 1) / page;
        Parallel.For(0, (pages + PagesPerTask - 1) / PagesPerTask, task =>
        {
            long sum = 0;
            long last = Math.Min(pages, (task + 1) * PagesPerTask);
            for (long p = task * PagesPerTask; p < last; p++)
            {
                sum += ((byte*)start)[p * page];
            }

            GC.KeepAlive(sum);
        });
    }

    internal void Free(MetalBuffer buffer)
    {
        if (buffer.Handle == 0)
        {
            return;
        }

        if (buffer.Mapping != 0)
        {
            lock (_mappings)
            {
                var entry = _mappings[buffer.Mapping];
                if (--entry.References == 0)
                {
                    Foundation.Release(entry.Handle);
                    _mappings.Remove(buffer.Mapping);
                }
            }

            return;
        }

        if (!buffer.WrapsHostMemory)
        {
            Interlocked.Add(ref _used, -buffer.Bytes);
        }

        buffer.Release();
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        foreach (var pipeline in _pipelines.Values)
        {
            if (pipeline.IsValueCreated)
            {
                Foundation.Release(pipeline.Value.Handle);
            }
        }

        _pipelines.Clear();
        _archive?.Dispose();
        Foundation.Release(Queue);
    }
}
