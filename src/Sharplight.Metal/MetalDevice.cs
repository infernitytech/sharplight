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
    private readonly long _budget;
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
            MetalCodeGenerator.EffectiveSubmissionBudget(info.SubmissionBudgetMs, options),
            MetalCodeGenerator.EffectiveConcurrentWorkgroups(info.ConcurrentWorkgroups, options), 0);
        ArtifactTargets = [Code.CreateCodeTarget(info.Provider)];
        _budget = options.MemoryBudget ?? info.TotalMemory;
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

    /// <summary>Code tier only: MSL kernels (<c>kernel/*</c>), content addressed.</summary>
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
        return new DeviceMemoryInfo(Info.TotalMemory, _budget, Math.Max(ours, allocated), ours, false);
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

    public ICompiledSegment Compile(GraphSegment segment, CompileOptions options)
    {
        var cache = options.CacheFor?.Invoke(this) ?? _ownArtifacts.For(this);
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
            Console.Error.WriteLine($"[load] kernels and pipelines segment {segment.Index}: waited {waitTimer.ElapsedMilliseconds} ms for {kernels.Count} kernels; slang {SlangCompiler.Statistics}; {PipelinesCreated} pipelines, {(double)PipelineTicks / Stopwatch.Frequency * 1000:F0} ms summed");
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

        using var pool = AutoreleasePool.Push();
        nint options = ObjCRuntime.Send(ObjCRuntime.Send(ObjCRuntime.GetClass("MTLCompileOptions"), Selectors.Alloc), Selectors.Init);
        nint library = 0;
        nint function = 0;
        try
        {
            ObjCRuntime.SendVoidBool(options, Selectors.SetFastMathEnabled, DiagnosticSwitches.FastMath ? (byte)1 : (byte)0);
            nint error = 0;
            library = ObjCRuntime.Send(Properties.Handle, Selectors.NewLibraryWithSource, Foundation.String(source.Source), options, &error);
            if (library == 0)
            {
                throw new KernelCompilationException($"Metal rejected kernel {kernel.Name}:{kernel.EntryPoint} ({kernel.CacheKey}): {Foundation.ErrorText(error, "no library")}");
            }

            function = ObjCRuntime.Send(library, Selectors.NewFunctionWithName, Foundation.String(source.Function));
            if (function == 0)
            {
                throw new KernelCompilationException($"Kernel {kernel.Name}:{kernel.EntryPoint}: no function {source.Function} in the compiled library.");
            }

            error = 0;
            nint state = ObjCRuntime.Send(Properties.Handle, Selectors.NewComputePipelineStateWithFunction, function, &error);
            if (state == 0)
            {
                throw new KernelCompilationException($"Metal could not create the pipeline of {kernel.Name}:{kernel.EntryPoint}: {Foundation.ErrorText(error, "no pipeline")}");
            }

            ulong maxThreads = ObjCRuntime.SendUInt64(state, Selectors.MaxTotalThreadsPerThreadgroup);
            if (source.Threads > maxThreads)
            {
                Foundation.Release(state);
                throw new KernelCompilationException($"Kernel {kernel.Name}:{kernel.EntryPoint} needs {source.Threads} threads per threadgroup; the pipeline allows {maxThreads}.");
            }

            Interlocked.Add(ref PipelineTicks, Stopwatch.GetTimestamp() - start);
            Interlocked.Increment(ref PipelinesCreated);
            if (DiagnosticSwitches.Trace)
            {
                Console.Error.WriteLine($"[mtl] pipeline {kernel.Name}:{kernel.EntryPoint} {kernel.CacheKey[..12]} threads {source.ThreadsX}x{source.ThreadsY}x{source.ThreadsZ} (max {maxThreads}) in {Stopwatch.GetElapsedTime(start).TotalMilliseconds:F1} ms");
            }

            return new MetalPipeline(state, new MTLSize(source.ThreadsX, source.ThreadsY, source.ThreadsZ));
        }
        finally
        {
            Foundation.Release(function);
            Foundation.Release(library);
            Foundation.Release(options);
        }
    }

    /// <summary>Allocates a zero-filled shared buffer of at least 16 bytes.</summary>
    /// <exception cref="OutOfMemoryException">Metal could not allocate the buffer.</exception>
    internal MetalBuffer Allocate(long bytes)
    {
        long size = Math.Max(16, bytes);
        nint handle = ObjCRuntime.Send(Properties.Handle, Selectors.NewBufferWithLength, (nuint)size, MetalConstants.SharedUntracked);
        if (handle == 0)
        {
            throw new OutOfMemoryException($"Metal device {Info.Name}: cannot allocate a buffer of {size} bytes.");
        }

        Interlocked.Add(ref _used, size);
        return new MetalBuffer(handle, size);
    }

    /// <summary>
    /// Wraps <paramref name="bytes"/> of host memory at <paramref name="pointer"/> (a weight in the memory-mapped model file) in a
    /// buffer without copying: unified memory lets the GPU read the mapped pages directly, so loading maps the model instead of
    /// copying it. The pages around the data are wrapped (Metal needs page-aligned memory); the caller keeps the memory alive and
    /// unmodified while the buffer exists. Returns null when Metal refuses the memory (the caller copies instead).
    /// </summary>
    internal MetalBuffer? TryWrap(nint pointer, long bytes)
    {
        long page = Environment.SystemPageSize;
        long start = pointer & ~(page - 1);
        long end = (pointer + bytes + page - 1) & ~(page - 1);
        nint handle = ObjCRuntime.Send(Properties.Handle, Selectors.NewBufferWithBytesNoCopy, (void*)start, (nuint)(end - start), MetalConstants.SharedUntracked, 0);
        return handle == 0 ? null : new MetalBuffer(handle, bytes, pointer - start, wrapsHostMemory: true);
    }

    internal void Free(MetalBuffer buffer)
    {
        if (buffer.Handle == 0)
        {
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
        Foundation.Release(Queue);
    }
}
