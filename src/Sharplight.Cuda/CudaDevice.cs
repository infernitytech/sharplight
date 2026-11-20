using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using Sharplight.Cuda.Interop;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Shaders;
using Sharplight.Shaders.Lowering;

namespace Sharplight.Cuda;

/// <summary>An opened CUDA device: the GPU's primary context, its loaded kernels and device memory.</summary>
internal sealed unsafe class CudaDevice : IComputeDevice
{
    /// <summary>Prefix of the device stage's cache keys (<c>cubin/&lt;kernel content hash&gt;</c>).</summary>
    internal const string CubinKeyPrefix = "cubin/";

    /// <summary>Artifact kind of the device stage.</summary>
    internal const string CubinKind = "cubin";

    /// <summary>Version of the kernel argument layout, part of the device stage's identity.</summary>
    private const string KernelAbiVersion = "1";

    /// <summary>Bytes of the driver linker's error log.</summary>
    private const int LinkLogBytes = 16384;

    private readonly ConcurrentDictionary<string, Lazy<CudaKernel>> _kernels = new();
    /// <summary>Records compilations that come without a model-level cache (graphs built in code) into the artifact store.</summary>
    private readonly CompilationArtifacts _ownArtifacts = new();
    private readonly bool _kernelBinaries;
    private readonly long _budget;
    private readonly nint _context;
    private readonly Lazy<CudaBlas?> _blas;
    private long _used;
    private long _pinnedUsed;
    private bool _disposed;

    /// <summary>Kernels whose cubin came from the device stage or the artifact store (diagnostics, tests).</summary>
    internal int CubinsFromCache;

    /// <summary>Kernels linked from PTX by the driver in this process (diagnostics, tests).</summary>
    internal int CubinsLinked;

    /// <summary>Summed time of loading kernels (linking and module loads).</summary>
    internal long KernelLoadTicks;

    internal CudaDevice(CudaDeviceProperties properties, DeviceInfo info, DeviceOptions options)
    {
        Properties = properties;
        Info = info;
        Options = options;
        EnabledFeatures = DeviceFeatures.Effective(info.Features, options.DisabledFeatures);
        ArtifactStore = ArtifactStore.For(options);
        Code = new CudaCodeGenerator(properties.ComputeCapability, properties.KernelSharedMemory, EnabledFeatures,
            CudaCodeGenerator.EffectiveSubmissionBudget(info.SubmissionBudgetMs, options),
            CudaCodeGenerator.EffectiveConcurrentWorkgroups(info.ConcurrentWorkgroups, options), properties.Multiprocessors);
        _kernelBinaries = !DeviceFeatures.DisabledByEnvironment.Contains(DeviceFeatures.KernelBinaries) && !options.DisabledFeatures.Contains(DeviceFeatures.KernelBinaries);
        LaunchGraphs = !DeviceFeatures.DisabledByEnvironment.Contains(DeviceFeatures.LaunchGraphs) && !options.DisabledFeatures.Contains(DeviceFeatures.LaunchGraphs);
        ArtifactTargets = _kernelBinaries ? [Code.CreateCodeTarget(info.Provider), CreateCubinTarget()] : [Code.CreateCodeTarget(info.Provider)];
        Limits = DeviceMemoryLimits.For(options);
        _budget = options.MemoryBudget ?? DeviceMemoryLimits.Apply(info.TotalMemory, Limits.DeviceMemory);
        _blas = new Lazy<CudaBlas?>(() => CudaBlas.TryCreate(this));
        nint context;
        CudaApi.Check(CudaApi.cuDevicePrimaryCtxRetain(&context, properties.Ordinal), "cuDevicePrimaryCtxRetain");
        _context = context;
        try
        {
            Enter();
            nint stream;
            CudaApi.Check(CudaApi.cuStreamCreate(&stream, 0), "cuStreamCreate");
            Stream = stream;
        }
        catch
        {
            CudaApi.cuDevicePrimaryCtxRelease(properties.Ordinal);
            throw;
        }
    }

    internal CudaDeviceProperties Properties { get; }

    /// <summary>
    /// The caps of <see cref="DeviceOptions.MemoryLimits"/> and the environment that apply to this device: the device memory cap fails
    /// allocations beyond it, the pinned host memory cap bounds page-locked staging (0: pageable copies). CUDA has no BAR window to
    /// cap and no per-allocation limit of its own; <see cref="DeviceMemoryLimits.MaxAllocation"/> only bounds staging buffers here.
    /// </summary>
    internal DeviceMemoryLimits Limits { get; }

    /// <summary>The stream of uploads and buffer initialization (each segment executes on a stream of its own).</summary>
    internal nint Stream { get; }

    /// <summary>Lowering and PTX generation for this device.</summary>
    internal CudaCodeGenerator Code { get; }

    public DeviceInfo Info { get; }

    public DeviceOptions Options { get; }

    /// <summary>Detected features minus the ones disabled for this device.</summary>
    public IReadOnlySet<string> EnabledFeatures { get; }

    /// <summary>
    /// Code tier: PTX kernels (<c>kernel/*</c>), content addressed. Binary tier (unless <see cref="DeviceFeatures.KernelBinaries"/> is
    /// disabled): a cubin per kernel (<c>cubin/*</c>), matched on GPU name, CUDA driver version and kernel ABI.
    /// </summary>
    public IReadOnlyList<ArtifactTarget> ArtifactTargets { get; }

    public ArtifactStore? ArtifactStore { get; }

    public long AvailableMemory => GetMemoryInfo().Available;

    /// <inheritdoc />
    public bool SharesWeights => true;

    /// <inheritdoc />
    public bool StreamsWeights => true;

    /// <summary>The submission budget in ms (0 = unlimited).</summary>
    public double SubmissionBudgetMs => Code.SubmissionBudgetMs;

    /// <summary>True once the device was disposed (its context and memory are gone).</summary>
    internal bool IsDisposed => _disposed;

    /// <summary>Makes the device's context current on the calling thread (every driver call needs it).</summary>
    internal void Enter()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CudaApi.Check(CudaApi.cuCtxSetCurrent(_context), "cuCtxSetCurrent");
    }

    /// <summary>
    /// Free and total memory of the GPU as the driver reports it (every process), the device's own allocations as ours. A configured
    /// budget or device memory cap bounds what this device may hold, so what is still available is the smaller of the driver's free
    /// memory and the cap minus our own use.
    /// </summary>
    public DeviceMemoryInfo GetMemoryInfo()
    {
        Enter();
        nuint free;
        nuint total;
        CudaApi.Check(CudaApi.cuMemGetInfo(&free, &total), "cuMemGetInfo");
        long ours = Interlocked.Read(ref _used);
        long capacity = DeviceMemoryLimits.Apply((long)total, Limits.DeviceMemory);
        // Usage stays the device-wide figure (what a memory bar shows); the budget is set so that Available = Budget - Usage.
        long usage = Math.Max(ours, (long)total - (long)free);
        long available = Math.Max(0, Math.Min((long)free, Math.Min(_budget, capacity) - ours));
        return new DeviceMemoryInfo(capacity, usage + available, usage, ours, true);
    }

    public OpSupport Supports(Node node) => GraphLowerer.Supports(node);

    public DeviceProfile CreateProfile()
    {
        var properties = new Dictionary<string, string>(Info.Capabilities)
        {
            ["uuid"] = Info.Uuid ?? "",
        };
        return Code.CreateProfile(Info, properties);
    }

    /// <summary>
    /// State buffers per compiled graph (<see cref="CompileOptions.StateScope"/>): the first segment of the graph that uses a state allocates
    /// (and frees) its buffer, the later ones reuse it.
    /// </summary>
    internal System.Runtime.CompilerServices.ConditionalWeakTable<object, Dictionary<Models.StateDefinition, CudaBuffer>> SharedStates { get; } = new();

    private CudaWeightUploader? _streamUploader;
    /// <summary>Host ranges registered for streamed weights (start address to length and holders), and their total.</summary>
    private readonly Dictionary<nint, (long Bytes, int Holders)> _registered = [];
    private long _registeredBytes;

    /// <summary>
    /// Bytes of mapped weight files that streamed segments may page-lock for direct copies (<see cref="TryRegisterHost"/>): the pinned host
    /// memory limit when one is set, else a quarter of the memory available to the process (the pages cannot be reclaimed while locked).
    /// </summary>
    private long RegisterBudget => Limits.PinnedHostMemory ?? GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / 4;

    /// <summary>
    /// Page-locks <paramref name="bytes"/> of read-only host memory at <paramref name="start"/> (page aligned; a mapped weight file) so the
    /// device copies from it directly, within <see cref="RegisterBudget"/>; a range registered before gains a holder. False when the budget or
    /// the driver refuses (the caller then copies through staging memory).
    /// </summary>
    internal bool TryRegisterHost(byte* start, long bytes)
    {
        lock (_registered)
        {
            if (_registered.TryGetValue((nint)start, out var entry) && entry.Bytes == bytes)
            {
                _registered[(nint)start] = (bytes, entry.Holders + 1);
                return true;
            }

            if (_registeredBytes + bytes > RegisterBudget)
            {
                return false;
            }

            Enter();
            if (CudaApi.cuMemHostRegister(start, (nuint)bytes, CudaApi.MemHostRegisterReadOnly) != 0)
            {
                return false;
            }

            _registered[(nint)start] = (bytes, 1);
            _registeredBytes += bytes;
            return true;
        }
    }

    /// <summary>Gives up a hold from <see cref="TryRegisterHost"/>; the last one unregisters the range.</summary>
    internal void UnregisterHost(byte* start)
    {
        lock (_registered)
        {
            if (!_registered.TryGetValue((nint)start, out var entry))
            {
                return;
            }

            if (entry.Holders > 1)
            {
                _registered[(nint)start] = (entry.Bytes, entry.Holders - 1);
                return;
            }

            _registered.Remove((nint)start);
            _registeredBytes -= entry.Bytes;
            Enter();
            CudaApi.cuMemHostUnregister(start);
        }
    }

    /// <summary>Serializes the use of the streaming uploader (<see cref="StreamUploader"/>).</summary>
    internal Lock StreamLock { get; } = new();

    /// <summary>
    /// The uploader that segments streaming their weights (<see cref="CompileOptions.StreamWeights"/>) share, created on first use with
    /// full-size staging buffers; hold <see cref="StreamLock"/> while using it. The context must be current.
    /// </summary>
    internal CudaWeightUploader StreamUploader(long largestRow)
    {
        if (_streamUploader is { } uploader && uploader.StagingSize >= largestRow)
        {
            return uploader;
        }

        _streamUploader?.Dispose();
        return _streamUploader = new CudaWeightUploader(this, long.MaxValue, largestRow);
    }

    /// <summary>Per <see cref="CompileOptions.StateScope"/>: the activation buffers its segments on this device share, by buffer name.</summary>
    internal System.Runtime.CompilerServices.ConditionalWeakTable<object, Dictionary<string, SharedArena<CudaBuffer>>> SharedArenas { get; } = new();

    /// <inheritdoc />
    public ICompiledSegment Compile(GraphSegment segment, CompileOptions options)
    {
        var cache = options.CacheFor?.Invoke(this) ?? _ownArtifacts.For(this);
        var program = Code.Lower(segment, this, options, cache, out var kernels);
        options.CancellationToken.ThrowIfCancellationRequested();

        // Slang and NVRTC keep compiling in the background while the weights are uploaded; kernels are loaded as they complete.
        var compiling = new LoadProgressCounter(options.Progress, LoadStage.Compiling, kernels.Count, Info.Name, kernels.Count(k => k.FromCache));
        var loading = Task.Run(() => LoadKernels(kernels, cache, compiling, options.CancellationToken));
        CudaSegment compiled;
        try
        {
            compiled = new CudaSegment(this, segment, program, options);
        }
        catch
        {
            WaitQuietly(loading);
            throw;
        }

        var waitTimer = Stopwatch.StartNew();
        try
        {
            loading.GetAwaiter().GetResult();
        }
        catch
        {
            compiled.Dispose();
            throw;
        }

        if (DiagnosticSwitches.LoadTiming)
        {
            var (nvrtcKernels, nvrtcTime) = NvrtcTranslator.Statistics;
            Console.Error.WriteLine($"[load] kernels segment {segment.Index}: waited {waitTimer.ElapsedMilliseconds} ms for {kernels.Count} kernels; slang {SlangCompiler.Statistics}; nvrtc {nvrtcKernels} kernels {nvrtcTime.TotalMilliseconds:F0} ms; {CubinsLinked} linked, {CubinsFromCache} cubins cached, {(double)KernelLoadTicks / Stopwatch.Frequency * 1000:F0} ms summed");
        }

        return compiled;
    }

    /// <summary>Loads the modules of <paramref name="kernels"/> in parallel, waiting for kernels that are still compiling.</summary>
    /// <exception cref="KernelCompilationException">Slang, NVRTC or the driver rejected one of the kernels.</exception>
    private void LoadKernels(IEnumerable<CompiledKernel> kernels, ICompilationCache? cache, LoadProgressCounter progress, CancellationToken ct)
    {
        try
        {
            Parallel.ForEach(kernels, new ParallelOptions { CancellationToken = ct }, kernel =>
            {
                Kernel(kernel, cache);
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
            // The caller is already failing; the kernel error would only hide the original one.
        }
    }

    /// <summary>The loaded kernel of <paramref name="kernel"/>, loaded on first use (thread-safe).</summary>
    /// <param name="kernel">The compiled kernel (PTX).</param>
    /// <param name="cache">The model-level cache the cubin is looked up in and recorded into, or null.</param>
    internal CudaKernel Kernel(CompiledKernel kernel, ICompilationCache? cache = null)
        => _kernels.GetOrAdd(kernel.CacheKey, _ => new Lazy<CudaKernel>(() => LoadKernel(kernel, cache))).Value;

    /// <summary>
    /// True when segments replay repeated steps as CUDA graphs (<see cref="DeviceFeatures.LaunchGraphs"/>, on unless disabled with
    /// <c>SHARPLIGHT_DISABLE_FEATURES</c> or the device options).
    /// </summary>
    internal bool LaunchGraphs { get; }

    /// <summary>The cuBLASLt handle of the library GEMMs, created on first use (null when cuBLASLt is not installed).</summary>
    internal CudaBlas? Blas => _blas.Value;

    private CudaKernel LoadKernel(CompiledKernel kernel, ICompilationCache? cache)
    {
        if (LibraryKernel.TryParse(kernel.Code, out string operation, out var parameters))
        {
            return LoadLibraryKernel(operation, parameters);
        }

        var ptx = CudaPtx.Parse(kernel.Code, $"{kernel.Name}:{kernel.EntryPoint}");
        long start = Stopwatch.GetTimestamp();
        Enter();
        string cubinKey = CubinKeyPrefix + kernel.CacheKey;
        byte[]? cubin = null;
        if (_kernelBinaries && cache is not null && cache.TryGet(cubinKey, out var stored))
        {
            cubin = stored.ToArray();
            Interlocked.Increment(ref CubinsFromCache);
        }

        if (cubin is null && _kernelBinaries)
        {
            cubin = Link(ptx, $"{kernel.Name}:{kernel.EntryPoint}");
            Interlocked.Increment(ref CubinsLinked);
            cache?.Put(cubinKey, cubin);
        }

        nint module;
        fixed (byte* image = cubin ?? ptx.Code)
        {
            int result = CudaApi.cuModuleLoadData(&module, image);
            if (result != 0)
            {
                throw new KernelCompilationException($"The CUDA driver could not load kernel {kernel.Name}:{kernel.EntryPoint} ({kernel.CacheKey}): {CudaApi.ErrorText(result)}");
            }
        }

        nint function;
        fixed (byte* name = Encoding.UTF8.GetBytes(ptx.Function + "\0"))
        {
            CudaApi.Check(CudaApi.cuModuleGetFunction(&function, module, name), $"cuModuleGetFunction({ptx.Function})");
        }

        int maxThreads;
        CudaApi.Check(CudaApi.cuFuncGetAttribute(&maxThreads, CudaApi.FunctionMaxThreadsPerBlock, function), "cuFuncGetAttribute");
        if (ptx.ThreadsX * ptx.ThreadsY * ptx.ThreadsZ > maxThreads)
        {
            CudaApi.cuModuleUnload(module);
            throw new KernelCompilationException($"Kernel {kernel.Name}:{kernel.EntryPoint} needs {ptx.ThreadsX * ptx.ThreadsY * ptx.ThreadsZ} threads per block; its registers allow {maxThreads}.");
        }

        if (ptx.SharedBytes > 0)
        {
            int result = CudaApi.cuFuncSetAttribute(function, CudaApi.FunctionMaxDynamicSharedSizeBytes, ptx.SharedBytes);
            if (result != 0)
            {
                CudaApi.cuModuleUnload(module);
                throw new KernelCompilationException($"Kernel {kernel.Name}:{kernel.EntryPoint} needs {ptx.SharedBytes} bytes of shared memory: {CudaApi.ErrorText(result)}");
            }
        }

        Interlocked.Add(ref KernelLoadTicks, Stopwatch.GetTimestamp() - start);
        if (DiagnosticSwitches.Trace)
        {
            Console.Error.WriteLine($"[cuda] kernel {kernel.Name}:{kernel.EntryPoint} {kernel.CacheKey[..12]} block {ptx.ThreadsX}x{ptx.ThreadsY}x{ptx.ThreadsZ} (max {maxThreads}), dynamic shared {ptx.SharedBytes} B in {Stopwatch.GetElapsedTime(start).TotalMilliseconds:F1} ms");
        }

        return new CudaKernel(module, function, ptx.ThreadsX, ptx.ThreadsY, ptx.ThreadsZ, ptx.SharedBytes) { TensorMaps = ptx.TensorMaps };
    }

    /// <summary>A library kernel: the library GEMM on cuBLASLt (no module).</summary>
    /// <exception cref="KernelCompilationException">The operation is not a library GEMM of a supported type, or cuBLASLt is not installed.</exception>
    private CudaKernel LoadLibraryKernel(string operation, IReadOnlyDictionary<string, string> parameters)
    {
        if (operation != "gemm" || CudaBlas.DataType(parameters["a"]) is not { } type)
        {
            throw new KernelCompilationException($"Library kernel {operation} ({string.Join(' ', parameters.Select(p => $"{p.Key}={p.Value}"))}) is not supported by the CUDA provider.");
        }

        if (Blas is null)
        {
            throw new KernelCompilationException("The library GEMM needs cuBLASLt (package Sharplight.Cuda.Blas), which is not installed.");
        }

        return new CudaKernel(0, 0, 1, 1, 1, 0)
        {
            Library = new CudaLibraryGemm(type, long.Parse(parameters["n"], CultureInfo.InvariantCulture), long.Parse(parameters["k"], CultureInfo.InvariantCulture)),
        };
    }

    /// <summary>Compiles PTX into a cubin for this GPU with the driver's JIT linker. The context must be current.</summary>
    /// <exception cref="KernelCompilationException">The driver rejected the PTX (e.g. a PTX version newer than the driver).</exception>
    private static byte[] Link(CudaPtx ptx, string what)
    {
        var errorLog = new byte[LinkLogBytes];
        fixed (byte* log = errorLog)
        fixed (byte* code = ptx.Code)
        {
            var options = stackalloc CuJitOption[] { CuJitOption.ErrorLogBuffer, CuJitOption.ErrorLogBufferSizeBytes };
            var values = stackalloc void*[] { log, (void*)LinkLogBytes };
            nint state;
            CudaApi.Check(CudaApi.cuLinkCreate(2, options, values, &state), "cuLinkCreate");
            try
            {
                int result = CudaApi.cuLinkAddData(state, CudaApi.JitInputPtx, code, (nuint)ptx.Code.Length, null, 0, null, null);
                if (result == 0)
                {
                    void* cubin;
                    nuint size;
                    result = CudaApi.cuLinkComplete(state, &cubin, &size);
                    if (result == 0)
                    {
                        return new ReadOnlySpan<byte>(cubin, (int)size).ToArray();
                    }
                }

                string detail = Encoding.UTF8.GetString(errorLog).TrimEnd('\0');
                string hint = result == (int)CuResult.UnsupportedPtxVersion
                    ? " The PTX was generated by a newer NVRTC than this driver supports: update the NVIDIA driver (see docs/cuda.md)." : "";
                throw new KernelCompilationException($"The CUDA driver could not compile kernel {what}: {CudaApi.ErrorText(result)}.{hint}\n{detail}");
            }
            finally
            {
                CudaApi.cuLinkDestroy(state);
            }
        }
    }

    /// <summary>
    /// The device stage: a cubin for every kernel of <paramref name="portable"/> (PTX), compiled for this GPU by the driver
    /// (<see cref="CudaProvider.CompileDeviceSpecificAsync"/>).
    /// </summary>
    internal DeviceCodeSet CompileDeviceCode(PortableCodeSet portable, CancellationToken cancellationToken)
    {
        var kernels = portable.Entries.Where(e => e.Key.StartsWith(SlangCompiler.KernelKeyPrefix, StringComparison.Ordinal)).ToList();
        if (kernels.Count == 0 || !_kernelBinaries)
        {
            return DeviceCodeSet.Empty(Info.Provider, Info.Name);
        }

        Enter();
        var entries = new Dictionary<string, ReadOnlyMemory<byte>>();
        foreach (var (key, code) in kernels)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (LibraryKernel.TryParse(code.Span, out _, out _))
            {
                // Library kernels have no device code.
                continue;
            }

            var ptx = CudaPtx.Parse(code.ToArray(), key);
            entries[CubinKeyPrefix + key[SlangCompiler.KernelKeyPrefix.Length..]] = Link(ptx, key);
        }

        var target = ArtifactTargets.Single(t => t.Kind == CubinKind);
        return new DeviceCodeSet(Info.Provider, Info.Name, target.Descriptor, entries);
    }

    /// <summary>The binary tier: one cubin per kernel, valid for the GPU model and CUDA driver it was compiled with.</summary>
    private ArtifactTarget CreateCubinTarget() => new(
        new Dictionary<string, string>
        {
            [ArtifactDescriptorKeys.Provider] = Info.Provider,
            [ArtifactDescriptorKeys.Tier] = ArtifactTiers.Binary,
            [ArtifactDescriptorKeys.Kind] = CubinKind,
            [ArtifactDescriptorKeys.DeviceName] = Info.Name,
            [ArtifactDescriptorKeys.Driver] = Info.Driver,
            [ArtifactDescriptorKeys.KernelAbi] = KernelAbiVersion,
        },
        [ArtifactDescriptorKeys.DeviceName, ArtifactDescriptorKeys.Driver, ArtifactDescriptorKeys.KernelAbi],
        [CubinKeyPrefix]);

    /// <summary>Allocates a zero-filled device buffer of at least 16 bytes (zeroed on <see cref="Stream"/>).</summary>
    /// <param name="bytes">The size.</param>
    /// <param name="purpose">What the buffer holds, for the error message.</param>
    /// <exception cref="DeviceOutOfMemoryException">The driver could not allocate the buffer, or it would exceed <see cref="DeviceMemoryLimits.DeviceMemory"/>.</exception>
    internal CudaBuffer Allocate(long bytes, string? purpose = null)
    {
        long size = Math.Max(16, (bytes + 15) & ~15L);
        Enter();
        if (Limits.DeviceMemory is { } cap && Interlocked.Read(ref _used) + size > cap)
        {
            throw OutOfMemory(size, purpose, "The device memory limit is reached.");
        }

        ulong pointer;
        int result = CudaApi.cuMemAlloc(&pointer, (nuint)size);
        if (result != 0)
        {
            throw OutOfMemory(size, purpose, $"cuMemAlloc: {CudaApi.ErrorText(result)}.");
        }

        CudaApi.Check(CudaApi.cuMemsetD8Async(pointer, 0, (nuint)size, Stream), "cuMemsetD8Async");
        Interlocked.Add(ref _used, size);
        return new CudaBuffer(pointer, size);
    }

    private DeviceOutOfMemoryException OutOfMemory(long size, string? purpose, string reason)
    {
        nuint free = 0;
        nuint total = 0;
        CudaApi.cuMemGetInfo(&free, &total);
        var details = new StringBuilder(reason);
        details.Append(CultureInfo.InvariantCulture,
            $" Device: {total / 1048576} MiB, free {free / 1048576} MiB (all processes), Sharplight {Interlocked.Read(ref _used) / 1048576} MiB.");
        if (!Limits.IsEmpty)
        {
            details.Append(CultureInfo.InvariantCulture, $" Limits: {Limits}.");
        }

        return new DeviceOutOfMemoryException(Info.Name, purpose ?? "device memory", size, details.ToString());
    }

    /// <summary>
    /// Host memory for staging copies: page-locked (fast asynchronous copies) while <see cref="DeviceMemoryLimits.PinnedHostMemory"/> and
    /// the system allow it, else pageable memory (the driver then copies through its own staging; still correct, slower).
    /// </summary>
    /// <param name="bytes">The size.</param>
    /// <param name="pinned">Whether the memory is page-locked; pass it to <see cref="FreeHost"/>.</param>
    internal byte* AllocateHost(long bytes, out bool pinned)
    {
        byte* memory = TryAllocatePinned(bytes);
        pinned = memory != null;
        return pinned ? memory : (byte*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)bytes, 4096);
    }

    /// <summary>Page-locked host memory within <see cref="DeviceMemoryLimits.PinnedHostMemory"/>, or null when the cap or the system refuses it.</summary>
    internal byte* TryAllocatePinned(long bytes)
    {
        Enter();
        if (Limits.PinnedHostMemory is { } cap && Interlocked.Read(ref _pinnedUsed) + bytes > cap)
        {
            return null;
        }

        void* memory;
        if (CudaApi.cuMemAllocHost(&memory, (nuint)bytes) != 0)
        {
            return null;
        }

        Interlocked.Add(ref _pinnedUsed, bytes);
        return (byte*)memory;
    }

    /// <summary>Frees memory from <see cref="AllocateHost"/>.</summary>
    internal void FreeHost(byte* memory, long bytes, bool pinned)
    {
        if (memory == null)
        {
            return;
        }

        if (pinned)
        {
            CudaApi.cuMemFreeHost(memory);
            Interlocked.Add(ref _pinnedUsed, -bytes);
        }
        else
        {
            System.Runtime.InteropServices.NativeMemory.AlignedFree(memory);
        }
    }

    internal void Free(CudaBuffer buffer)
    {
        if (!buffer.Owned || buffer.Pointer == 0 || _disposed)
        {
            return;
        }

        Enter();
        CudaApi.cuMemFree(buffer.Pointer);
        Interlocked.Add(ref _used, -buffer.Bytes);
    }

    /// <summary>Waits for the uploads and initializations queued on <see cref="Stream"/>.</summary>
    internal void Synchronize()
    {
        Enter();
        CudaApi.Check(CudaApi.cuStreamSynchronize(Stream), "cuStreamSynchronize");
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        CudaApi.cuCtxSetCurrent(_context);
        if (_blas.IsValueCreated)
        {
            _blas.Value?.Dispose();
        }

        foreach (var kernel in _kernels.Values)
        {
            if (kernel.IsValueCreated && kernel.Value.Module != 0)
            {
                CudaApi.cuModuleUnload(kernel.Value.Module);
            }
        }

        _kernels.Clear();
        _streamUploader?.Dispose();
        _streamUploader = null;
        CudaApi.cuStreamDestroy(Stream);
        _disposed = true;
        CudaApi.cuDevicePrimaryCtxRelease(Properties.Ordinal);
    }
}
