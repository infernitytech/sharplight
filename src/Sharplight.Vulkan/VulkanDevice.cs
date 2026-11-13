using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Shaders;
using Sharplight.Shaders.Lowering;
using Vortice.Vulkan;

namespace Sharplight.Vulkan;

/// <summary>An opened Vulkan device: logical device, compute queue, memory and pipeline caches.</summary>
internal sealed unsafe class VulkanDevice : IComputeDevice
{
    internal readonly VulkanInstance Inst;
    internal readonly PhysicalDeviceInfo Phys;
    internal readonly VkDevice Handle;
    internal readonly VkDeviceApi Api;
    internal readonly VkQueue Queue;
    internal readonly uint QueueFamily;
    internal readonly VkCommandPool CommandPool;
    internal readonly VkPipelineLayout PipelineLayout;
    internal readonly uint PushConstantSize;
    internal readonly MemoryAllocator Memory;
    /// <summary>Lowering and SPIR-V generation for this device's profile.</summary>
    internal readonly SpirvCodeGenerator Code;
    private readonly ConcurrentDictionary<string, Lazy<VkPipeline>> _pipelines = new();
    /// <summary>
    /// The kernels of the programs compiled with each model-level cache (<see cref="CompileOptions.CacheFor"/>), so an export
    /// stores exactly that model's pipelines, including ones not created yet.
    /// </summary>
    private readonly ConditionalWeakTable<ICompilationCache, ConcurrentDictionary<string, CompiledKernel>> _kernelsByCache = new();
    /// <summary>Whether the driver pipeline cache holds more than the pipelines of one model cache (seeded from the per-user cache file, or several caches compiled here).</summary>
    private bool _pipelineCacheShared;
    private readonly Lock _pipelineCacheLock = new();
    private readonly string _kernelCacheDirectory;
    private VkPipelineCache _pipelineCache;
    private int _pipelinesAtLastSave;
    private const string PipelineCacheKey = "vulkan/pipeline_cache";

    /// <summary>Largest push-constant block used for kernel arguments (the Vulkan minimum guaranteed is 128 bytes).</summary>
    private const uint MaxKernelPushConstantBytes = 256;
    private readonly List<VkShaderModule> _modules = [];
    // Thread safety: one VkDevice/queue is shared by every model (and thread) using this device. vkQueueSubmit needs
    // external synchronization (_queueLock via Submit); command pools are externally synchronized too, so every segment
    // records into its own pool and the device-level pool is only used under _poolLock.
    private readonly Lock _queueLock = new();
    private readonly Lock _poolLock = new();
    private readonly bool _budgetExt;
    private readonly long _budget;
    private long _used;
    private bool _disposed;

    public DeviceInfo Info { get; }
    public DeviceOptions Options { get; }
    /// <summary>Detected features minus the ones disabled for this device; only these are enabled in the driver.</summary>
    public IReadOnlySet<string> EnabledFeatures { get; }
    /// <summary>
    /// Code tier: SPIR-V kernels (<c>kernel/*</c>), content addressed, so any stored set for the same SPIR-V target may be
    /// consulted. Binary tier: the driver pipeline cache, only valid for the same vendor, device and driver.
    /// </summary>
    public IReadOnlyList<ArtifactTarget> ArtifactTargets { get; }
    private bool Has(string feature) => EnabledFeatures.Contains(feature);
    public long AvailableMemory => GetMemoryInfo().Available;

    /// <summary>Live memory figures: VK_EXT_memory_budget when the driver has it (accounts for other processes), else our own tally against the configured budget.</summary>
    public DeviceMemoryInfo GetMemoryInfo()
    {
        long ours = Interlocked.Read(ref _used);
        long total = Phys.DeviceLocalBytes;
        if (!_budgetExt)
        {
            return new DeviceMemoryInfo(total, _budget, ours, ours, false);
        }

        var bp = new VkPhysicalDeviceMemoryBudgetPropertiesEXT();
        var mp = new VkPhysicalDeviceMemoryProperties2 { pNext = &bp };
        Inst.Api.vkGetPhysicalDeviceMemoryProperties2(Phys.Handle, &mp);
        long heapBudget = (long)bp.heapBudget[Phys.DeviceLocalHeap];
        long heapUsage = (long)bp.heapUsage[Phys.DeviceLocalHeap];
        if (heapBudget <= 0)
        {
            return new DeviceMemoryInfo(total, _budget, ours, ours, false);
        }

        return new DeviceMemoryInfo(total, Math.Min(_budget, heapBudget), Math.Max(ours, heapUsage), ours, true);
    }
    /// <summary>Effective submission budget in ms (0 = unlimited): <see cref="DeviceInfo.SubmissionBudgetMs"/> unless overridden.</summary>
    public double SubmissionBudgetMs => Code.SubmissionBudgetMs;

    internal VulkanDevice(VulkanInstance inst, PhysicalDeviceInfo phys, DeviceInfo info, DeviceOptions options)
    {
        Inst = inst;
        Phys = phys;
        Info = info;
        Options = options;
        EnabledFeatures = DeviceFeatures.Effective(info.Features, options.DisabledFeatures);
        _kernelCacheDirectory = options.KernelCacheDirectory ?? SlangCompiler.DefaultCacheDirectory;
        Code = new SpirvCodeGenerator((int)phys.SubgroupSize, phys.MaxSharedMemory, EnabledFeatures,
            SpirvCodeGenerator.EffectiveSubmissionBudget(info.SubmissionBudgetMs, options), _kernelCacheDirectory);
        ArtifactTargets = CreateArtifactTargets();

        QueueFamily = SelectComputeQueueFamily(inst.Api, phys);
        var device = CreateLogicalDevice(inst.Api, out _budgetExt, out _pipelineStats);
        Handle = device;
        Api = new VkDeviceApi(inst.Api, device);
        VkQueue queue;
        Api.vkGetDeviceQueue(QueueFamily, 0, &queue);
        Queue = queue;

        CommandPool = CreateCommandPool();

        PushConstantSize = Math.Min(MaxKernelPushConstantBytes, phys.MaxPushConstants);
        PipelineLayout = CreatePipelineLayout();

        Memory = new MemoryAllocator(this);
        _budget = options.MemoryBudget ?? (long)(phys.DeviceLocalBytes * 0.92);
    }

    /// <summary>Prefers a dedicated compute queue (async compute), else any queue with compute.</summary>
    /// <summary>
    /// Creates the logical device with the enabled features and extensions chained into one feature structure.
    /// <paramref name="memoryBudgetExt"/> reports whether VK_EXT_memory_budget was enabled. <paramref name="pipelineStats"/> reports whether pipeline statistics capture was enabled (diagnostic switch).
    /// </summary>
    private VkDevice CreateLogicalDevice(VkInstanceApi instanceApi, out bool memoryBudgetExt, out bool pipelineStats)
    {
        var f13 = new VkPhysicalDeviceVulkan13Features
        {
            subgroupSizeControl = Has(DeviceFeatures.SubgroupSizeControl),
            computeFullSubgroups = Has(DeviceFeatures.SubgroupSizeControl),
            maintenance4 = true,
            shaderIntegerDotProduct = Has(DeviceFeatures.IntDot),
        };
        var f12 = new VkPhysicalDeviceVulkan12Features
        {
            pNext = Phys.ApiVersion >= VkVersion.Version_1_3 ? &f13 : null,
            bufferDeviceAddress = true,
            storageBuffer8BitAccess = true,
            uniformAndStorageBuffer8BitAccess = true,
            shaderInt8 = true,
            shaderFloat16 = Has(DeviceFeatures.Float16),
            vulkanMemoryModel = Phys.VulkanMemoryModel,
            vulkanMemoryModelDeviceScope = Phys.VulkanMemoryModel,
            scalarBlockLayout = true,
            timelineSemaphore = true,
        };
        var f11 = new VkPhysicalDeviceVulkan11Features { pNext = &f12, storageBuffer16BitAccess = true, uniformAndStorageBuffer16BitAccess = true };
        var coop = new VkPhysicalDeviceCooperativeMatrixFeaturesKHR { pNext = f13.pNext, cooperativeMatrix = true };
        bool coopMat = Has(DeviceFeatures.MatrixF16) || Has(DeviceFeatures.MatrixBF16) || Has(DeviceFeatures.MatrixInt8);
        bool coopMat2 = coopMat && Has(DeviceFeatures.MatrixWorkgroup);
        var coop2 = new VkPhysicalDeviceCooperativeMatrix2FeaturesNV
        {
            pNext = coop.pNext,
            cooperativeMatrixWorkgroupScope = true,
            cooperativeMatrixFlexibleDimensions = true,
            cooperativeMatrixTensorAddressing = true,
            cooperativeMatrixBlockLoads = true,
        };
        if (coopMat2)
        {
            coop.pNext = &coop2;
        }

        if (coopMat)
        {
            f13.pNext = &coop;
        }

        // bf16 arithmetic and bf16 cooperative matrices (VK_KHR_shader_bfloat16), e.g. the bf16x3 strict-precision conv.
        bool bfloat16 = Has(DeviceFeatures.BFloat16);
        var bf16 = new VkPhysicalDeviceShaderBfloat16FeaturesKHR
        {
            pNext = &f11,
            shaderBFloat16Type = true,
            shaderBFloat16CooperativeMatrix = Has(DeviceFeatures.MatrixBF16),
        };
        // Diagnostics: driver statistics of each pipeline (SHARPLIGHT_VK_PIPELINE_STATS).
        pipelineStats = DiagnosticSwitches.PipelineStats && Phys.Extensions.Contains("VK_KHR_pipeline_executable_properties");
        var executableProperties = new VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR { pNext = bfloat16 ? &bf16 : &f11, pipelineExecutableInfo = true };
        var f2 = new VkPhysicalDeviceFeatures2 { pNext = pipelineStats ? &executableProperties : bfloat16 ? &bf16 : &f11 };
        f2.features.shaderInt64 = true;
        f2.features.shaderInt16 = true;

        float prio = 1f;
        var qci = new VkDeviceQueueCreateInfo { queueFamilyIndex = QueueFamily, queueCount = 1, pQueuePriorities = &prio };
        var exts = new List<string>();
        memoryBudgetExt = Phys.MemoryBudgetExt && !DeviceFeatures.DisabledByEnvironment.Contains(DeviceFeatures.MemoryBudget) && !Options.DisabledFeatures.Contains(DeviceFeatures.MemoryBudget);
        if (memoryBudgetExt)
        {
            exts.Add("VK_EXT_memory_budget");
        }

        if (coopMat)
        {
            exts.Add("VK_KHR_cooperative_matrix");
        }

        if (coopMat2)
        {
            exts.Add("VK_NV_cooperative_matrix2");
        }

        if (bfloat16)
        {
            exts.Add("VK_KHR_shader_bfloat16");
        }

        if (pipelineStats)
        {
            exts.Add("VK_KHR_pipeline_executable_properties");
        }

        var extNames = exts.Select(e => (nint)Marshal.StringToHGlobalAnsi(e)).ToArray();
        var dci = new VkDeviceCreateInfo { pNext = &f2, queueCreateInfoCount = 1, pQueueCreateInfos = &qci };
        fixed (nint* en = extNames)
        {
            dci.enabledExtensionCount = (uint)extNames.Length;
            dci.ppEnabledExtensionNames = (byte**)en;
        }
        VkDevice device;
        try
        {
            instanceApi.vkCreateDevice(Phys.Handle, &dci, null, &device).CheckResult();
        }
        finally
        {
            foreach (var name in extNames)
            {
                Marshal.FreeHGlobal(name);
            }
        }

        return device;
    }

    private static uint SelectComputeQueueFamily(VkInstanceApi api, PhysicalDeviceInfo phys)
    {
        uint count = 0;
        api.vkGetPhysicalDeviceQueueFamilyProperties(phys.Handle, &count, null);
        var families = new VkQueueFamilyProperties[count];
        fixed (VkQueueFamilyProperties* p = families)
        {
            api.vkGetPhysicalDeviceQueueFamilyProperties(phys.Handle, &count, p);
        }

        int index = Array.FindIndex(families, q => (q.queueFlags & VkQueueFlags.Compute) != 0 && (q.queueFlags & VkQueueFlags.Graphics) == 0);
        if (index < 0)
        {
            index = Array.FindIndex(families, q => (q.queueFlags & VkQueueFlags.Compute) != 0);
        }

        if (index < 0)
        {
            throw new NotSupportedException("No compute queue.");
        }

        return (uint)index;
    }

    /// <summary>One push-constant range for the kernel arguments, shared by every pipeline.</summary>
    private VkPipelineLayout CreatePipelineLayout()
    {
        var range = new VkPushConstantRange { stageFlags = VkShaderStageFlags.Compute, offset = 0, size = PushConstantSize };
        var createInfo = new VkPipelineLayoutCreateInfo { pushConstantRangeCount = 1, pPushConstantRanges = &range };
        VkPipelineLayout layout;
        Api.vkCreatePipelineLayout(&createInfo, null, &layout).CheckResult();
        return layout;
    }

    private IReadOnlyList<ArtifactTarget> CreateArtifactTargets()
    {
        var code = Code.CreateCodeTarget(Info.Provider);
        var binary = new ArtifactTarget(
            new Dictionary<string, string>
            {
                [ArtifactDescriptorKeys.Provider] = Info.Provider,
                [ArtifactDescriptorKeys.Tier] = ArtifactTiers.Binary,
                [ArtifactDescriptorKeys.Kind] = "pipeline-cache",
                [ArtifactDescriptorKeys.Vendor] = Phys.VendorId.ToString("x", CultureInfo.InvariantCulture),
                [ArtifactDescriptorKeys.Device] = Phys.DeviceId.ToString("x", CultureInfo.InvariantCulture),
                [ArtifactDescriptorKeys.CacheId] = Phys.PipelineCacheUuid,
                [ArtifactDescriptorKeys.Driver] = Phys.DriverVersion.ToString("x", CultureInfo.InvariantCulture),
            },
            // Same model and driver cache id: valid on any card of that model, not only the one it was built on.
            [ArtifactDescriptorKeys.Vendor, ArtifactDescriptorKeys.Device, ArtifactDescriptorKeys.CacheId],
            []);
        return [code, binary];
    }

    internal void Track(long bytes) => Interlocked.Add(ref _used, bytes);

    public OpSupport Supports(Node node) => GraphLowerer.Supports(node);

    internal long PipelineTicks;
    internal int PipelinesCreated;

    public DeviceProfile CreateProfile()
    {
        var properties = new Dictionary<string, string>(Info.Capabilities)
        {
            ["vendor_id"] = Phys.VendorId.ToString("x4", CultureInfo.InvariantCulture),
            ["device_id"] = Phys.DeviceId.ToString("x4", CultureInfo.InvariantCulture),
            ["driver_version"] = Phys.DriverVersion.ToString("x", CultureInfo.InvariantCulture),
        };
        return Code.CreateProfile(Info, properties);
    }

    public ICompiledSegment Compile(GraphSegment segment, CompileOptions options)
    {
        var cache = options.CacheFor?.Invoke(this);
        EnsurePipelineCache(cache);
        var program = Code.Lower(segment, this, options, cache, out var kernels);
        RecordKernels(cache, kernels);

        // Slang keeps compiling in the background while the weights upload; pipelines are created as the kernels complete.
        var pipelines = Task.Run(() => CreatePipelines(kernels));
        VulkanSegment compiled;
        try
        {
            compiled = new VulkanSegment(this, segment, program, options);
        }
        catch
        {
            WaitQuietly(pipelines);
            throw;
        }

        var swWait = Stopwatch.StartNew();
        pipelines.GetAwaiter().GetResult();
        SavePipelineCache();
        if (DiagnosticSwitches.LoadTiming)
        {
            Console.Error.WriteLine($"[load] kernels and pipelines segment {segment.Index}: waited {swWait.ElapsedMilliseconds} ms for {kernels.Count} kernels; slang {SlangCompiler.Statistics}; {PipelinesCreated} pipelines, {(double)PipelineTicks / Stopwatch.Frequency * 1000:F0} ms summed");
        }

        return compiled;
    }

    /// <summary>Creates the pipelines of <paramref name="kernels"/> in parallel, waiting for kernels that are still compiling.</summary>
    /// <exception cref="KernelCompilationException">Slang rejected one of the kernels.</exception>
    private void CreatePipelines(IEnumerable<CompiledKernel> kernels)
    {
        try
        {
            Parallel.ForEach(kernels, kernel => Pipeline(kernel));
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
            // The caller is already failing with another exception.
        }
    }

    /// <summary>
    /// Creates the driver pipeline cache on first compile, seeded from a saved model's cache when present, else from the cache
    /// file of this device in the kernel cache directory (the driver validates the blob's header and ignores data from another
    /// device/driver).
    /// </summary>
    private void EnsurePipelineCache(ICompilationCache? cache)
    {
        lock (_pipelineCacheLock)
        {
            if (_pipelineCache != VkPipelineCache.Null)
            {
                return;
            }

            ReadOnlyMemory<byte> data = default;
            bool fromFile = false;
            if (cache is not null && cache.TryGet(PipelineCacheKey, out var d))
            {
                data = d;
            }
            else if (TryReadPipelineCacheFile() is { } fromDisk)
            {
                data = fromDisk;
                fromFile = true;
                _pipelineCacheShared = true;
            }

            if (TryCreatePipelineCache(data.Span, out var pc))
            {
                _pipelineCache = pc;
                return;
            }

            // The driver refused the seed blob (seen on NVIDIA with very large blobs as VK_ERROR_UNKNOWN): start empty, and drop
            // a refused cache file so later processes do not retry it.
            if (fromFile)
            {
                TryDeleteFile(PipelineCachePath);
            }

            if (!TryCreatePipelineCache(ReadOnlySpan<byte>.Empty, out pc))
            {
                throw new VkException(VkResult.ErrorInitializationFailed, "vkCreatePipelineCache failed without initial data");
            }

            _pipelineCache = pc;
        }
    }

    private bool TryCreatePipelineCache(ReadOnlySpan<byte> data, out VkPipelineCache pipelineCache)
    {
        fixed (byte* p = data)
        {
            var ci = new VkPipelineCacheCreateInfo { initialDataSize = (nuint)data.Length, pInitialData = data.Length > 0 ? p : null };
            VkPipelineCache pc;
            var result = Api.vkCreatePipelineCache(&ci, null, &pc);
            pipelineCache = result == VkResult.Success ? pc : VkPipelineCache.Null;
            return result == VkResult.Success;
        }
    }

    /// <summary>Remembers which kernels were compiled with <paramref name="cache"/> (see <see cref="ExportCompilationState"/>).</summary>
    private void RecordKernels(ICompilationCache? cache, IReadOnlyList<CompiledKernel> kernels)
    {
        lock (_pipelineCacheLock)
        {
            if (cache is null)
            {
                // Pipelines of a compilation without a model cache end up in the device's driver cache too.
                _pipelineCacheShared = true;
                return;
            }

            if (!_kernelsByCache.TryGetValue(cache, out var recorded))
            {
                _pipelineCacheShared |= _kernelsByCache.Any();
                recorded = new ConcurrentDictionary<string, CompiledKernel>();
                _kernelsByCache.Add(cache, recorded);
            }

            foreach (var kernel in kernels)
            {
                recorded.TryAdd(kernel.CacheKey, kernel);
            }
        }
    }

    /// <summary>
    /// Stores the driver pipeline cache of the kernels compiled with <paramref name="cache"/>. When the device's own driver cache
    /// also holds other pipelines (seeded from the per-user cache file, or other models compiled on this device), the pipelines
    /// are recreated into a fresh driver cache so the stored blob holds only this model's.
    /// </summary>
    public void ExportCompilationState(ICompilationCache cache)
    {
        EnsurePipelineCache(null);
        CompiledKernel[] kernels;
        bool shared;
        lock (_pipelineCacheLock)
        {
            kernels = _kernelsByCache.TryGetValue(cache, out var recorded) ? [.. recorded.Values] : [];
            shared = _pipelineCacheShared;
        }

        if (kernels.Length == 0)
        {
            return;
        }

        if (!shared)
        {
            CreatePipelines(kernels);
            cache.Put(PipelineCacheKey, GetPipelineCacheData(_pipelineCache));
            return;
        }

        if (!TryCreatePipelineCache(ReadOnlySpan<byte>.Empty, out var dedicated))
        {
            throw new VkException(VkResult.ErrorInitializationFailed, "vkCreatePipelineCache failed without initial data");
        }

        try
        {
            CreatePipelinesInto(dedicated, kernels);
            cache.Put(PipelineCacheKey, GetPipelineCacheData(dedicated));
        }
        finally
        {
            Api.vkDestroyPipelineCache(dedicated, null);
        }
    }

    /// <summary>Creates and immediately destroys the pipelines of <paramref name="kernels"/> in <paramref name="pipelineCache"/>, which records them.</summary>
    private void CreatePipelinesInto(VkPipelineCache pipelineCache, IEnumerable<CompiledKernel> kernels)
    {
        try
        {
            Parallel.ForEach(kernels, kernel =>
            {
                var (pipeline, module) = CreatePipeline(kernel, pipelineCache);
                Api.vkDestroyPipeline(pipeline, null);
                Api.vkDestroyShaderModule(module, null);
            });
        }
        catch (AggregateException e)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw();
        }
    }

    private byte[] GetPipelineCacheData(VkPipelineCache pipelineCache)
    {
        nuint size = 0;
        Api.vkGetPipelineCacheData(pipelineCache, &size, null).CheckResult();
        var data = new byte[size];
        fixed (byte* p = data)
        {
            Api.vkGetPipelineCacheData(pipelineCache, &size, p).CheckResult();
        }

        return data.AsSpan(0, (int)size).ToArray();
    }

    /// <summary>The driver pipeline cache file of this device and driver in the kernel cache directory.</summary>
    private string PipelineCachePath
        => Path.Combine(_kernelCacheDirectory, $"pipelines-{Phys.VendorId:x}-{Phys.Uuid}-{Phys.DriverVersion:x}.bin");

    /// <summary>
    /// Largest pipeline cache file that is loaded or written. The driver blob accumulates every pipeline ever created on the
    /// device (all models, all specializations); past this size it is reset so it is rebuilt from the pipelines in actual use.
    /// </summary>
    private const long MaxPipelineCacheFileBytes = 256L * 1024 * 1024;

    private byte[]? TryReadPipelineCacheFile()
    {
        try
        {
            DeleteStaleTemporaryFiles();
            var file = new FileInfo(PipelineCachePath);
            if (!file.Exists)
            {
                return null;
            }

            if (file.Length > MaxPipelineCacheFileBytes)
            {
                TryDeleteFile(file.FullName);
                return null;
            }

            return File.ReadAllBytes(file.FullName);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>Removes temporary cache files left behind by processes that died while saving.</summary>
    private void DeleteStaleTemporaryFiles()
    {
        var name = Path.GetFileName(PipelineCachePath);
        foreach (var temporary in Directory.EnumerateFiles(_kernelCacheDirectory, name + ".*.tmp"))
        {
            if (File.GetLastWriteTimeUtc(temporary) < DateTime.UtcNow - TimeSpan.FromHours(1))
            {
                TryDeleteFile(temporary);
            }
        }
    }

    private static void TryDeleteFile(string path)
    {
        try
        {
            File.Delete(path);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            // Another process may hold or have removed it; the cache is only an accelerator.
        }
    }

    /// <summary>
    /// Persists the driver pipeline cache next to the kernel cache when pipelines were created since it was last saved, so
    /// that the next process skips the driver's SPIR-V compilation. Failing to write it is not an error.
    /// </summary>
    private void SavePipelineCache()
    {
        lock (_pipelineCacheLock)
        {
            int created = Volatile.Read(ref PipelinesCreated);
            if (_pipelineCache == VkPipelineCache.Null || created == _pipelinesAtLastSave)
            {
                return;
            }

            _pipelinesAtLastSave = created;
            var temporary = $"{PipelineCachePath}.{Environment.ProcessId}.tmp";
            try
            {
                var data = GetPipelineCacheData(_pipelineCache);
                if (data.Length > MaxPipelineCacheFileBytes)
                {
                    // Too large to keep: reset, so the next process rebuilds the cache from the pipelines it uses.
                    TryDeleteFile(PipelineCachePath);
                    return;
                }

                File.WriteAllBytes(temporary, data);
                File.Move(temporary, PipelineCachePath, overwrite: true);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                // Read-only or full cache directory: the cache is only an accelerator.
                TryDeleteFile(temporary);
            }
        }
    }

    /// <summary>The compute pipeline of <paramref name="kernel"/>, created on first use; safe to call from several threads.</summary>
    internal VkPipeline Pipeline(CompiledKernel kernel)
    {
        if (_pipelines.TryGetValue(kernel.CacheKey, out var existing))
        {
            return existing.Value;
        }

        // Lazy: GetOrAdd may run its factory twice under a race (leaking a VkPipeline); pipelines of different kernels are
        // created in parallel (the driver's pipeline cache is internally synchronized).
        return _pipelines.GetOrAdd(kernel.CacheKey, _ => new Lazy<VkPipeline>(() => CreateDevicePipeline(kernel))).Value;
    }

    /// <summary>Creates a pipeline in the device's driver cache; its shader module lives as long as the device.</summary>
    private VkPipeline CreateDevicePipeline(CompiledKernel kernel)
    {
        var (pipeline, module) = CreatePipeline(kernel, _pipelineCache);
        lock (_modules)
        {
            _modules.Add(module);
        }

        return pipeline;
    }

    /// <summary>Creates the shader module and compute pipeline of <paramref name="kernel"/>, recorded in <paramref name="pipelineCache"/>.</summary>
    private (VkPipeline Pipeline, VkShaderModule Module) CreatePipeline(CompiledKernel kernel, VkPipelineCache pipelineCache)
    {
        // Waits for a kernel that is still compiling; that time is not pipeline creation.
        var spirv = kernel.Code;
        long startTicks = Stopwatch.GetTimestamp();
        try
        {
            VkShaderModule module;
            fixed (byte* code = spirv)
            {
                var moduleInfo = new VkShaderModuleCreateInfo { codeSize = (nuint)spirv.Length, pCode = (uint*)code };
                Api.vkCreateShaderModule(&moduleInfo, null, &module).CheckResult();
            }

            // Slang names the single SPIR-V entry point "main" regardless of the source function name.
            var entryName = "main\0"u8.ToArray();
            fixed (byte* entryPtr = entryName)
            {
                var subgroupSize = new VkPipelineShaderStageRequiredSubgroupSizeCreateInfo { requiredSubgroupSize = Phys.SubgroupSize };
                bool control = Has(DeviceFeatures.SubgroupSizeControl) && Phys.MinSubgroupSize != Phys.MaxSubgroupSize;
                var stage = new VkPipelineShaderStageCreateInfo
                {
                    pNext = control ? &subgroupSize : null,
                    stage = VkShaderStageFlags.Compute,
                    module = module,
                    pName = entryPtr,
                    flags = Has(DeviceFeatures.SubgroupSizeControl) ? VkPipelineShaderStageCreateFlags.RequireFullSubgroups : 0,
                };
                var pipelineInfo = new VkComputePipelineCreateInfo
                {
                    layout = PipelineLayout,
                    stage = stage,
                    flags = _pipelineStats ? VkPipelineCreateFlags.CaptureStatisticsKHR : 0,
                };
                VkPipeline pipeline;
                var result = Api.vkCreateComputePipelines(pipelineCache, 1, &pipelineInfo, null, &pipeline);
                if (result != VkResult.Success)
                {
                    Api.vkDestroyShaderModule(module, null);
                    result.CheckResult();
                }

                if (_pipelineStats)
                {
                    PrintPipelineStatistics(kernel, pipeline);
                }

                return (pipeline, module);
            }
        }
        finally
        {
            Interlocked.Add(ref PipelineTicks, Stopwatch.GetTimestamp() - startTicks);
            Interlocked.Increment(ref PipelinesCreated);
        }
    }

    private readonly bool _pipelineStats;

    /// <summary>Writes the driver's statistics of the first executable of <paramref name="pipeline"/> to stderr.</summary>
    private void PrintPipelineStatistics(CompiledKernel kernel, VkPipeline pipeline)
    {
        var info = new VkPipelineExecutableInfoKHR { pipeline = pipeline, executableIndex = 0 };
        uint count = 0;
        if (Api.vkGetPipelineExecutableStatisticsKHR(&info, &count, null) != VkResult.Success || count == 0)
        {
            return;
        }

        var stats = new VkPipelineExecutableStatisticKHR[count];
        for (int i = 0; i < stats.Length; i++)
        {
            stats[i].sType = VkStructureType.PipelineExecutableStatisticKHR;
        }

        var parts = new List<string>();
        fixed (VkPipelineExecutableStatisticKHR* p = stats)
        {
            Api.vkGetPipelineExecutableStatisticsKHR(&info, &count, p).CheckResult();
            for (int i = 0; i < count; i++)
            {
                var name = System.Runtime.InteropServices.Marshal.PtrToStringUTF8((nint)p[i].name) ?? "?";
                string value = p[i].format switch
                {
                    VkPipelineExecutableStatisticFormatKHR.Bool32 => ((bool)p[i].value.b32).ToString(),
                    VkPipelineExecutableStatisticFormatKHR.Int64 => p[i].value.i64.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    VkPipelineExecutableStatisticFormatKHR.Uint64 => p[i].value.u64.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    _ => p[i].value.f64.ToString(System.Globalization.CultureInfo.InvariantCulture),
                };
                parts.Add($"{name}={value}");
            }
        }

        Console.Error.WriteLine($"[pipeline] {kernel.Name} {kernel.EntryPoint}: {string.Join(", ", parts)}");
    }

    /// <summary>Submits a command buffer and waits for completion.</summary>
    internal void SubmitAndWait(VkCommandBuffer cb)
    {
        var fci = new VkFenceCreateInfo();
        VkFence fence;
        Api.vkCreateFence(&fci, null, &fence).CheckResult();
        try
        {
            var si = new VkSubmitInfo { commandBufferCount = 1, pCommandBuffers = &cb };
            lock (_queueLock)
            {
                Api.vkQueueSubmit(Queue, 1, &si, fence).CheckResult();
            }

            Api.vkWaitForFences(1, &fence, true, ulong.MaxValue).CheckResult();
        }
        finally
        {
            Api.vkDestroyFence(fence, null);
        }
    }

    internal VkCommandPool CreateCommandPool()
    {
        var cpci = new VkCommandPoolCreateInfo { queueFamilyIndex = QueueFamily, flags = VkCommandPoolCreateFlags.ResetCommandBuffer };
        VkCommandPool pool;
        Api.vkCreateCommandPool(&cpci, null, &pool).CheckResult();
        return pool;
    }

    /// <summary>Submits one command buffer; vkQueueSubmit needs external synchronization and the queue is shared by all users of the device.</summary>
    internal void Submit(VkCommandBuffer cb, VkFence fence) => Submit(new ReadOnlySpan<VkCommandBuffer>(in cb), fence);

    /// <summary>Submits command buffers that run in order as one batch, signalling <paramref name="fence"/> after the last.</summary>
    internal void Submit(ReadOnlySpan<VkCommandBuffer> cbs, VkFence fence)
    {
        fixed (VkCommandBuffer* p = cbs)
        {
            var si = new VkSubmitInfo { commandBufferCount = (uint)cbs.Length, pCommandBuffers = p };
            lock (_queueLock)
            {
                Api.vkQueueSubmit(Queue, 1, &si, fence).CheckResult();
            }
        }
    }

    /// <summary>Records and runs a one-off command buffer (device-level pool, serialized by a lock while recording).</summary>
    internal void Immediate(Action<VkCommandBuffer> record)
    {
        VkCommandBuffer cb;
        lock (_poolLock)
        {
            var ai = new VkCommandBufferAllocateInfo { commandPool = CommandPool, level = VkCommandBufferLevel.Primary, commandBufferCount = 1 };
            Api.vkAllocateCommandBuffers(&ai, &cb).CheckResult();
            var bi = new VkCommandBufferBeginInfo { flags = VkCommandBufferUsageFlags.OneTimeSubmit };
            Api.vkBeginCommandBuffer(cb, &bi).CheckResult();
            record(cb);
            Api.vkEndCommandBuffer(cb).CheckResult();
        }
        try
        {
            SubmitAndWait(cb);
        }
        finally
        {
            lock (_poolLock)
            {
                Api.vkFreeCommandBuffers(CommandPool, 1, &cb);
            }
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        Api.vkDeviceWaitIdle();
        foreach (var p in _pipelines.Values.Where(l => l.IsValueCreated).Select(l => l.Value))
        {
            Api.vkDestroyPipeline(p, null);
        }

        foreach (var m in _modules)
        {
            Api.vkDestroyShaderModule(m, null);
        }

        if (_pipelineCache != VkPipelineCache.Null)
        {
            Api.vkDestroyPipelineCache(_pipelineCache, null);
        }

        Memory.Dispose();
        Api.vkDestroyPipelineLayout(PipelineLayout, null);
        Api.vkDestroyCommandPool(CommandPool, null);
        Api.vkDestroyDevice(null);
    }
}
