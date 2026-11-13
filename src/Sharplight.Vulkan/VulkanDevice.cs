using System.Collections.Concurrent;
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
    internal readonly SlangCompiler Compiler;
    private readonly ConcurrentDictionary<string, Lazy<VkPipeline>> _pipelines = new();
    /// <summary>Every kernel of every compiled program (so an export can cover pipelines not created yet).</summary>
    private readonly ConcurrentDictionary<string, CompiledKernel> _kernels = new();
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
    public int SubgroupSize => (int)Phys.SubgroupSize;
    /// <summary>Effective submission budget in ms (0 = unlimited): <see cref="DeviceInfo.SubmissionBudgetMs"/> unless overridden.</summary>
    public double SubmissionBudgetMs { get; }

    internal VulkanDevice(VulkanInstance inst, PhysicalDeviceInfo phys, DeviceInfo info, DeviceOptions options)
    {
        Inst = inst;
        Phys = phys;
        Info = info;
        Options = options;
        EnabledFeatures = DeviceFeatures.Effective(info.Features, options.DisabledFeatures);
        SubmissionBudgetMs = info.SubmissionBudgetMs;
        options.Settings.TryGetValue("submission_budget_ms", out var budgetSetting);
        if (DiagnosticSwitches.SubmissionBudgetMs(budgetSetting) is { } budgetMs)
        {
            SubmissionBudgetMs = Math.Max(0, budgetMs);
        }

        var iapi = inst.Api;

        QueueFamily = SelectComputeQueueFamily(iapi, phys);

        // ---- features -----------------------------------------------------------------------------
        var f13 = new VkPhysicalDeviceVulkan13Features
        {
            subgroupSizeControl = Has(DeviceFeatures.SubgroupSizeControl),
            computeFullSubgroups = Has(DeviceFeatures.SubgroupSizeControl),
            maintenance4 = true,
        };
        var f12 = new VkPhysicalDeviceVulkan12Features
        {
            pNext = phys.ApiVersion >= VkVersion.Version_1_3 ? &f13 : null,
            bufferDeviceAddress = true,
            storageBuffer8BitAccess = true,
            uniformAndStorageBuffer8BitAccess = true,
            shaderInt8 = true,
            shaderFloat16 = Has(DeviceFeatures.Float16),
            vulkanMemoryModel = phys.VulkanMemoryModel,
            vulkanMemoryModelDeviceScope = phys.VulkanMemoryModel,
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

        var f2 = new VkPhysicalDeviceFeatures2 { pNext = &f11 };
        f2.features.shaderInt64 = true;
        f2.features.shaderInt16 = true;

        float prio = 1f;
        var qci = new VkDeviceQueueCreateInfo { queueFamilyIndex = QueueFamily, queueCount = 1, pQueuePriorities = &prio };
        var exts = new List<string>();
        _budgetExt = phys.MemoryBudgetExt && !DeviceFeatures.DisabledByEnvironment.Contains(DeviceFeatures.MemoryBudget) && !options.DisabledFeatures.Contains(DeviceFeatures.MemoryBudget);
        if (_budgetExt)
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

        var extNames = exts.Select(e => (nint)System.Runtime.InteropServices.Marshal.StringToHGlobalAnsi(e)).ToArray();
        var dci = new VkDeviceCreateInfo { pNext = &f2, queueCreateInfoCount = 1, pQueueCreateInfos = &qci };
        fixed (nint* en = extNames)
        {
            dci.enabledExtensionCount = (uint)extNames.Length;
            dci.ppEnabledExtensionNames = (byte**)en;
        }
        VkDevice dev;
        try
        {
            iapi.vkCreateDevice(phys.Handle, &dci, null, &dev).CheckResult();
        }
        finally
        {
            foreach (var name in extNames)
            {
                System.Runtime.InteropServices.Marshal.FreeHGlobal(name);
            }
        }

        Handle = dev;
        Api = new VkDeviceApi(iapi, dev);
        VkQueue queue;
        Api.vkGetDeviceQueue(QueueFamily, 0, &queue);
        Queue = queue;

        CommandPool = CreateCommandPool();

        PushConstantSize = Math.Min(MaxKernelPushConstantBytes, phys.MaxPushConstants);
        PipelineLayout = CreatePipelineLayout();

        Memory = new MemoryAllocator(this);
        _budget = options.MemoryBudget ?? (long)(phys.DeviceLocalBytes * 0.92);
        _kernelCacheDirectory = options.KernelCacheDirectory ?? SlangCompiler.DefaultCacheDirectory;
        Compiler = new SlangCompiler(_kernelCacheDirectory);
    }

    /// <summary>Prefers a dedicated compute queue (async compute), else any queue with compute.</summary>
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

    internal void Track(long bytes) => Interlocked.Add(ref _used, bytes);

    public OpSupport Supports(Node node) => GraphLowerer.Supports(node);

    internal long PipelineTicks;
    internal int PipelinesCreated;

    public ICompiledSegment Compile(GraphSegment segment, CompileOptions options)
    {
        var swLower = System.Diagnostics.Stopwatch.StartNew();
        var cache = options.CacheFor?.Invoke(this);
        EnsurePipelineCache(cache);
        var program = GraphLowerer.Lower(segment, Compiler, new LoweringOptions
        {
            Target = ShaderTarget.SpirV,
            SubgroupSize = (int)Phys.SubgroupSize,
            MaxSharedMemory = Phys.MaxSharedMemory,
            Features = options.FeaturesFor(this),
            Compile = options,
            SubmissionBudgetMs = SubmissionBudgetMs,
            Cache = cache,
        });
        foreach (var l in program.Launches.Concat(program.ResetLaunches))
        {
            foreach (var v in l.Variants)
            {
                _kernels.TryAdd(v.Kernel.CacheKey, v.Kernel);
            }
        }

        if (DiagnosticSwitches.LoadTiming)
        {
            Console.Error.WriteLine($"[load] lower segment {segment.Index}: {swLower.ElapsedMilliseconds} ms ({program.Launches.Count} launches, {program.Buffers.Count} buffers)");
        }

        // Slang keeps compiling in the background while the weights upload; pipelines are created as the kernels complete.
        var kernels = program.Launches.Concat(program.ResetLaunches).SelectMany(l => l.Variants).Select(v => v.Kernel).DistinctBy(k => k.CacheKey).ToList();
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

        var swWait = System.Diagnostics.Stopwatch.StartNew();
        pipelines.GetAwaiter().GetResult();
        SavePipelineCache();
        if (DiagnosticSwitches.LoadTiming)
        {
            Console.Error.WriteLine($"[load] kernels and pipelines segment {segment.Index}: waited {swWait.ElapsedMilliseconds} ms for {kernels.Count} kernels; slang {SlangCompiler.Statistics}; {PipelinesCreated} pipelines, {(double)PipelineTicks / System.Diagnostics.Stopwatch.Frequency * 1000:F0} ms summed");
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
            if (cache is not null && cache.TryGet(PipelineCacheKey, out var d))
            {
                data = d;
            }
            else if (TryReadPipelineCacheFile() is { } fromDisk)
            {
                data = fromDisk;
            }

            fixed (byte* p = data.Span)
            {
                var ci = new VkPipelineCacheCreateInfo { initialDataSize = (nuint)data.Length, pInitialData = data.Length > 0 ? p : null };
                VkPipelineCache pc;
                Api.vkCreatePipelineCache(&ci, null, &pc).CheckResult();
                _pipelineCache = pc;
            }
        }
    }

    /// <summary>Stores the driver pipeline cache, covering every kernel compiled on this device.</summary>
    public void ExportCompilationState(ICompilationCache cache)
    {
        EnsurePipelineCache(null);
        CreatePipelines(_kernels.Values);
        cache.Put(PipelineCacheKey, GetPipelineCacheData());
    }

    private byte[] GetPipelineCacheData()
    {
        nuint size = 0;
        Api.vkGetPipelineCacheData(_pipelineCache, &size, null).CheckResult();
        var data = new byte[size];
        fixed (byte* p = data)
        {
            Api.vkGetPipelineCacheData(_pipelineCache, &size, p).CheckResult();
        }

        return data.AsSpan(0, (int)size).ToArray();
    }

    /// <summary>The driver pipeline cache file of this device and driver in the kernel cache directory.</summary>
    private string PipelineCachePath
        => Path.Combine(_kernelCacheDirectory, $"pipelines-{Phys.VendorId:x}-{Phys.Uuid}-{Phys.DriverVersion:x}.bin");

    private byte[]? TryReadPipelineCacheFile()
    {
        try
        {
            return File.Exists(PipelineCachePath) ? File.ReadAllBytes(PipelineCachePath) : null;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return null;
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
            try
            {
                var temporary = $"{PipelineCachePath}.{Environment.ProcessId}.tmp";
                File.WriteAllBytes(temporary, GetPipelineCacheData());
                File.Move(temporary, PipelineCachePath, overwrite: true);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                // Read-only or full cache directory: the cache is only an accelerator.
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
        return _pipelines.GetOrAdd(kernel.CacheKey, _ => new Lazy<VkPipeline>(() => CreatePipeline(kernel))).Value;
    }

    private VkPipeline CreatePipeline(CompiledKernel kernel)
    {
        // Waits for a kernel that is still compiling; that time is not pipeline creation.
        var spirv = kernel.Code;
        long startTicks = System.Diagnostics.Stopwatch.GetTimestamp();
        try
        {
            VkShaderModule module;
            fixed (byte* code = spirv)
            {
                var moduleInfo = new VkShaderModuleCreateInfo { codeSize = (nuint)spirv.Length, pCode = (uint*)code };
                Api.vkCreateShaderModule(&moduleInfo, null, &module).CheckResult();
            }

            lock (_modules)
            {
                _modules.Add(module);
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
                var pipelineInfo = new VkComputePipelineCreateInfo { layout = PipelineLayout, stage = stage };
                VkPipeline pipeline;
                Api.vkCreateComputePipelines(_pipelineCache, 1, &pipelineInfo, null, &pipeline).CheckResult();
                return pipeline;
            }
        }
        finally
        {
            Interlocked.Add(ref PipelineTicks, System.Diagnostics.Stopwatch.GetTimestamp() - startTicks);
            Interlocked.Increment(ref PipelinesCreated);
        }
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
    internal void Submit(VkCommandBuffer cb, VkFence fence)
    {
        var si = new VkSubmitInfo { commandBufferCount = 1, pCommandBuffers = &cb };
        lock (_queueLock)
        {
            Api.vkQueueSubmit(Queue, 1, &si, fence).CheckResult();
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
