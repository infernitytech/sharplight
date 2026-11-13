using System.Collections.Concurrent;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Shaders;
using Sharplight.Shaders.Lowering;
using Vortice.Vulkan;

namespace Sharplight.Vulkan;

/// <summary>An opened Vulkan device: logical device, compute queue, memory and pipeline caches.</summary>
public sealed unsafe class VulkanDevice : IComputeDevice
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
    readonly ConcurrentDictionary<string, VkPipeline> _pipelines = new();
    /// <summary>Every kernel of every compiled program (so an export can cover pipelines not created yet).</summary>
    readonly ConcurrentDictionary<string, CompiledKernel> _kernels = new();
    readonly Lock _pipelineCacheLock = new();
    VkPipelineCache _pipelineCache;
    const string PipelineCacheKey = "vulkan/pipeline_cache";
    readonly List<VkShaderModule> _modules = [];
    readonly Lock _queueLock = new();
    readonly long _budget;
    long _used;
    bool _disposed;

    public DeviceInfo Info { get; }
    public DeviceOptions Options { get; }
    /// <summary>Detected features minus the ones disabled for this device; only these are enabled in the driver.</summary>
    public IReadOnlySet<string> EnabledFeatures { get; }
    bool Has(string feature) => EnabledFeatures.Contains(feature);
    public long AvailableMemory => _budget - Interlocked.Read(ref _used);
    public int SubgroupSize => (int)Phys.SubgroupSize;

    internal VulkanDevice(VulkanInstance inst, PhysicalDeviceInfo phys, DeviceInfo info, DeviceOptions options)
    {
        Inst = inst;
        Phys = phys;
        Info = info;
        Options = options;
        EnabledFeatures = DeviceFeatures.Effective(info.Features, options.DisabledFeatures);
        var iapi = inst.Api;

        // ---- queue ------------------------------------------------------------------------------
        uint qn = 0;
        iapi.vkGetPhysicalDeviceQueueFamilyProperties(phys.Handle, &qn, null);
        var qfs = new VkQueueFamilyProperties[qn];
        fixed (VkQueueFamilyProperties* q = qfs) iapi.vkGetPhysicalDeviceQueueFamilyProperties(phys.Handle, &qn, q);
        // Prefer a dedicated compute queue (async compute), else any queue with compute.
        int qf = Array.FindIndex(qfs, q => (q.queueFlags & VkQueueFlags.Compute) != 0 && (q.queueFlags & VkQueueFlags.Graphics) == 0);
        if (qf < 0) qf = Array.FindIndex(qfs, q => (q.queueFlags & VkQueueFlags.Compute) != 0);
        if (qf < 0) throw new NotSupportedException("No compute queue.");
        QueueFamily = (uint)qf;

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
        if (coopMat) f13.pNext = &coop;
        var f2 = new VkPhysicalDeviceFeatures2 { pNext = &f11 };
        f2.features.shaderInt64 = true;
        f2.features.shaderInt16 = true;

        float prio = 1f;
        var qci = new VkDeviceQueueCreateInfo { queueFamilyIndex = QueueFamily, queueCount = 1, pQueuePriorities = &prio };
        var exts = new List<string>();
        if (coopMat) exts.Add("VK_KHR_cooperative_matrix");
        var extNames = exts.Select(e => (nint)System.Runtime.InteropServices.Marshal.StringToHGlobalAnsi(e)).ToArray();
        var dci = new VkDeviceCreateInfo { pNext = &f2, queueCreateInfoCount = 1, pQueueCreateInfos = &qci };
        fixed (nint* en = extNames) { dci.enabledExtensionCount = (uint)extNames.Length; dci.ppEnabledExtensionNames = (byte**)en; }
        VkDevice dev;
        try { iapi.vkCreateDevice(phys.Handle, &dci, null, &dev).CheckResult(); }
        finally { foreach (var e in extNames) System.Runtime.InteropServices.Marshal.FreeHGlobal(e); }
        Handle = dev;
        Api = new VkDeviceApi(iapi, dev);
        VkQueue queue;
        Api.vkGetDeviceQueue(QueueFamily, 0, &queue);
        Queue = queue;

        var cpci = new VkCommandPoolCreateInfo { queueFamilyIndex = QueueFamily, flags = VkCommandPoolCreateFlags.ResetCommandBuffer };
        VkCommandPool pool;
        Api.vkCreateCommandPool(&cpci, null, &pool).CheckResult();
        CommandPool = pool;

        PushConstantSize = Math.Min(256u, phys.MaxPushConstants);
        var pcr = new VkPushConstantRange { stageFlags = VkShaderStageFlags.Compute, offset = 0, size = PushConstantSize };
        var plci = new VkPipelineLayoutCreateInfo { pushConstantRangeCount = 1, pPushConstantRanges = &pcr };
        VkPipelineLayout layout;
        Api.vkCreatePipelineLayout(&plci, null, &layout).CheckResult();
        PipelineLayout = layout;

        Memory = new MemoryAllocator(this);
        _budget = options.MemoryBudget ?? (long)(phys.DeviceLocalBytes * 0.92);
        Compiler = new SlangCompiler(options.KernelCacheDirectory ?? SlangCompiler.DefaultCacheDirectory);
    }

    internal void Track(long bytes) => Interlocked.Add(ref _used, bytes);

    public OpSupport Supports(Node node) => GraphLowerer.Supports(node);

    /// <summary>Load-phase timing to stderr (SHARPLIGHT_LOAD_TIMING=1).</summary>
    internal static readonly bool s_loadTiming = Environment.GetEnvironmentVariable("SHARPLIGHT_LOAD_TIMING") == "1";
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
            Features = options.FeaturesFor(this),
            Compile = options,
            Cache = cache,
        });
        foreach (var l in program.Launches.Concat(program.ResetLaunches))
            foreach (var v in l.Variants) _kernels.TryAdd(v.Kernel.CacheKey, v.Kernel);
        if (s_loadTiming) Console.Error.WriteLine($"[load] lower segment {segment.Index}: {swLower.ElapsedMilliseconds} ms ({program.Launches.Count} launches, {program.Buffers.Count} buffers)");
        return new VulkanSegment(this, segment, program, options);
    }

    /// <summary>Creates the driver pipeline cache on first compile, seeded from a saved model's cache when present
    /// (the driver validates the blob's header and ignores data from another device/driver).</summary>
    void EnsurePipelineCache(ICompilationCache? cache)
    {
        lock (_pipelineCacheLock)
        {
            if (_pipelineCache != VkPipelineCache.Null) return;
            ReadOnlyMemory<byte> data = default;
            if (cache is not null && cache.TryGet(PipelineCacheKey, out var d)) data = d;
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
        foreach (var k in _kernels.Values) Pipeline(k);
        nuint size = 0;
        Api.vkGetPipelineCacheData(_pipelineCache, &size, null).CheckResult();
        var data = new byte[size];
        fixed (byte* p = data) Api.vkGetPipelineCacheData(_pipelineCache, &size, p).CheckResult();
        cache.Put(PipelineCacheKey, data.AsMemory(0, (int)size));
    }

    internal VkPipeline Pipeline(CompiledKernel kernel)
    {
        return _pipelines.GetOrAdd(kernel.CacheKey, _ =>
        {
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            try {
            VkShaderModule module;
            fixed (byte* code = kernel.Code)
            {
                var smci = new VkShaderModuleCreateInfo { codeSize = (nuint)kernel.Code.Length, pCode = (uint*)code };
                Api.vkCreateShaderModule(&smci, null, &module).CheckResult();
            }
            lock (_modules) _modules.Add(module);
            // Slang names the single SPIR-V entry point "main" regardless of the source function name.
            var entryBytes = "main\0"u8.ToArray();
            fixed (byte* entryPtr = entryBytes) {
            var sub = new VkPipelineShaderStageRequiredSubgroupSizeCreateInfo { requiredSubgroupSize = Phys.SubgroupSize };
            bool control = Has(DeviceFeatures.SubgroupSizeControl) && Phys.MinSubgroupSize != Phys.MaxSubgroupSize;
            var stage = new VkPipelineShaderStageCreateInfo
            {
                pNext = control ? &sub : null,
                stage = VkShaderStageFlags.Compute,
                module = module,
                pName = entryPtr,
                flags = Has(DeviceFeatures.SubgroupSizeControl) ? VkPipelineShaderStageCreateFlags.RequireFullSubgroups : 0,
            };
            var ci = new VkComputePipelineCreateInfo { layout = PipelineLayout, stage = stage };
            VkPipeline pipe;
            Api.vkCreateComputePipelines(_pipelineCache, 1, &ci, null, &pipe).CheckResult();
            return pipe;
            }
            } finally { Interlocked.Add(ref PipelineTicks, System.Diagnostics.Stopwatch.GetTimestamp() - t0); Interlocked.Increment(ref PipelinesCreated); }
        });
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
            lock (_queueLock) Api.vkQueueSubmit(Queue, 1, &si, fence).CheckResult();
            Api.vkWaitForFences(1, &fence, true, ulong.MaxValue).CheckResult();
        }
        finally
        {
            Api.vkDestroyFence(fence, null);
        }
    }

    internal VkCommandBuffer AllocateCommandBuffer()
    {
        var ai = new VkCommandBufferAllocateInfo { commandPool = CommandPool, level = VkCommandBufferLevel.Primary, commandBufferCount = 1 };
        VkCommandBuffer cb;
        Api.vkAllocateCommandBuffers(&ai, &cb).CheckResult();
        return cb;
    }

    internal void FreeCommandBuffer(VkCommandBuffer cb) => Api.vkFreeCommandBuffers(CommandPool, 1, &cb);

    /// <summary>Records and runs a one-off command buffer.</summary>
    internal void Immediate(Action<VkCommandBuffer> record)
    {
        var cb = AllocateCommandBuffer();
        try
        {
            var bi = new VkCommandBufferBeginInfo { flags = VkCommandBufferUsageFlags.OneTimeSubmit };
            Api.vkBeginCommandBuffer(cb, &bi).CheckResult();
            record(cb);
            Api.vkEndCommandBuffer(cb).CheckResult();
            SubmitAndWait(cb);
        }
        finally
        {
            FreeCommandBuffer(cb);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Api.vkDeviceWaitIdle();
        foreach (var p in _pipelines.Values) Api.vkDestroyPipeline(p, null);
        foreach (var m in _modules) Api.vkDestroyShaderModule(m, null);
        if (_pipelineCache != VkPipelineCache.Null) Api.vkDestroyPipelineCache(_pipelineCache, null);
        Memory.Dispose();
        Api.vkDestroyPipelineLayout(PipelineLayout, null);
        Api.vkDestroyCommandPool(CommandPool, null);
        Api.vkDestroyDevice(null);
    }
}
