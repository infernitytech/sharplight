using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
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
    /// <summary>
    /// Queue family of the transfer-only queue that downloads large outputs (the copy engines), or -1 when the device has none or
    /// <c>SHARPLIGHT_VK_TRANSFER_QUEUE=0</c>. On the NVIDIA driver measured, vkCmdCopyBuffer from device memory into host memory on
    /// the compute queue moved about 4.3 GB/s, on the transfer queue about 26 GB/s (segformer-b0 at batch 8 returns 79 MB:
    /// 18 ms against 3 ms).
    /// </summary>
    internal readonly int TransferFamily;
    /// <summary>The transfer queue of <see cref="TransferFamily"/> (null handle when there is none).</summary>
    internal readonly VkQueue TransferQueue;
    /// <summary>The queue families buffers are shared by (compute and transfer), or empty when only the compute queue is used.</summary>
    internal readonly uint[] SharingFamilies;
    private readonly ConcurrentDictionary<string, Lazy<VkPipeline>> _pipelines = new();
    private readonly Lock _pipelineCacheLock = new();
    /// <summary>Records compilations that come without a model-level cache (graphs built in code) into the artifact store.</summary>
    private readonly CompilationArtifacts _ownArtifacts = new();
    private VkPipelineCache _pipelineCache;
    private int _pipelinesAtLastSave;
    /// <summary>Model-level cache key of the driver pipeline cache blob (device code without kernel binaries).</summary>
    internal const string PipelineCacheKey = "vulkan/pipeline_cache";
    /// <summary>Prefix of the model-level cache keys of kernel binaries (<c>vulkan/pipeline_binary/&lt;kernel key&gt;</c>).</summary>
    internal const string PipelineBinaryPrefix = "vulkan/pipeline_binary/";
    /// <summary>Artifact kind of device code stored as one driver pipeline cache blob.</summary>
    internal const string PipelineCacheKind = "pipeline-cache";
    /// <summary>Artifact kind of device code stored as kernel binaries (VK_KHR_pipeline_binary).</summary>
    internal const string PipelineBinaryKind = "pipeline-binary";
    /// <summary>
    /// Version of how kernels receive their arguments (one pipeline layout: a push-constant block of buffer addresses and scalars, no
    /// descriptor sets). Raise it when that changes, so device code built for the old layout is never consulted.
    /// </summary>
    private const int KernelAbiVersion = 1;
    /// <summary>Kernel binaries (VK_KHR_pipeline_binary) are enabled: device code is stored and loaded per kernel.</summary>
    private readonly bool _kernelBinaries;
    /// <summary>Creates the pipelines of kernels (from SPIR-V, from stored binaries, or capturing binaries).</summary>
    internal readonly PipelineBuilder Pipelines;
    /// <summary>Pipelines created from stored kernel binaries (no driver compilation).</summary>
    internal int PipelinesFromBinaries;
    /// <summary>Stored device code (kernel binaries or a pipeline cache blob) that was present but rejected as not matching or damaged.</summary>
    internal int RejectedDeviceCode;

    /// <summary>Largest push-constant block used for kernel arguments (the Vulkan minimum guaranteed is 128 bytes).</summary>
    private const uint MaxKernelPushConstantBytes = 256;
    private readonly List<VkShaderModule> _modules = [];
    // Thread safety: one VkDevice/queue is shared by every model (and thread) using this device. vkQueueSubmit needs
    // external synchronization (_queueLock via Submit); command pools are externally synchronized too, so every segment
    // records into its own pool and the device-level pool is only used under _poolLock.
    private readonly Lock _queueLock = new();
    private readonly Lock _transferLock = new();
    private readonly Lock _poolLock = new();
    private readonly bool _budgetExt;
    private readonly long _budget;
    /// <summary>The budget the options set, or null (the driver's budget, else 92 % of the device-local heap).</summary>
    private readonly long? _configuredBudget;
    /// <summary>The caps of <see cref="DeviceOptions.MemoryLimits"/> and the environment that apply to this device.</summary>
    internal readonly DeviceMemoryLimits Limits;
    /// <summary>
    /// The compute queue writes timestamps (timestampValidBits &gt; 0). Without them <see cref="VulkanSegment.LastGpuMs"/> stays 0 and
    /// launch costs are measured on the host.
    /// </summary>
    internal readonly bool Timestamps;
    /// <summary>
    /// Buffers can be bound to several allocations (sparseBinding on the compute queue): buffers larger than the largest allocation are
    /// then split over several (<see cref="MemoryAllocator"/>).
    /// </summary>
    internal readonly bool SparseBinding;
    private long _used;
    private bool _disposed;

    public DeviceInfo Info { get; }
    public DeviceOptions Options { get; }
    /// <summary>Detected features minus the ones disabled for this device; only these are enabled in the driver.</summary>
    public IReadOnlySet<string> EnabledFeatures { get; }
    /// <summary>
    /// Code tier: SPIR-V kernels (<c>kernel/*</c>), content addressed, so any stored set for the same SPIR-V target may be
    /// consulted. Binary tier: kernel binaries when <see cref="DeviceFeatures.KernelBinaries"/> is enabled (matched on device model,
    /// driver binary key and kernel ABI), and the driver pipeline cache (matched on device model and driver cache id).
    /// </summary>
    public IReadOnlyList<ArtifactTarget> ArtifactTargets { get; }

    /// <summary>The persistent artifact store (per user unless the options say otherwise), or null when disabled.</summary>
    public ArtifactStore? ArtifactStore { get; }
    private bool Has(string feature) => EnabledFeatures.Contains(feature);
    public long AvailableMemory => GetMemoryInfo().Available;
    /// <inheritdoc />
    public bool SharesWeights => true;

    /// <inheritdoc />
    public bool StreamsWeights => true;

    /// <summary>
    /// Live memory figures of the largest device-local heap: VK_EXT_memory_budget when the driver has it (accounts for other
    /// processes), else our own tally against the configured budget. <see cref="DeviceMemoryLimits.DeviceMemory"/> caps the total.
    /// </summary>
    public DeviceMemoryInfo GetMemoryInfo()
    {
        long ours = Interlocked.Read(ref _used);
        long total = DeviceMemoryLimits.Apply(Phys.DeviceLocalBytes, Limits.DeviceMemory);
        long budget = Math.Min(_budget, total);
        if (!_budgetExt)
        {
            return new DeviceMemoryInfo(total, budget, ours, ours, false);
        }

        var heaps = Memory.ReadHeapBudgets();
        long heapBudget = (long)heaps.Budget[Phys.DeviceLocalHeap];
        long heapUsage = (long)heaps.Usage[Phys.DeviceLocalHeap];
        if (heapBudget <= 0)
        {
            return new DeviceMemoryInfo(total, budget, ours, ours, false);
        }

        // The driver's budget already leaves what the system and other processes need: the 92 % estimate only stands in where it is
        // missing (Qwen3.5-122B on a 32 GB RTX PRO 4500: 2.6 GB more for weights, four layers of routed experts).
        return new DeviceMemoryInfo(total, Math.Min(_configuredBudget ?? total, heapBudget), Math.Max(ours, heapUsage), ours, true);
    }

    /// <summary>Imports of mapped host memory (VK_EXT_external_memory_host), or null when the device has no import.</summary>
    internal HostMemoryImports? HostImports { get; }

    /// <inheritdoc />
    /// <remarks>Where the device imports host memory: the selected experts are copied from the imported pages of the mapped file.</remarks>
    public bool StreamsSelectedExperts => HostImports is not null;

    /// <summary>VK_EXT_memory_budget is enabled: <see cref="MemoryAllocator.ReadHeapBudgets"/> reports live per-heap figures.</summary>
    internal bool MemoryBudgetEnabled => _budgetExt;

    /// <summary>Effective submission budget in ms (0 = unlimited): <see cref="DeviceInfo.SubmissionBudgetMs"/> unless overridden.</summary>
    public double SubmissionBudgetMs => Code.SubmissionBudgetMs;

    internal VulkanDevice(VulkanInstance inst, PhysicalDeviceInfo phys, DeviceInfo info, DeviceOptions options)
    {
        Inst = inst;
        Phys = phys;
        Info = info;
        Options = options;
        EnabledFeatures = DeviceFeatures.Effective(info.Features, options.DisabledFeatures);
        ArtifactStore = ArtifactStore.For(options);
        Code = new SpirvCodeGenerator((int)phys.SubgroupSize, phys.MaxSharedMemory, EnabledFeatures,
            SpirvCodeGenerator.EffectiveSubmissionBudget(info.SubmissionBudgetMs, options),
            SpirvCodeGenerator.EffectiveConcurrentWorkgroups(info.ConcurrentWorkgroups, options),
            SpirvCodeGenerator.ComputeUnitsOf(info.Capabilities));
        _kernelBinaries = phys.PipelineBinaries && !phys.PrefersInternalPipelineCache
            && !DeviceFeatures.DisabledByEnvironment.Contains(DeviceFeatures.KernelBinaries) && !options.DisabledFeatures.Contains(DeviceFeatures.KernelBinaries);

        Limits = DeviceMemoryLimits.For(options);
        QueueFamily = SelectComputeQueueFamily(inst.Api, phys);
        var computeFamily = QueueFamilyProperties(inst.Api, phys)[QueueFamily];
        Timestamps = computeFamily.timestampValidBits > 0;
        SparseBinding = phys.SparseBinding && (computeFamily.queueFlags & VkQueueFlags.SparseBinding) != 0;
        TransferFamily = DiagnosticSwitches.TransferQueue ? SelectTransferQueueFamily(inst.Api, phys) : -1;
        SharingFamilies = TransferFamily >= 0 ? [QueueFamily, (uint)TransferFamily] : [];
        var device = CreateLogicalDevice(inst.Api, out _budgetExt, out _pipelineStats, out bool hostImport);
        Handle = device;
        Api = new VkDeviceApi(inst.Api, device);
        VkQueue queue;
        Api.vkGetDeviceQueue(QueueFamily, 0, &queue);
        Queue = queue;
        if (TransferFamily >= 0)
        {
            VkQueue transfer;
            Api.vkGetDeviceQueue((uint)TransferFamily, 0, &transfer);
            TransferQueue = transfer;
        }

        CommandPool = CreateCommandPool();

        PushConstantSize = Math.Min(MaxKernelPushConstantBytes, phys.MaxPushConstants);
        PipelineLayout = CreatePipelineLayout();
        Pipelines = new PipelineBuilder(Api, PipelineLayout, Phys.SubgroupSize,
            Has(DeviceFeatures.SubgroupSizeControl) && Phys.MinSubgroupSize != Phys.MaxSubgroupSize, Has(DeviceFeatures.SubgroupSizeControl),
            _pipelineStats ? VkPipelineCreateFlags.CaptureStatisticsKHR : 0,
            (Phys.PortabilitySubset && DiagnosticSwitches.PortabilityRewrite) || DiagnosticSwitches.PortabilityRewriteAll);
        ArtifactTargets = CreateArtifactTargets();

        Memory = new MemoryAllocator(this);
        HostImports = hostImport ? HostMemoryImports.TryCreate(this) : null;
        _configuredBudget = options.MemoryBudget;
        _budget = options.MemoryBudget ?? (long)(DeviceMemoryLimits.Apply(phys.DeviceLocalBytes, Limits.DeviceMemory) * 0.92);
    }

    /// <summary>Prefers a dedicated compute queue (async compute), else any queue with compute.</summary>
    /// <summary>
    /// Creates the logical device with the enabled features and extensions chained into one feature structure.
    /// <paramref name="memoryBudgetExt"/> reports whether VK_EXT_memory_budget was enabled, <paramref name="hostImport"/> whether
    /// VK_EXT_external_memory_host was. <paramref name="pipelineStats"/> reports whether pipeline statistics capture was enabled (diagnostic switch).
    /// </summary>
    private VkDevice CreateLogicalDevice(VkInstanceApi instanceApi, out bool memoryBudgetExt, out bool pipelineStats, out bool hostImport)
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
            uniformAndStorageBuffer8BitAccess = Phys.UniformAndStorageBuffer8BitAccess,
            shaderInt8 = true,
            shaderFloat16 = Has(DeviceFeatures.Float16),
            vulkanMemoryModel = Phys.VulkanMemoryModel,
            vulkanMemoryModelDeviceScope = Phys.VulkanMemoryModel,
            scalarBlockLayout = true,
        };
        var f11 = new VkPhysicalDeviceVulkan11Features { pNext = &f12, storageBuffer16BitAccess = true, uniformAndStorageBuffer16BitAccess = Phys.UniformAndStorageBuffer16BitAccess };
        var coop = new VkPhysicalDeviceCooperativeMatrixFeaturesKHR { pNext = f13.pNext, cooperativeMatrix = true };
        bool coopMat = Has(DeviceFeatures.MatrixF16) || Has(DeviceFeatures.MatrixBF16) || Has(DeviceFeatures.MatrixInt8) || Has(DeviceFeatures.MatrixFp8);
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
        // 8-bit floats and their cooperative matrices (VK_EXT_shader_float8), e.g. the fp8 tensor-core GEMM.
        bool float8 = Has(DeviceFeatures.Float8) || Has(DeviceFeatures.MatrixFp8);
        var f8 = new VkPhysicalDeviceShaderFloat8FeaturesEXT
        {
            pNext = &f11,
            shaderFloat8 = true,
            shaderFloat8CooperativeMatrix = Has(DeviceFeatures.MatrixFp8),
        };
        void* tail = float8 ? &f8 : &f11;
        bool bfloat16 = Has(DeviceFeatures.BFloat16);
        var bf16 = new VkPhysicalDeviceShaderBfloat16FeaturesKHR
        {
            pNext = tail,
            shaderBFloat16Type = true,
            shaderBFloat16CooperativeMatrix = Has(DeviceFeatures.MatrixBF16),
        };
        // Diagnostics: driver statistics of each pipeline (SHARPLIGHT_VK_PIPELINE_STATS).
        pipelineStats = DiagnosticSwitches.PipelineStats && Phys.Extensions.Contains("VK_KHR_pipeline_executable_properties");
        var executableProperties = new VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR { pNext = bfloat16 ? &bf16 : tail, pipelineExecutableInfo = true };
        void* chain = pipelineStats ? &executableProperties : bfloat16 ? &bf16 : tail;
        // Kernel binaries (VK_KHR_pipeline_binary, which needs maintenance5) for device code stored per kernel.
        var pipelineBinary = new VkPhysicalDevicePipelineBinaryFeaturesKHR { pNext = chain, pipelineBinaries = true };
        var maintenance5 = new VkPhysicalDeviceMaintenance5Features { pNext = &pipelineBinary, maintenance5 = true };
        var f2 = new VkPhysicalDeviceFeatures2 { pNext = _kernelBinaries ? &maintenance5 : chain };
        f2.features.shaderInt64 = true;
        f2.features.sparseBinding = SparseBinding;
        f2.features.shaderInt16 = true;

        float prio = 1f;
        var queueInfos = stackalloc VkDeviceQueueCreateInfo[2];
        queueInfos[0] = new VkDeviceQueueCreateInfo { queueFamilyIndex = QueueFamily, queueCount = 1, pQueuePriorities = &prio };
        uint queueInfoCount = 1;
        if (TransferFamily >= 0)
        {
            queueInfos[queueInfoCount++] = new VkDeviceQueueCreateInfo { queueFamilyIndex = (uint)TransferFamily, queueCount = 1, pQueuePriorities = &prio };
        }
        var exts = new List<string>();
        memoryBudgetExt = Phys.MemoryBudgetExt && !DeviceFeatures.DisabledByEnvironment.Contains(DeviceFeatures.MemoryBudget) && !Options.DisabledFeatures.Contains(DeviceFeatures.MemoryBudget);
        if (memoryBudgetExt)
        {
            exts.Add("VK_EXT_memory_budget");
        }

        hostImport = Phys.HostPointerAlignment > 0 && Phys.ApiVersion >= VkVersion.Version_1_1 && !DeviceFeatures.DisabledByEnvironment.Contains(DeviceFeatures.HostMemoryImport) && !Options.DisabledFeatures.Contains(DeviceFeatures.HostMemoryImport);
        if (hostImport)
        {
            exts.Add("VK_EXT_external_memory_host");
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

        if (float8)
        {
            exts.Add("VK_EXT_shader_float8");
        }

        if (pipelineStats)
        {
            exts.Add("VK_KHR_pipeline_executable_properties");
        }

        if (_kernelBinaries)
        {
            exts.Add("VK_KHR_pipeline_binary");
            if (Phys.Extensions.Contains("VK_KHR_maintenance5"))
            {
                exts.Add("VK_KHR_maintenance5");
            }
        }

        // A portability implementation (MoltenVK) requires its subset extension to be enabled. None of the subset's optional
        // features are requested: the compute kernels use no events, image views, samplers or graphics state.
        if (Phys.PortabilitySubset)
        {
            exts.Add("VK_KHR_portability_subset");
        }

        var extNames = exts.Select(e => (nint)Marshal.StringToHGlobalAnsi(e)).ToArray();
        VkDevice device;
        try
        {
            // The name array must stay pinned until vkCreateDevice returns: a pointer taken in a fixed block that ends before the
            // call dangles once the GC compacts the heap (parallel test collections allocate heavily while devices are created),
            // and the loader then crashes validating the names.
            fixed (nint* en = extNames)
            {
                var dci = new VkDeviceCreateInfo
                {
                    pNext = &f2, queueCreateInfoCount = queueInfoCount, pQueueCreateInfos = queueInfos,
                    enabledExtensionCount = (uint)extNames.Length, ppEnabledExtensionNames = (byte**)en,
                };
                instanceApi.vkCreateDevice(Phys.Handle, &dci, null, &device).CheckResult();
            }
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

    /// <summary>A queue family with transfer but neither compute nor graphics (the copy engines), or -1.</summary>
    private static int SelectTransferQueueFamily(VkInstanceApi api, PhysicalDeviceInfo phys)
    {
        var families = QueueFamilyProperties(api, phys);
        return Array.FindIndex(families, q => (q.queueFlags & VkQueueFlags.Transfer) != 0 && (q.queueFlags & (VkQueueFlags.Compute | VkQueueFlags.Graphics)) == 0);
    }

    private static VkQueueFamilyProperties[] QueueFamilyProperties(VkInstanceApi api, PhysicalDeviceInfo phys)
    {
        uint count = 0;
        api.vkGetPhysicalDeviceQueueFamilyProperties(phys.Handle, &count, null);
        var families = new VkQueueFamilyProperties[count];
        fixed (VkQueueFamilyProperties* p = families)
        {
            api.vkGetPhysicalDeviceQueueFamilyProperties(phys.Handle, &count, p);
        }

        return families;
    }

    private static uint SelectComputeQueueFamily(VkInstanceApi api, PhysicalDeviceInfo phys)
    {
        var families = QueueFamilyProperties(api, phys);
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
        var abi = $"{KernelAbiVersion}.{PushConstantSize}";
        var identity = new Dictionary<string, string>
        {
            [ArtifactDescriptorKeys.Provider] = Info.Provider,
            [ArtifactDescriptorKeys.Tier] = ArtifactTiers.Binary,
            [ArtifactDescriptorKeys.Vendor] = Phys.VendorId.ToString("x", CultureInfo.InvariantCulture),
            [ArtifactDescriptorKeys.Device] = Phys.DeviceId.ToString("x", CultureInfo.InvariantCulture),
            [ArtifactDescriptorKeys.CacheId] = Phys.PipelineCacheUuid,
            [ArtifactDescriptorKeys.Driver] = Phys.DriverVersion.ToString("x", CultureInfo.InvariantCulture),
            [ArtifactDescriptorKeys.DeviceName] = Phys.Name,
            [ArtifactDescriptorKeys.KernelAbi] = abi,
        };
        var cache = new ArtifactTarget(
            new Dictionary<string, string>(identity) { [ArtifactDescriptorKeys.Kind] = PipelineCacheKind },
            // Same model and driver cache id: valid on any card of that model, not only the one it was built on.
            [ArtifactDescriptorKeys.Vendor, ArtifactDescriptorKeys.Device, ArtifactDescriptorKeys.CacheId],
            [PipelineCacheKey]);
        if (!_kernelBinaries)
        {
            return [code, cache];
        }

        var binaries = new ArtifactTarget(
            new Dictionary<string, string>(identity)
            {
                [ArtifactDescriptorKeys.Kind] = PipelineBinaryKind,
                [ArtifactDescriptorKeys.BinaryKey] = Pipelines.GlobalBinaryKey(),
            },
            [ArtifactDescriptorKeys.Vendor, ArtifactDescriptorKeys.Device, ArtifactDescriptorKeys.BinaryKey, ArtifactDescriptorKeys.KernelAbi],
            [PipelineBinaryPrefix]);
        return [code, binaries, cache];
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

    /// <summary>
    /// State buffers per compiled graph (<see cref="CompileOptions.StateScope"/>): the first segment of the graph that uses a state allocates
    /// (and frees) its buffer, the later ones reuse it.
    /// </summary>
    internal System.Runtime.CompilerServices.ConditionalWeakTable<object, Dictionary<Models.StateDefinition, Allocation>> SharedStates { get; } = new();

    /// <summary>
    /// The segment that last submitted work using buffers shared with other segments (activation arenas, streamed weights), until a later
    /// such segment waits for it: executions are asynchronous and the queue does not order separate submissions.
    /// </summary>
    internal VulkanSegment? PendingShared { get; set; }

    private WeightUploader? _streamUploader;
    private long _streamLargestRow;
    private VkCommandPool _streamPool;

    /// <summary>Serializes the use of the streaming uploader (<see cref="StreamUploader"/>).</summary>
    internal Lock StreamLock { get; } = new();

    /// <summary>
    /// The uploader that segments streaming their weights (<see cref="CompileOptions.StreamWeights"/>) share, with its own command pool;
    /// its staging buffers are allocated on first use and kept. Hold <see cref="StreamLock"/> while using it and call
    /// <see cref="WeightUploader.Complete"/> before releasing the lock.
    /// </summary>
    internal WeightUploader StreamUploader(long largestRow)
    {
        if (_streamUploader is { } uploader && _streamLargestRow >= largestRow)
        {
            return uploader;
        }

        _streamUploader?.Dispose();
        if (_streamPool.Handle == 0)
        {
            _streamPool = CreateCommandPool();
        }

        _streamLargestRow = Math.Max(1, largestRow);
        return _streamUploader = new WeightUploader(this, _streamPool, long.MaxValue, _streamLargestRow);
    }

    /// <summary>Per <see cref="CompileOptions.StateScope"/>: the activation buffers its segments on this device share, by buffer name.</summary>
    internal System.Runtime.CompilerServices.ConditionalWeakTable<object, Dictionary<string, SharedArena<Allocation>>> SharedArenas { get; } = new();

    public ICompiledSegment Compile(GraphSegment segment, CompileOptions options)
    {
        ThrowIfDisposed();
        var cache = options.CacheFor?.Invoke(this) ?? _ownArtifacts.For(this);
        EnsurePipelineCache(cache);
        var program = Code.Lower(segment, this, options, cache, out var kernels);
        options.CancellationToken.ThrowIfCancellationRequested();

        // Slang keeps compiling in the background while the weights upload; pipelines are created as the kernels complete.
        var compiling = new LoadProgressCounter(options.Progress, LoadStage.Compiling, kernels.Count, Info.Name, kernels.Count(k => k.FromCache));
        var pipelines = Task.Run(() => CreatePipelines(kernels, cache, compiling, options.CancellationToken));
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
        try
        {
            pipelines.GetAwaiter().GetResult();
        }
        catch
        {
            compiled.Dispose();
            throw;
        }

        SavePipelineCache();
        if (DiagnosticSwitches.LoadTiming)
        {
            Console.Error.WriteLine($"[load] kernels and pipelines segment {segment.Index}: waited {swWait.ElapsedMilliseconds} ms for {kernels.Count} kernels; slang {SlangCompiler.Statistics}; {PipelinesCreated} pipelines, {(double)PipelineTicks / Stopwatch.Frequency * 1000:F0} ms summed ({PipelinesFromBinaries} from kernel binaries, {RejectedDeviceCode} stored device code rejected)");
        }

        return compiled;
    }

    /// <summary>Creates the pipelines of <paramref name="kernels"/> in parallel, waiting for kernels that are still compiling.</summary>
    /// <exception cref="KernelCompilationException">Slang rejected one of the kernels.</exception>
    private void CreatePipelines(IEnumerable<CompiledKernel> kernels, ICompilationCache? cache, LoadProgressCounter progress, CancellationToken ct)
    {
        try
        {
            Parallel.ForEach(kernels, new ParallelOptions { CancellationToken = ct }, kernel =>
            {
                Pipeline(kernel, cache);
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
            // The caller is already failing with another exception.
        }
    }

    /// <summary>
    /// Creates the driver pipeline cache on first compile, seeded from a saved model's cache when present and its header names this
    /// device and driver build, else from this device's pipeline cache set in the artifact store (same descriptor and key as the
    /// package's; the driver validates the blob's header too and ignores data from another device/driver).
    /// </summary>
    private void EnsurePipelineCache(ICompilationCache? cache)
    {
        lock (_pipelineCacheLock)
        {
            if (_pipelineCache != VkPipelineCache.Null)
            {
                return;
            }

            var data = ValidPipelineCache(cache);
            bool fromStore = false;
            if (data.IsEmpty && StoredPipelineCache is { } store)
            {
                data = ValidPipelineCache(store);
                fromStore = !data.IsEmpty;
            }

            if (TryCreatePipelineCache(data.Span, out var pc))
            {
                _pipelineCache = pc;
                return;
            }

            // The driver refused the seed blob (seen on NVIDIA with very large blobs as VK_ERROR_UNKNOWN): start empty, and drop
            // a refused stored blob so later processes do not retry it.
            if (fromStore)
            {
                StoredPipelineCache?.Remove(PipelineCacheKey);
            }

            if (!TryCreatePipelineCache(ReadOnlySpan<byte>.Empty, out pc))
            {
                throw new VkException(VkResult.ErrorInitializationFailed, "vkCreatePipelineCache failed without initial data");
            }

            _pipelineCache = pc;
        }
    }

    /// <summary>The pipeline cache blob in <paramref name="cache"/> when its header names this device and driver build, else empty.</summary>
    private ReadOnlyMemory<byte> ValidPipelineCache(ICompilationCache? cache)
    {
        if (cache is null || !cache.TryGet(PipelineCacheKey, out var data) || data.IsEmpty)
        {
            return default;
        }

        if (data.Length > MaxPipelineCacheBytes || !PipelineCacheHeader.Matches(data.Span, Phys.VendorId, Phys.DeviceId, Phys.PipelineCacheUuid))
        {
            // Damaged, oversized, or written by another driver build: never handed to the driver.
            Interlocked.Increment(ref RejectedDeviceCode);
            return default;
        }

        return data;
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

    /// <summary>
    /// The device-specific stage for this device (<see cref="VulkanProvider.CompileDeviceSpecificAsync"/>): creates the pipelines of every
    /// kernel of <paramref name="portable"/> and returns what the driver compiled them into, as kernel binaries when
    /// <see cref="DeviceFeatures.KernelBinaries"/> is enabled (falling back to a pipeline cache blob when the driver cannot capture
    /// them), else as one dedicated pipeline cache holding only these pipelines.
    /// </summary>
    /// <exception cref="ArgumentException">The code was generated for another SPIR-V profile than this device's.</exception>
    internal DeviceCodeSet CompileDeviceCode(PortableCodeSet portable, CancellationToken cancellationToken)
    {
        var kernels = portable.Entries.Where(e => e.Key.StartsWith(SlangCompiler.KernelKeyPrefix, StringComparison.Ordinal)).ToList();
        if (kernels.Count == 0)
        {
            return DeviceCodeSet.Empty(Info.Provider, Info.Name);
        }

        CheckProfile(portable);
        if (_kernelBinaries)
        {
            try
            {
                return CaptureKernelBinaries(kernels, cancellationToken);
            }
            catch (VkException)
            {
                // The driver could not capture binaries for some kernel: store the pipeline cache instead, which every driver supports.
            }
        }

        return CapturePipelineCache(kernels, cancellationToken);
    }

    /// <summary>Portable code is only meaningful for the profile it was generated for (subgroup size, shared memory, features).</summary>
    private void CheckProfile(PortableCodeSet portable)
    {
        var own = Code.CreateCodeTarget(Info.Provider).Descriptor;
        string[] keys = [ArtifactDescriptorKeys.Target, ArtifactDescriptorKeys.Subgroup, ArtifactDescriptorKeys.SharedMemory, ArtifactDescriptorKeys.Features];
        var different = keys.Where(k => ArtifactDescriptor.Get(own, k) != ArtifactDescriptor.Get(portable.Descriptor, k)).ToList();
        if (portable.Provider != Info.Provider || different.Count > 0)
        {
            throw new ArgumentException($"The portable code ({ArtifactDescriptor.Format(portable.Descriptor)}) was not generated for the profile of {Info} "
                + $"({ArtifactDescriptor.Format(own)}); differing: {string.Join(", ", different)}.", nameof(portable));
        }
    }

    private DeviceCodeSet CaptureKernelBinaries(List<KeyValuePair<string, ReadOnlyMemory<byte>>> kernels, CancellationToken cancellationToken)
    {
        var entries = new ConcurrentDictionary<string, ReadOnlyMemory<byte>>(StringComparer.Ordinal);
        RunParallel(kernels, cancellationToken, kernel =>
        {
            var module = Pipelines.CreateModule(kernel.Value.ToArray());
            try
            {
                var entry = Pipelines.Capture(module);
                entries[PipelineBinaryPrefix + kernel.Key[SlangCompiler.KernelKeyPrefix.Length..]] = PipelineBinaryCodec.Encode(entry);
            }
            finally
            {
                Api.vkDestroyShaderModule(module, null);
            }
        });
        var target = ArtifactTargets.Single(t => t.Kind == PipelineBinaryKind);
        return new DeviceCodeSet(Info.Provider, Info.Name, target.Descriptor, entries);
    }

    private DeviceCodeSet CapturePipelineCache(List<KeyValuePair<string, ReadOnlyMemory<byte>>> kernels, CancellationToken cancellationToken)
    {
        if (!TryCreatePipelineCache(ReadOnlySpan<byte>.Empty, out var dedicated))
        {
            throw new VkException(VkResult.ErrorInitializationFailed, "vkCreatePipelineCache failed without initial data");
        }

        try
        {
            RunParallel(kernels, cancellationToken, kernel =>
            {
                var module = Pipelines.CreateModule(kernel.Value.ToArray());
                try
                {
                    Api.vkDestroyPipeline(Pipelines.Create(module, dedicated), null);
                }
                finally
                {
                    Api.vkDestroyShaderModule(module, null);
                }
            });
            var target = ArtifactTargets.Single(t => t.Kind == PipelineCacheKind);
            return new DeviceCodeSet(Info.Provider, Info.Name, target.Descriptor,
                new Dictionary<string, ReadOnlyMemory<byte>> { [PipelineCacheKey] = GetPipelineCacheData(dedicated) });
        }
        finally
        {
            Api.vkDestroyPipelineCache(dedicated, null);
        }
    }

    private static void RunParallel<T>(IEnumerable<T> items, CancellationToken cancellationToken, Action<T> body)
    {
        try
        {
            Parallel.ForEach(items, new ParallelOptions { CancellationToken = cancellationToken }, body);
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

    /// <summary>
    /// This device's pipeline cache set in the artifact store (the binary-tier set a package stores the blob in, keyed by device model
    /// and driver cache id), or null without a store.
    /// </summary>
    private FileCompilationCache? StoredPipelineCache
        => ArtifactStore?.EntriesFor(ArtifactTargets.Single(t => t.Kind == PipelineCacheKind));

    /// <summary>
    /// This device's kernel binary set in the artifact store (device code a host stored with <see cref="ArtifactStore.Add"/>, e.g. for a
    /// model loaded straight from a GGUF file), or null without a store or kernel binaries.
    /// </summary>
    private FileCompilationCache? StoredKernelBinaries
        => _kernelBinaries ? ArtifactStore?.EntriesFor(ArtifactTargets.Single(t => t.Kind == PipelineBinaryKind)) : null;

    /// <summary>
    /// Largest pipeline cache blob that is loaded or stored. The driver blob accumulates every pipeline ever created on the device (all
    /// models, all specializations); past this size it is reset so it is rebuilt from the pipelines in actual use.
    /// </summary>
    private const long MaxPipelineCacheBytes = 256L * 1024 * 1024;

    /// <summary>
    /// Stores the driver pipeline cache in the artifact store when pipelines were created since it was last saved, so that the next
    /// process skips the driver's SPIR-V compilation. Failing to write it is not an error.
    /// </summary>
    private void SavePipelineCache()
    {
        lock (_pipelineCacheLock)
        {
            int created = Volatile.Read(ref PipelinesCreated);
            if (_pipelineCache == VkPipelineCache.Null || created == _pipelinesAtLastSave || StoredPipelineCache is not { } store)
            {
                return;
            }

            _pipelinesAtLastSave = created;
            var data = GetPipelineCacheData(_pipelineCache);
            if (data.Length > MaxPipelineCacheBytes)
            {
                // Too large to keep: reset, so the next process rebuilds the cache from the pipelines it uses.
                store.Remove(PipelineCacheKey);
                return;
            }

            store.Put(PipelineCacheKey, data);
        }
    }

    /// <summary>The compute pipeline of <paramref name="kernel"/>, created on first use; safe to call from several threads.</summary>
    /// <param name="kernel">The kernel.</param>
    /// <param name="cache">Model-level cache that may hold the kernel's stored binary, or <see langword="null"/>.</param>
    internal VkPipeline Pipeline(CompiledKernel kernel, ICompilationCache? cache = null)
    {
        if (_pipelines.TryGetValue(kernel.CacheKey, out var existing))
        {
            return existing.Value;
        }

        // Lazy: GetOrAdd may run its factory twice under a race (leaking a VkPipeline); pipelines of different kernels are
        // created in parallel (the driver's pipeline cache is internally synchronized).
        return _pipelines.GetOrAdd(kernel.CacheKey, _ => new Lazy<VkPipeline>(() => CreateDevicePipeline(kernel, cache))).Value;
    }

    /// <summary>
    /// Creates a pipeline from the kernel's stored binary when there is a valid one for this device, else compiles it in the device's
    /// driver cache; its shader module lives as long as the device.
    /// </summary>
    private VkPipeline CreateDevicePipeline(CompiledKernel kernel, ICompilationCache? cache)
    {
        // Waits for a kernel that is still compiling; that time is not pipeline creation.
        var spirv = kernel.Code;
        long startTicks = Stopwatch.GetTimestamp();
        var module = Pipelines.CreateModule(spirv);
        lock (_modules)
        {
            _modules.Add(module);
        }

        try
        {
            if (TryCreateFromStoredBinary(kernel, module, cache) is { } fromBinary)
            {
                return fromBinary;
            }

            var pipeline = Pipelines.Create(module, _pipelineCache);
            if (_pipelineStats)
            {
                PrintPipelineStatistics(kernel, pipeline);
            }

            return pipeline;
        }
        finally
        {
            Interlocked.Add(ref PipelineTicks, Stopwatch.GetTimestamp() - startTicks);
            Interlocked.Increment(ref PipelinesCreated);
        }
    }

    /// <summary>The pipeline from the kernel's stored binary, or null when there is none, it is damaged or the driver rejects it.</summary>
    private VkPipeline? TryCreateFromStoredBinary(CompiledKernel kernel, VkShaderModule module, ICompilationCache? cache)
    {
        var key = PipelineBinaryPrefix + kernel.CacheKey;
        ReadOnlyMemory<byte> stored = default;
        if (!_kernelBinaries
            || !((cache is not null && cache.TryGet(key, out stored)) || (StoredKernelBinaries is { } store && store.TryGet(key, out stored))))
        {
            return null;
        }

        if (PipelineBinaryCodec.TryDecode(stored.Span, out var entry))
        {
            var pipeline = Pipelines.TryCreateFromBinaries(module, entry!);
            if (pipeline != VkPipeline.Null)
            {
                Interlocked.Increment(ref PipelinesFromBinaries);
                return pipeline;
            }
        }

        Interlocked.Increment(ref RejectedDeviceCode);
        return null;
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

    /// <summary>Binds memory to sparse buffers on the compute queue (which needs the same external synchronization as submits) and waits.</summary>
    internal void BindSparseAndWait(VkBindSparseInfo* info)
    {
        var fci = new VkFenceCreateInfo();
        VkFence fence;
        Api.vkCreateFence(&fci, null, &fence).CheckResult();
        try
        {
            lock (_queueLock)
            {
                Api.vkQueueBindSparse(Queue, 1, info, fence).CheckResult();
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

    /// <summary>
    /// Submits <paramref name="cbs"/> on the compute queue, signalling <paramref name="ready"/>, and then <paramref name="download"/>
    /// on the transfer queue, waiting for <paramref name="ready"/> and signalling <paramref name="fence"/>.
    /// </summary>
    internal void SubmitWithDownload(ReadOnlySpan<VkCommandBuffer> cbs, VkSemaphore ready, VkCommandBuffer download, VkFence fence)
    {
        fixed (VkCommandBuffer* p = cbs)
        {
            var si = new VkSubmitInfo { commandBufferCount = (uint)cbs.Length, pCommandBuffers = p, signalSemaphoreCount = 1, pSignalSemaphores = &ready };
            lock (_queueLock)
            {
                Api.vkQueueSubmit(Queue, 1, &si, VkFence.Null).CheckResult();
            }
        }

        var stage = VkPipelineStageFlags.Transfer;
        var ti = new VkSubmitInfo { commandBufferCount = 1, pCommandBuffers = &download, waitSemaphoreCount = 1, pWaitSemaphores = &ready, pWaitDstStageMask = &stage };
        lock (_transferLock)
        {
            Api.vkQueueSubmit(TransferQueue, 1, &ti, fence).CheckResult();
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

    /// <summary>
    /// Throws when the device was disposed. Its memory is unmapped and freed by then, so a model that still uses it (a device
    /// disposed while models compiled on it were alive) would otherwise write to unmapped staging memory and crash the process.
    /// </summary>
    /// <exception cref="ObjectDisposedException">The device was disposed.</exception>
    internal void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    /// <summary>Whether the device was disposed (its Vulkan objects and memory are gone).</summary>
    internal bool IsDisposed => _disposed;

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        VulkanShutdown.Untrack(this);
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

        _streamUploader?.Dispose();
        if (_streamPool.Handle != 0)
        {
            Api.vkDestroyCommandPool(_streamPool, null);
        }

        HostImports?.Dispose();
        Memory.Dispose();
        Api.vkDestroyPipelineLayout(PipelineLayout, null);
        Api.vkDestroyCommandPool(CommandPool, null);
        Api.vkDestroyDevice(null);
    }
}
