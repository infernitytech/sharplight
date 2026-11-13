using System.Runtime.InteropServices;
using System.Text;
using Sharplight.Runtime;
using Vortice.Vulkan;

namespace Sharplight.Vulkan;

/// <summary>Capabilities of one physical device.</summary>
internal sealed unsafe class PhysicalDeviceInfo
{
    public VkPhysicalDevice Handle { get; }
    public string Name { get; }
    public VkPhysicalDeviceType Type { get; }
    public uint VendorId { get; }
    public uint DriverVersion { get; }
    public VkVersion ApiVersion { get; }
    public HashSet<string> Extensions { get; } = [];
    public long DeviceLocalBytes { get; }
    /// <summary>Index of the largest device-local heap (the VRAM heap the budget is read from).</summary>
    public int DeviceLocalHeap { get; }
    /// <summary>VK_EXT_memory_budget is available.</summary>
    public bool MemoryBudgetExt { get; }
    public uint SubgroupSize { get; }
    public uint MinSubgroupSize { get; }
    public uint MaxSubgroupSize { get; }
    public bool SubgroupSizeControl { get; }
    public bool ComputeFullSubgroups { get; }
    public uint MaxPushConstants { get; }
    /// <summary>Nanoseconds per timestamp tick.</summary>
    public float TimestampPeriod { get; }
    public uint MaxSharedMemory { get; }
    public uint[] MaxWorkgroupCount { get; }
    public bool CooperativeMatrix { get; }
    public bool CooperativeMatrix2 { get; }
    /// <summary>Supported subgroup-scope cooperative matrix configurations (VK_KHR_cooperative_matrix).</summary>
    public IReadOnlyList<CoopMatShape> CoopMatShapes { get; } = [];
    public bool ShaderFloat16 { get; }
    public bool ShaderBFloat16 { get; }
    public bool BFloat16CooperativeMatrix { get; }
    public bool IntegerDotProduct { get; }
    /// <summary>Provider-neutral features (<see cref="DeviceFeatures"/>) this device supports.</summary>
    public IReadOnlySet<string> Features { get; }
    public bool VulkanMemoryModel { get; }
    public List<string> MissingFeatures { get; } = [];
    public string? Uuid { get; }

    public PhysicalDeviceInfo(VulkanInstance inst, VkPhysicalDevice pd)
    {
        Handle = pd;
        var api = inst.Api;

        uint ec = 0;
        api.vkEnumerateDeviceExtensionProperties(pd, null, &ec, null);
        var eps = new VkExtensionProperties[ec];
        fixed (VkExtensionProperties* e = eps)
        {
            api.vkEnumerateDeviceExtensionProperties(pd, null, &ec, e);
        }

        foreach (var e in eps)
        {
            Extensions.Add(VulkanInstance.Name(e));
        }

        var sub = new VkPhysicalDeviceSubgroupProperties();
        var ssc = new VkPhysicalDeviceSubgroupSizeControlProperties { pNext = &sub };
        var idp = new VkPhysicalDeviceIDProperties { pNext = &ssc };
        var props2 = new VkPhysicalDeviceProperties2 { pNext = &idp };
        api.vkGetPhysicalDeviceProperties2(pd, &props2);
        var p = props2.properties;
        Name = Encoding.UTF8.GetString(MemoryMarshal.CreateReadOnlySpanFromNullTerminated(p.deviceName));
        Type = p.deviceType;
        VendorId = p.vendorID;
        DriverVersion = p.driverVersion;
        ApiVersion = new VkVersion(p.apiVersion);
        SubgroupSize = sub.subgroupSize;
        MinSubgroupSize = ssc.minSubgroupSize;
        MaxSubgroupSize = ssc.maxSubgroupSize;
        MaxPushConstants = p.limits.maxPushConstantsSize;
        TimestampPeriod = p.limits.timestampPeriod;
        MaxSharedMemory = p.limits.maxComputeSharedMemorySize;
        MaxWorkgroupCount = [p.limits.maxComputeWorkGroupCount[0], p.limits.maxComputeWorkGroupCount[1], p.limits.maxComputeWorkGroupCount[2]];
        Uuid = Convert.ToHexString(new ReadOnlySpan<byte>(idp.deviceUUID, 16));

        // features
        var coop2 = new VkPhysicalDeviceCooperativeMatrix2FeaturesNV();
        var coop = new VkPhysicalDeviceCooperativeMatrixFeaturesKHR { pNext = Extensions.Contains("VK_NV_cooperative_matrix2") ? &coop2 : null };
        var f13 = new VkPhysicalDeviceVulkan13Features { pNext = Extensions.Contains("VK_KHR_cooperative_matrix") ? &coop : null };
        var f12 = new VkPhysicalDeviceVulkan12Features { pNext = &f13 };
        var f11 = new VkPhysicalDeviceVulkan11Features { pNext = &f12 };
        bool bf16Extension = Extensions.Contains("VK_KHR_shader_bfloat16");
        var bf16 = new VkPhysicalDeviceShaderBfloat16FeaturesKHR { pNext = &f11 };
        var f2 = new VkPhysicalDeviceFeatures2 { pNext = bf16Extension ? &bf16 : &f11 };
        api.vkGetPhysicalDeviceFeatures2(pd, &f2);
        SubgroupSizeControl = f13.subgroupSizeControl;
        ComputeFullSubgroups = f13.computeFullSubgroups;
        ShaderFloat16 = f12.shaderFloat16;
        VulkanMemoryModel = f12.vulkanMemoryModel;
        CooperativeMatrix = coop.cooperativeMatrix;
        if (CooperativeMatrix)
        {
            uint cn = 0;
            api.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(pd, &cn, null);
            var cps = new VkCooperativeMatrixPropertiesKHR[cn];
            for (int i = 0; i < cn; i++)
            {
                cps[i] = new VkCooperativeMatrixPropertiesKHR();
            }

            fixed (VkCooperativeMatrixPropertiesKHR* cp = cps)
            {
                api.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(pd, &cn, cp);
            }

            CoopMatShapes = cps.Where(c => c.scope == VkScopeKHR.Subgroup)
                .Select(c => new CoopMatShape(c.MSize, c.NSize, c.KSize, c.AType.ToString(), c.BType.ToString(), c.CType.ToString(), c.ResultType.ToString()))
                .ToArray();
        }
        // VK_NV_cooperative_matrix2: workgroup-scope matrices with tensor-addressed / decoded loads. Our kernels use
        // 256-invocation workgroups with f16 inputs and f32 accumulation (the configuration llama.cpp also requires).
        if (CooperativeMatrix && Extensions.Contains("VK_NV_cooperative_matrix2") && coop2.cooperativeMatrixWorkgroupScope
            && coop2.cooperativeMatrixFlexibleDimensions && coop2.cooperativeMatrixTensorAddressing && coop2.cooperativeMatrixBlockLoads)
        {
            uint fn = 0;
            api.vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(pd, &fn, null);
            var fps = new VkCooperativeMatrixFlexibleDimensionsPropertiesNV[fn];
            for (int i = 0; i < fn; i++)
            {
                fps[i] = new VkCooperativeMatrixFlexibleDimensionsPropertiesNV();
            }

            fixed (VkCooperativeMatrixFlexibleDimensionsPropertiesNV* fp = fps)
            {
                api.vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(pd, &fn, fp);
            }

            CooperativeMatrix2 = fps.Any(f => f.scope == VkScopeKHR.Workgroup && f.workgroupInvocations == 256
                && f.AType == VkComponentTypeKHR.Float16 && f.BType == VkComponentTypeKHR.Float16
                && f.CType == VkComponentTypeKHR.Float32 && f.ResultType == VkComponentTypeKHR.Float32
                && f.MGranularity <= 32 && f.NGranularity <= 32 && f.KGranularity <= 16);
        }
        ShaderBFloat16 = bf16Extension && bf16.shaderBFloat16Type;
        BFloat16CooperativeMatrix = ShaderBFloat16 && bf16.shaderBFloat16CooperativeMatrix;
        IntegerDotProduct = f13.shaderIntegerDotProduct;

        // Map Vulkan mechanisms onto provider-neutral feature names. A matrix feature is only reported
        // when the exact tile configuration our kernels use is supported.
        var feats = new HashSet<string> { DeviceFeatures.SubgroupOps };
        bool Shape(string s) => CoopMatShapes.Any(c => c.ToString() == s);
        if (ShaderFloat16)
        {
            feats.Add(DeviceFeatures.Float16);
        }

        if (ShaderBFloat16)
        {
            feats.Add(DeviceFeatures.BFloat16);
        }

        if (IntegerDotProduct)
        {
            feats.Add(DeviceFeatures.IntDot);
        }

        if (SubgroupSizeControl && ComputeFullSubgroups)
        {
            feats.Add(DeviceFeatures.SubgroupSizeControl);
        }

        if (CooperativeMatrix && ShaderFloat16 && Shape("16x16x16:f16*f16+f32->f32"))
        {
            feats.Add(DeviceFeatures.MatrixF16);
        }

        if (CooperativeMatrix && BFloat16CooperativeMatrix && Shape("16x16x16:bf16*bf16+f32->f32"))
        {
            feats.Add(DeviceFeatures.MatrixBF16);
        }

        if (CooperativeMatrix && Shape("16x16x32:i8*i8+i32->i32"))
        {
            feats.Add(DeviceFeatures.MatrixInt8);
        }

        if (CooperativeMatrix2 && feats.Contains(DeviceFeatures.MatrixF16) && SubgroupSize == 32)
        {
            feats.Add(DeviceFeatures.MatrixWorkgroup);
        }

        Features = feats;

        void Need(bool ok, string what) { if (!ok) { MissingFeatures.Add(what); } }
        Need(ApiVersion >= VkVersion.Version_1_2, "Vulkan 1.2");
        Need(f12.bufferDeviceAddress, "bufferDeviceAddress");
        Need(f12.storageBuffer8BitAccess, "storageBuffer8BitAccess");
        Need(f12.shaderInt8, "shaderInt8");
        Need(f11.storageBuffer16BitAccess, "storageBuffer16BitAccess");
        Need(f2.features.shaderInt64, "shaderInt64");
        Need(f2.features.shaderInt16, "shaderInt16");
        Need((sub.supportedOperations & VkSubgroupFeatureFlags.Arithmetic) != 0, "subgroup arithmetic");
        Need((sub.supportedOperations & VkSubgroupFeatureFlags.Shuffle) != 0, "subgroup shuffle");
        Need(SubgroupSize >= 8, "subgroup size >= 8");

        var mp = new VkPhysicalDeviceMemoryProperties();
        api.vkGetPhysicalDeviceMemoryProperties(pd, &mp);
        long local = 0;
        int heap = 0;
        for (int i = 0; i < mp.memoryHeapCount; i++)
        {
            if ((mp.memoryHeaps[i].flags & VkMemoryHeapFlags.DeviceLocal) != 0 && (long)mp.memoryHeaps[i].size > local)
            {
                local = (long)mp.memoryHeaps[i].size;
                heap = i;
            }
        }

        DeviceLocalBytes = local;
        DeviceLocalHeap = heap;
        MemoryBudgetExt = Extensions.Contains("VK_EXT_memory_budget");
    }

    public DeviceInfo ToDeviceInfo(int index) => new()
    {
        Provider = "vulkan",
        Index = index,
        Name = Name,
        Kind = Type switch
        {
            VkPhysicalDeviceType.DiscreteGpu => DeviceKind.DiscreteGpu,
            VkPhysicalDeviceType.IntegratedGpu => DeviceKind.IntegratedGpu,
            VkPhysicalDeviceType.VirtualGpu => DeviceKind.VirtualGpu,
            VkPhysicalDeviceType.Cpu => DeviceKind.Cpu,
            _ => DeviceKind.Other,
        },
        TotalMemory = DeviceLocalBytes,
        Vendor = VendorId switch { 0x10DE => "NVIDIA", 0x1002 => "AMD", 0x8086 => "Intel", 0x13B5 => "ARM", 0x5143 => "Qualcomm", 0x106B => "Apple", _ => $"0x{VendorId:X4}" },
        Driver = DriverVersion.ToString(),
        Uuid = Uuid,
        Features = Features,
        // Display-driving / host-memory-sharing GPUs get a watchdog-friendly submission size; discrete GPUs run flat out.
        SubmissionBudgetMs = Type == VkPhysicalDeviceType.IntegratedGpu ? 250 : 0,
        Capabilities = new Dictionary<string, string>
        {
            ["api_version"] = ApiVersion.ToString(),
            ["subgroup_size"] = SubgroupSize.ToString(),
            ["subgroup_size_range"] = $"{MinSubgroupSize}-{MaxSubgroupSize}",
            ["fp16"] = ShaderFloat16.ToString(),
            ["bf16"] = ShaderBFloat16.ToString(),
            ["cooperative_matrix"] = CooperativeMatrix.ToString(),
            ["cooperative_matrix2_nv"] = CooperativeMatrix2.ToString(),
            ["coopmat_shapes"] = string.Join(" ", CoopMatShapes.Select(c => c.ToString())),
            ["integer_dot_product"] = IntegerDotProduct.ToString(),
            ["max_shared_memory"] = MaxSharedMemory.ToString(),
            ["max_push_constants"] = MaxPushConstants.ToString(),
            ["memory_budget"] = MemoryBudgetExt.ToString(),
        },
    };
}
