using System.Runtime.InteropServices;
using System.Text;
using Sharplight.Runtime;
using Vortice.Vulkan;
using static Vortice.Vulkan.Vulkan;

namespace Sharplight.Vulkan;

/// <summary>
/// Vulkan compute execution provider. Requires Vulkan 1.2+ with buffer device addresses,
/// 8/16-bit storage and 64-bit integers (kernels address memory through raw pointers).
/// </summary>
public sealed unsafe class VulkanProvider : IExecutionProvider
{
    static readonly Lock s_lock = new();
    static VulkanInstance? s_instance;
    static string? s_initError;

    public string Name => "vulkan";

    /// <summary>Enable the Khronos validation layer for instances created after this is set.</summary>
    public static bool EnableValidation { get; set; } = Environment.GetEnvironmentVariable("SHARPLIGHT_VK_VALIDATION") == "1";

    internal static VulkanInstance Instance
    {
        get
        {
            lock (s_lock)
            {
                if (s_instance is not null) return s_instance;
                if (s_initError is not null) throw new InvalidOperationException(s_initError);
                try { return s_instance = new VulkanInstance(EnableValidation); }
                catch (Exception e) { s_initError = e.Message; throw; }
            }
        }
    }

    public ProviderAvailability CheckAvailability()
    {
        try
        {
            var inst = Instance;
            var usable = inst.PhysicalDevices.Where(p => p.MissingFeatures.Count == 0).ToList();
            if (usable.Count == 0)
                return ProviderAvailability.Unavailable("No Vulkan device with the required features: " +
                    string.Join("; ", inst.PhysicalDevices.Select(p => $"{p.Name}: missing {string.Join(", ", p.MissingFeatures)}")));
            return ProviderAvailability.Available($"Vulkan {inst.ApiVersion}, {usable.Count} usable device(s)");
        }
        catch (Exception e)
        {
            return ProviderAvailability.Unavailable("Vulkan loader/driver not available: " + e.Message);
        }
    }

    public IReadOnlyList<DeviceInfo> GetDevices()
    {
        var inst = Instance;
        var list = new List<DeviceInfo>();
        for (int i = 0; i < inst.PhysicalDevices.Count; i++)
        {
            var p = inst.PhysicalDevices[i];
            if (p.MissingFeatures.Count > 0) continue;
            list.Add(p.ToDeviceInfo(i));
        }
        return list;
    }

    public IComputeDevice OpenDevice(DeviceInfo device, DeviceOptions? options = null)
    {
        if (device.Provider != Name) throw new ArgumentException($"Device belongs to provider '{device.Provider}'.");
        return new VulkanDevice(Instance, Instance.PhysicalDevices[device.Index], device, options ?? new DeviceOptions());
    }

    public void Dispose() { }
}

/// <summary>Process-wide Vulkan instance and physical device inventory.</summary>
internal sealed unsafe class VulkanInstance
{
    public VkInstance Handle { get; }
    public VkInstanceApi Api { get; }
    public IReadOnlyList<PhysicalDeviceInfo> PhysicalDevices { get; }
    public VkVersion ApiVersion { get; }

    public VulkanInstance(bool validation)
    {
        vkInitialize().CheckResult();
        uint ver = 0;
        vkEnumerateInstanceVersion(&ver);
        ApiVersion = new VkVersion(ver);
        if (ApiVersion < VkVersion.Version_1_2) throw new NotSupportedException($"Vulkan {ApiVersion} < 1.2");

        VkUtf8ReadOnlyString appName = "Sharplight"u8;
        var app = new VkApplicationInfo { pApplicationName = appName, apiVersion = VkVersion.Version_1_3, engineVersion = new VkVersion(0, 1, 0), pEngineName = appName };
        var layers = new List<nint>();
        if (validation) layers.Add(Marshal.StringToHGlobalAnsi("VK_LAYER_KHRONOS_validation"));
        // Allow portability drivers (MoltenVK) to be enumerated.
        var exts = new List<nint>();
        uint extCount = 0;
        vkEnumerateInstanceExtensionProperties(&extCount, null);
        var extProps = new VkExtensionProperties[extCount];
        fixed (VkExtensionProperties* ep = extProps) vkEnumerateInstanceExtensionProperties(&extCount, ep);
        bool portability = extProps.Any(e => Name(e) == "VK_KHR_portability_enumeration");
        if (portability) exts.Add(Marshal.StringToHGlobalAnsi("VK_KHR_portability_enumeration"));
        try
        {
            fixed (nint* pl = layers.ToArray())
            fixed (nint* pe = exts.ToArray())
            {
                var ci = new VkInstanceCreateInfo
                {
                    pApplicationInfo = &app,
                    enabledLayerCount = (uint)layers.Count,
                    ppEnabledLayerNames = (byte**)pl,
                    enabledExtensionCount = (uint)exts.Count,
                    ppEnabledExtensionNames = (byte**)pe,
                    flags = portability ? VkInstanceCreateFlags.EnumeratePortabilityKHR : 0,
                };
                VkInstance inst;
                vkCreateInstance(&ci, null, &inst).CheckResult();
                Handle = inst;
            }
        }
        finally
        {
            foreach (var p in layers.Concat(exts)) Marshal.FreeHGlobal(p);
        }
        Api = new VkInstanceApi(Handle);

        uint n = 0;
        Api.vkEnumeratePhysicalDevices(&n, null);
        var pds = new VkPhysicalDevice[n];
        fixed (VkPhysicalDevice* pp = pds) Api.vkEnumeratePhysicalDevices(&n, pp);
        PhysicalDevices = [.. pds.Select(pd => new PhysicalDeviceInfo(this, pd))];
    }

    internal static string Name(VkExtensionProperties e) => Encoding.UTF8.GetString(MemoryMarshal.CreateReadOnlySpanFromNullTerminated(e.extensionName));
}

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
        fixed (VkExtensionProperties* e = eps) api.vkEnumerateDeviceExtensionProperties(pd, null, &ec, e);
        foreach (var e in eps) Extensions.Add(VulkanInstance.Name(e));

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
        var coop = new VkPhysicalDeviceCooperativeMatrixFeaturesKHR();
        var f13 = new VkPhysicalDeviceVulkan13Features { pNext = Extensions.Contains("VK_KHR_cooperative_matrix") ? &coop : null };
        var f12 = new VkPhysicalDeviceVulkan12Features { pNext = &f13 };
        var f11 = new VkPhysicalDeviceVulkan11Features { pNext = &f12 };
        var f2 = new VkPhysicalDeviceFeatures2 { pNext = &f11 };
        api.vkGetPhysicalDeviceFeatures2(pd, &f2);
        SubgroupSizeControl = f13.subgroupSizeControl;
        ComputeFullSubgroups = f13.computeFullSubgroups;
        ShaderFloat16 = f12.shaderFloat16;
        VulkanMemoryModel = f12.vulkanMemoryModel;
        CooperativeMatrix = coop.cooperativeMatrix;
        CooperativeMatrix2 = Extensions.Contains("VK_NV_cooperative_matrix2");
        if (CooperativeMatrix)
        {
            uint cn = 0;
            api.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(pd, &cn, null);
            var cps = new VkCooperativeMatrixPropertiesKHR[cn];
            for (int i = 0; i < cn; i++) cps[i] = new VkCooperativeMatrixPropertiesKHR();
            fixed (VkCooperativeMatrixPropertiesKHR* cp = cps) api.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(pd, &cn, cp);
            CoopMatShapes = cps.Where(c => c.scope == VkScopeKHR.Subgroup)
                .Select(c => new CoopMatShape(c.MSize, c.NSize, c.KSize, c.AType.ToString(), c.BType.ToString(), c.CType.ToString(), c.ResultType.ToString()))
                .ToArray();
        }
        ShaderBFloat16 = Extensions.Contains("VK_KHR_shader_bfloat16");
        IntegerDotProduct = f13.shaderIntegerDotProduct;

        // Map Vulkan mechanisms onto provider-neutral feature names. A matrix feature is only reported
        // when the exact tile configuration our kernels use is supported.
        var feats = new HashSet<string> { DeviceFeatures.SubgroupOps };
        bool Shape(string s) => CoopMatShapes.Any(c => c.ToString() == s);
        if (ShaderFloat16) feats.Add(DeviceFeatures.Float16);
        if (ShaderBFloat16) feats.Add(DeviceFeatures.BFloat16);
        if (IntegerDotProduct) feats.Add(DeviceFeatures.IntDot);
        if (SubgroupSizeControl && ComputeFullSubgroups) feats.Add(DeviceFeatures.SubgroupSizeControl);
        if (CooperativeMatrix && ShaderFloat16 && Shape("16x16x16:f16*f16+f32->f32")) feats.Add(DeviceFeatures.MatrixF16);
        if (CooperativeMatrix && ShaderBFloat16 && Shape("16x16x16:bf16*bf16+f32->f32")) feats.Add(DeviceFeatures.MatrixBF16);
        if (CooperativeMatrix && Shape("16x16x32:i8*i8+i32->i32")) feats.Add(DeviceFeatures.MatrixInt8);
        Features = feats;

        void Need(bool ok, string what) { if (!ok) MissingFeatures.Add(what); }
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
        for (int i = 0; i < mp.memoryHeapCount; i++)
            if ((mp.memoryHeaps[i].flags & VkMemoryHeapFlags.DeviceLocal) != 0) local = Math.Max(local, (long)mp.memoryHeaps[i].size);
        DeviceLocalBytes = local;
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
        },
    };
}

/// <summary>One supported cooperative-matrix configuration: A[M,K]·B[K,N] + C[M,N].</summary>
internal readonly record struct CoopMatShape(uint M, uint N, uint K, string A, string B, string C, string Result)
{
    static string Short(string t) => t.Replace("Float", "f").Replace("Sint", "i").Replace("Uint", "u").Replace("KHR", "").Replace("Bfloat", "bf");
    public override string ToString() => $"{M}x{N}x{K}:{Short(A)}*{Short(B)}+{Short(C)}->{Short(Result)}";
    public bool Is(uint m, uint n, uint k, string a, string c) => M == m && N == n && K == k && A == a && B == a && C == c && Result == c;
}
