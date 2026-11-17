using System.Globalization;
using Sharplight.Metal.Interop;
using Sharplight.Runtime;

namespace Sharplight.Metal;

/// <summary>What one Metal GPU reports: identity, memory, limits and the GPU families its features derive from.</summary>
internal sealed class MetalDeviceProperties
{
    /// <summary>Threadgroup memory the kernels are specialized for at most (the Apple GPUs have 32 KB).</summary>
    private const long MaxKernelSharedMemory = 32768;

    /// <summary><see cref="DeviceInfo.ConcurrentWorkgroups"/> of the Apple GPUs (sized like integrated GPUs, see docs/macos.md).</summary>
    private const int SmallDeviceWorkgroups = 64;

    private MetalDeviceProperties()
    {
    }

    /// <summary>The <c>MTLDevice</c> (retained by the provider for the process lifetime).</summary>
    public nint Handle { get; private init; }

    public string Name { get; private init; } = "";

    /// <summary>The IORegistry id: stable for a GPU while the machine runs.</summary>
    public ulong RegistryId { get; private init; }

    public bool UnifiedMemory { get; private init; }

    /// <summary>Memory the GPU should use at most without hurting performance (<c>recommendedMaxWorkingSetSize</c>).</summary>
    public long WorkingSetBytes { get; private init; }

    public long MaxBufferBytes { get; private init; }

    public long MaxSharedMemory { get; private init; }

    public long MaxThreadsPerThreadgroup { get; private init; }

    /// <summary>Highest <c>MTLGPUFamilyApple*</c> generation the GPU supports (7 = M1), 0 for none.</summary>
    public int AppleFamily { get; private init; }

    /// <summary>Supports <c>MTLGPUFamilyMetal3</c>.</summary>
    public bool Metal3 { get; private init; }

    /// <summary>Supports <c>MTLGPUFamilyMac2</c>.</summary>
    public bool Mac2 { get; private init; }

    /// <summary><c>argumentBuffersSupport</c>: 0 = tier 1, 1 = tier 2 (required).</summary>
    public ulong ArgumentBuffersTier { get; private init; }

    public bool Metal4 { get; private init; }

    /// <summary>Required capabilities the GPU lacks (the device is not listed when non-empty).</summary>
    public IReadOnlyList<string> MissingFeatures { get; private init; } = [];

    /// <summary>Reads the properties of <paramref name="device"/>. The caller holds an autorelease pool.</summary>
    public static MetalDeviceProperties Query(nint device)
    {
        int apple = 0;
        foreach (var (family, generation) in new[] { (MetalConstants.FamilyApple10, 10), (MetalConstants.FamilyApple9, 9), (MetalConstants.FamilyApple8, 8), (MetalConstants.FamilyApple7, 7) })
        {
            if (Supports(device, family))
            {
                apple = generation;
                break;
            }
        }

        bool metal3 = Supports(device, MetalConstants.FamilyMetal3);
        ulong argumentBuffers = ObjCRuntime.SendUInt64(device, Selectors.ArgumentBuffersSupport);
        // The kernels take buffer GPU addresses (MTLBuffer.gpuAddress, macOS 13) inside their argument struct, which needs argument
        // buffers tier 2. The Metal 3 family implies both, but paravirtual GPUs (CI) report tier 2 without claiming the family.
        var missing = new List<string>();
        if (!OperatingSystem.IsMacOSVersionAtLeast(13))
        {
            missing.Add("macOS 13 or later (buffer GPU addresses)");
        }

        if (argumentBuffers < MetalConstants.ArgumentBuffersTier2)
        {
            missing.Add("argument buffers tier 2");
        }

        var threads = ObjCRuntime.SendSize(device, Selectors.MaxThreadsPerThreadgroup);
        return new MetalDeviceProperties
        {
            Handle = device,
            Name = Foundation.ToManaged(ObjCRuntime.Send(device, Selectors.Name)),
            RegistryId = ObjCRuntime.SendUInt64(device, Selectors.RegistryId),
            UnifiedMemory = ObjCRuntime.SendBool(device, Selectors.HasUnifiedMemory) != 0,
            WorkingSetBytes = (long)ObjCRuntime.SendUInt64(device, Selectors.RecommendedMaxWorkingSetSize),
            MaxBufferBytes = (long)ObjCRuntime.SendUInt64(device, Selectors.MaxBufferLength),
            MaxSharedMemory = Math.Min(MaxKernelSharedMemory, (long)ObjCRuntime.SendUInt64(device, Selectors.MaxThreadgroupMemoryLength)),
            MaxThreadsPerThreadgroup = (long)threads.Width,
            AppleFamily = apple,
            Metal3 = metal3,
            Mac2 = Supports(device, MetalConstants.FamilyMac2),
            ArgumentBuffersTier = argumentBuffers,
            Metal4 = Supports(device, MetalConstants.FamilyMetal4),
            MissingFeatures = missing,
        };
    }

    private static bool Supports(nint device, nint family) => ObjCRuntime.SendBool(device, Selectors.SupportsFamily, family) != 0;

    /// <summary>
    /// Device features with the neutral names of <see cref="DeviceFeatures"/>: half precision and SIMD-group operations exist on every
    /// GPU the provider lists, as is <see cref="DeviceFeatures.IntDot"/> (emulated); <c>simdgroup_matrix</c> is <see cref="DeviceFeatures.MatrixF16Small"/> (8x8 fragments, so the kernels
    /// written for the 16x16 fragments of <see cref="DeviceFeatures.MatrixF16"/> keep their SIMT fallbacks).
    /// </summary>
    public IReadOnlySet<string> Features()
    {
        // int_dot: Metal has no packed int8 dot instruction; the integer-dot matvecs compile to unpacked integer multiply-adds, which
        // still beat the f32 matvecs for Q8_0 decode (fewer bytes of activations per weight byte).
        var features = new HashSet<string> { DeviceFeatures.Float16, DeviceFeatures.SubgroupOps, DeviceFeatures.IntDot };
        if (AppleFamily >= 7 || Mac2)
        {
            // simdgroup_matrix (8x8 f16 / f32 fragments): Apple7 (A14, M1) and later, and the Mac2 family.
            features.Add(DeviceFeatures.MatrixF16Small);
        }

        return features;
    }

    public DeviceInfo ToDeviceInfo(int index) => new()
    {
        Provider = MetalProvider.ProviderName,
        Index = index,
        Name = Name,
        Kind = UnifiedMemory ? DeviceKind.IntegratedGpu : DeviceKind.DiscreteGpu,
        TotalMemory = WorkingSetBytes,
        Vendor = "Apple",
        Driver = Environment.OSVersion.Version.ToString(),
        Uuid = RegistryId.ToString("x16", CultureInfo.InvariantCulture),
        Features = Features(),
        ConcurrentWorkgroups = UnifiedMemory ? SmallDeviceWorkgroups : 0,
        Capabilities = new Dictionary<string, string>
        {
            ["apple_family"] = AppleFamily.ToString(CultureInfo.InvariantCulture),
            ["metal3"] = Metal3.ToString(),
            ["metal4"] = Metal4.ToString(),
            ["mac2"] = Mac2.ToString(),
            ["argument_buffers_tier"] = (ArgumentBuffersTier + 1).ToString(CultureInfo.InvariantCulture),
            ["unified_memory"] = UnifiedMemory.ToString(),
            ["max_shared_memory"] = MaxSharedMemory.ToString(CultureInfo.InvariantCulture),
            ["max_threads_per_threadgroup"] = MaxThreadsPerThreadgroup.ToString(CultureInfo.InvariantCulture),
            ["max_buffer_bytes"] = MaxBufferBytes.ToString(CultureInfo.InvariantCulture),
            ["subgroup_size"] = MetalCodeGenerator.SimdWidth.ToString(CultureInfo.InvariantCulture),
        },
    };
}
