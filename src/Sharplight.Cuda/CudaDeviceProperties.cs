using System.Globalization;
using System.Runtime.InteropServices;
using Sharplight.Cuda.Interop;
using Sharplight.Runtime;

namespace Sharplight.Cuda;

/// <summary>What one CUDA device reports through the driver API: identity, memory, limits and compute capability.</summary>
internal sealed class CudaDeviceProperties
{
    /// <summary>
    /// Shared memory the kernels are specialized for at most. The Slang kernels declare their shared arrays statically, and CUDA limits
    /// static shared memory to 48 KB per block (more needs dynamic shared memory and an opt-in per kernel, phase 2 of docs/cuda.md).
    /// </summary>

    /// <summary>Oldest compute capability the provider lists (Turing, the oldest the CUDA 13 compiler targets).</summary>
    public const int MinimumComputeCapability = 75;

    private CudaDeviceProperties()
    {
    }

    /// <summary>The driver's device ordinal (<c>CUdevice</c>).</summary>
    public int Ordinal { get; private init; }

    public string Name { get; private init; } = "";

    public Guid Uuid { get; private init; }

    public long TotalMemory { get; private init; }

    /// <summary>Compute capability as major * 10 + minor (e.g. 120 for 12.0).</summary>
    public int ComputeCapability { get; private init; }

    public int Multiprocessors { get; private init; }

    public int WarpSize { get; private init; }

    public long MaxSharedMemoryPerBlock { get; private init; }

    public long MaxSharedMemoryOptin { get; private init; }

    public long MaxSharedMemoryPerMultiprocessor { get; private init; }

    public int MaxThreadsPerBlock { get; private init; }

    public long L2CacheBytes { get; private init; }

    public bool Integrated { get; private init; }

    public bool ClusterLaunch { get; private init; }

    public string PciBusId { get; private init; } = "";

    /// <summary>The CUDA version the driver implements (e.g. 13020 for 13.2).</summary>
    public int DriverVersion { get; private init; }

    /// <summary>
    /// Shared memory per block the kernels are specialized for: the device's opt-in maximum (kernels above the 48 KB static limit
    /// run with dynamic shared memory, see <see cref="SharedMemoryRewriter"/>), or <c>SHARPLIGHT_CUDA_MAX_SHARED</c> bytes
    /// (49152 keeps the static limit, the shared memory of the Vulkan provider).
    /// </summary>
    public long KernelSharedMemory => Math.Max(MaxSharedMemoryPerBlock, Math.Min(DiagnosticSwitches.MaxSharedMemory ?? MaxSharedMemoryOptin, MaxSharedMemoryOptin));

    /// <summary>Why the device is not listed, or null when it is usable.</summary>
    public string? Unsupported => ComputeCapability < MinimumComputeCapability
        ? $"compute capability {ComputeCapability / 10}.{ComputeCapability % 10} is below {MinimumComputeCapability / 10}.{MinimumComputeCapability % 10}"
        : WarpSize != 32 ? $"warp size {WarpSize}" : null;

    /// <summary>The driver version as text (<c>13.2</c>).</summary>
    public string DriverText => $"{DriverVersion / 1000}.{DriverVersion % 1000 / 10}";

    /// <summary>Reads the properties of device <paramref name="ordinal"/>.</summary>
    public static unsafe CudaDeviceProperties Query(int ordinal, int driverVersion)
    {
        int handle;
        CudaApi.Check(CudaApi.cuDeviceGet(&handle, ordinal), "cuDeviceGet");
        int device = handle;
        var name = stackalloc byte[256];
        CudaApi.Check(CudaApi.cuDeviceGetName(name, 256, device), "cuDeviceGetName");
        nuint total;
        CudaApi.Check(CudaApi.cuDeviceTotalMem(&total, device), "cuDeviceTotalMem");
        var uuid = stackalloc byte[16];
        CudaApi.Check(CudaApi.cuDeviceGetUuid(uuid, device), "cuDeviceGetUuid");

        int Attribute(CuDeviceAttribute attribute)
        {
            int value;
            CudaApi.Check(CudaApi.cuDeviceGetAttribute(&value, attribute, device), $"cuDeviceGetAttribute({attribute})");
            return value;
        }

        return new CudaDeviceProperties
        {
            Ordinal = device,
            Name = Marshal.PtrToStringUTF8((nint)name) ?? "",
            // The UUID bytes in the order nvidia-smi and Vulkan's deviceUUID print them.
            Uuid = new Guid(new ReadOnlySpan<byte>(uuid, 16), bigEndian: true),
            TotalMemory = (long)total,
            ComputeCapability = Attribute(CuDeviceAttribute.ComputeCapabilityMajor) * 10 + Attribute(CuDeviceAttribute.ComputeCapabilityMinor),
            Multiprocessors = Attribute(CuDeviceAttribute.MultiprocessorCount),
            WarpSize = Attribute(CuDeviceAttribute.WarpSize),
            MaxSharedMemoryPerBlock = Attribute(CuDeviceAttribute.MaxSharedMemoryPerBlock),
            MaxSharedMemoryOptin = Attribute(CuDeviceAttribute.MaxSharedMemoryPerBlockOptin),
            MaxSharedMemoryPerMultiprocessor = Attribute(CuDeviceAttribute.MaxSharedMemoryPerMultiprocessor),
            MaxThreadsPerBlock = Attribute(CuDeviceAttribute.MaxThreadsPerBlock),
            L2CacheBytes = Attribute(CuDeviceAttribute.L2CacheSize),
            Integrated = Attribute(CuDeviceAttribute.Integrated) != 0,
            ClusterLaunch = Attribute(CuDeviceAttribute.ClusterLaunch) != 0,
            PciBusId = string.Create(CultureInfo.InvariantCulture,
                $"{Attribute(CuDeviceAttribute.PciDomainId):x4}:{Attribute(CuDeviceAttribute.PciBusId):x2}:{Attribute(CuDeviceAttribute.PciDeviceId):x2}"),
            DriverVersion = driverVersion,
        };
    }

    /// <summary>Device features with the neutral names of <see cref="DeviceFeatures"/>, see <see cref="FeaturesFor"/>.</summary>
    /// <remarks>
    /// Adds <see cref="DeviceFeatures.LibraryGemm"/> from compute capability 8.0 when cuBLASLt (package <c>Sharplight.Cuda.Blas</c>) is
    /// installed.
    /// </remarks>
    public IReadOnlySet<string> Features()
    {
        var features = new HashSet<string>(FeaturesFor(ComputeCapability));
        if (ComputeCapability >= 80 && CudaBlas.Available)
        {
            features.Add(DeviceFeatures.LibraryGemm);
        }

        return features;
    }

    /// <summary>
    /// Device features of a compute capability. Every listed GPU (7.5 and later) has half-precision arithmetic, warp collectives, the
    /// packed int8 dot product (<c>dp4a</c>) and a fixed warp size of 32 (<see cref="DeviceFeatures.SubgroupSizeControl"/>). From 8.0
    /// (Ampere) the tensor cores run the cooperative-matrix kernels through their CUDA forms (<c>cuda_mma.slang</c>: <c>ldmatrix</c> and
    /// <c>mma.sync</c> on f16, bf16 and int8 with the MMA register layout, <see cref="DeviceFeatures.MatrixMmaLayout"/>), and bfloat16
    /// is native; from 8.9 (Ada, Hopper, Blackwell) also E4M3 (<c>mma.sync ... e4m3</c>), the 8-bit float types and the hand-written
    /// FP8 GEMM (<see cref="DeviceFeatures.NativeGemmFp8"/>); from 9.0 (Hopper, Blackwell) tensor copies
    /// (<see cref="DeviceFeatures.TensorCopies"/>). Turing (7.5) has
    /// <c>mma.sync</c> only in smaller shapes and keeps the SIMT kernels.
    /// </summary>
    public static IReadOnlySet<string> FeaturesFor(int computeCapability)
    {
        var features = new HashSet<string> { DeviceFeatures.Float16, DeviceFeatures.SubgroupOps, DeviceFeatures.IntDot, DeviceFeatures.SubgroupSizeControl };
        if (computeCapability >= 80)
        {
            features.UnionWith([DeviceFeatures.MatrixF16, DeviceFeatures.MatrixBF16, DeviceFeatures.MatrixInt8, DeviceFeatures.MatrixMmaLayout, DeviceFeatures.BFloat16,
                DeviceFeatures.NativeGemmF16, DeviceFeatures.NativeGemmMoe]);
        }

        if (computeCapability >= 89)
        {
            // The hand-written FP8 GEMM (gemm_fp8_native.cu): 99 KB of dynamic shared memory, which every GPU from 8.9 has.
            features.UnionWith([DeviceFeatures.MatrixFp8, DeviceFeatures.Float8, DeviceFeatures.NativeGemmFp8]);
        }

        if (computeCapability >= 90)
        {
            // Tensor copies (cp.async.bulk.tensor with mbarriers): the warp-specialized FP8 GEMM (gemm_fp8_tma.cu).
            features.Add(DeviceFeatures.TensorCopies);
        }

        return features;
    }

    public DeviceInfo ToDeviceInfo(int index) => new()
    {
        Provider = CudaProvider.ProviderName,
        Index = index,
        Name = Name,
        Kind = Integrated ? DeviceKind.IntegratedGpu : DeviceKind.DiscreteGpu,
        TotalMemory = TotalMemory,
        Vendor = "NVIDIA",
        Driver = DriverText,
        Uuid = Uuid.ToString(),
        Features = Features(),
        SubmissionBudgetMs = Integrated ? 250 : 0,
        ConcurrentWorkgroups = 0,
        Capabilities = new Dictionary<string, string>
        {
            ["compute_capability"] = $"{ComputeCapability / 10}.{ComputeCapability % 10}",
            ["compute_units"] = Multiprocessors.ToString(CultureInfo.InvariantCulture),
            ["subgroup_size"] = WarpSize.ToString(CultureInfo.InvariantCulture),
            ["max_shared_memory"] = KernelSharedMemory.ToString(CultureInfo.InvariantCulture),
            ["max_shared_memory_optin"] = MaxSharedMemoryOptin.ToString(CultureInfo.InvariantCulture),
            ["max_shared_memory_per_sm"] = MaxSharedMemoryPerMultiprocessor.ToString(CultureInfo.InvariantCulture),
            ["max_threads_per_block"] = MaxThreadsPerBlock.ToString(CultureInfo.InvariantCulture),
            ["l2_cache_bytes"] = L2CacheBytes.ToString(CultureInfo.InvariantCulture),
            ["cluster_launch"] = ClusterLaunch.ToString(),
            ["pci_bus_id"] = PciBusId,
            ["cuda_driver"] = DriverText,
            ["cublaslt"] = CudaBlas.Version == 0 ? "none" : $"{CudaBlas.Version / 10000}.{CudaBlas.Version / 100 % 100}.{CudaBlas.Version % 100}",
        },
    };
}
