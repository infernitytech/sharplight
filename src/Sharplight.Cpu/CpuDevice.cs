using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>The host CPU as a compute device.</summary>
internal sealed class CpuDevice : IComputeDevice
{
    private readonly DeviceOptions? _options;

    internal CpuDevice(DeviceInfo info, DeviceOptions? options)
    {
        Info = info;
        _options = options;
        EnabledFeatures = DeviceFeatures.Effective(info.Features, options?.DisabledFeatures);
        if (options?.Settings.TryGetValue("threads", out var threads) == true && int.TryParse(threads, out var count) && count > 0)
        {
            CpuParallel.Threads = count;
        }
    }

    public DeviceInfo Info { get; }

    public IReadOnlySet<string> EnabledFeatures { get; }

    public long AvailableMemory
    {
        get
        {
            long available = HostCpuInfo.MemInfo("MemAvailable") ?? Info.TotalMemory;
            return _options?.MemoryBudget is { } budget ? Math.Min(budget, available) : available;
        }
    }

    internal static DeviceInfo DescribeHost()
    {
        var capabilities = new Dictionary<string, string>
        {
            ["threads"] = Environment.ProcessorCount.ToString(),
            ["arch"] = RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant(),
            ["avx2"] = Flag(Avx2.IsSupported),
            ["avx512"] = Flag(Avx512F.IsSupported),
            ["avx512bw"] = Flag(Avx512BW.IsSupported),
            ["fma"] = Flag(Fma.IsSupported),
            ["f16c"] = Flag(Avx2.IsSupported),
            ["avx_vnni"] = Flag(AvxVnni.IsSupported),
            ["avx512_vnni"] = Flag(AvxVnni.V512.IsSupported),
            ["isa"] = CpuIsa.Name(CpuIsa.Level),
            ["vector_bits"] = Vector512.IsHardwareAccelerated ? "512" : Vector256.IsHardwareAccelerated ? "256" : "128",
        };
        long total = HostCpuInfo.MemInfo("MemTotal") ?? GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        return new DeviceInfo
        {
            Provider = CpuExecutionProvider.ProviderName,
            Index = 0,
            Name = HostCpuInfo.ModelName(),
            Kind = DeviceKind.Cpu,
            TotalMemory = total,
            Vendor = HostCpuInfo.Vendor(),
            Driver = RuntimeInformation.FrameworkDescription,
            Capabilities = capabilities,
            Features = DetectFeatures(),
        };
    }

    /// <summary>
    /// Provider-neutral features of the host: <see cref="DeviceFeatures.IntDot"/> when the kernels have packed int8 dot products
    /// (AVX2 <c>vpmaddubsw</c> or VNNI), which enables the quantized-activation matmul for quantized weights.
    /// </summary>
    private static HashSet<string> DetectFeatures()
    {
        var features = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (CpuIsa.Level >= CpuIsaLevel.Avx2)
        {
            features.Add(DeviceFeatures.IntDot);
        }

        return features;
    }

    private static string Flag(bool supported) => supported ? "true" : "false";

    public ICompiledSegment Compile(GraphSegment segment, CompileOptions options) => new CpuCompiledSegment(segment, options, this);

    public OpSupport Supports(Node node) => CpuOpSupport.Check(node);

    public void Dispose()
    {
    }
}
