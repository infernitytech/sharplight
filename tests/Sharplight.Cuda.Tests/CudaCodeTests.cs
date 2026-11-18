using System.Text;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Shaders;
using Sharplight.Tensors;
using Sharplight.Vulkan.Tests;

namespace Sharplight.Cuda.Tests;

/// <summary>
/// The CUDA provider's code generation, which needs NVRTC but no GPU: the shared kernels compile to CUDA C++ with their block size, NVRTC
/// turns them into PTX, and the compile-only device of a CUDA profile lowers a transformer block, attention, experts and a convolution
/// and compiles every kernel of them. These run on CI machines without a GPU (the runtime package's NVRTC only).
/// </summary>
public sealed class CudaCodeTests
{
    [Fact]
    public void LibraryKernelCompilesToCudaWithItsBlockSize()
    {
        var compiler = new SlangCompiler();
        var kernel = compiler.CompileLibrary("norm.slang", "main", ShaderTarget.CudaSource, NormMacros());

        string source = Encoding.UTF8.GetString(kernel.Code);

        Assert.StartsWith(SlangCompiler.CudaThreadsHeader + "128 1 1\n", source);
        Assert.Contains("__global__", source);
    }

    [Fact]
    public void LibraryKernelCompilesToPtxThroughNvrtc()
    {
        RequireNvrtc();
        var compiler = new SlangCompiler(new NvrtcTranslator(75));
        var kernel = compiler.CompileLibrary("norm.slang", "main", ShaderTarget.Ptx, NormMacros());

        var ptx = CudaPtx.Parse(kernel.Code, "norm");

        Assert.Equal((128u, 1u, 1u), (ptx.ThreadsX, ptx.ThreadsY, ptx.ThreadsZ));
        Assert.Contains(".target sm_75", Encoding.UTF8.GetString(ptx.Code));
        Assert.StartsWith("main", ptx.Function);
    }

    [Theory]
    [InlineData(75)]
    [InlineData(89)]
    [InlineData(120)]
    public async Task ProfileDeviceCompilesEveryKernelOfAWorkload(int computeCapability)
    {
        RequireNvrtc();
        var provider = new CudaProvider();
        var profile = Profile(computeCapability);
        Assert.True(provider.CanCompileFor(profile));

        var portable = await provider.CompilePortableAsync(new PortableCompilationRequest(profile, Workload), TestContext.Current.CancellationToken);

        Assert.Equal("ptx", provider.PortableCodeFormat);
        Assert.Equal("ptx", portable.Format);
        Assert.Equal("cuda", portable.Provider);
        Assert.NotEmpty(portable.Entries);
        Assert.All(portable.Entries.Keys, key => Assert.StartsWith(SlangCompiler.KernelKeyPrefix, key));
        Assert.All(portable.Entries.Values, code => Assert.Contains($".target sm_{computeCapability}", Encoding.UTF8.GetString(code.Span)));
    }

    [Fact]
    public void ProviderListsTheDriversDevices()
    {
        var provider = new CudaProvider();
        var availability = provider.CheckAvailability();
        Assert.SkipUnless(availability.IsAvailable, "No CUDA device: " + availability.Reason);

        var info = provider.GetDevices()[0];
        Assert.Equal("cuda", info.Provider);
        Assert.Equal("NVIDIA", info.Vendor);
        Assert.Contains(DeviceFeatures.Float16, info.Features);
        Assert.Contains(DeviceFeatures.SubgroupOps, info.Features);
        Assert.True(info.Capabilities.ContainsKey("compute_capability"));
        using var device = provider.OpenDevice(info);
        var memory = device.GetMemoryInfo();
        Assert.True(memory.Total > 0 && memory.IsExact);
    }

    [Fact]
    public async Task DeviceStageCubinsServeTheKernelsOfLaterLoads()
    {
        RequireNvrtc();
        var provider = new CudaProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "No CUDA device.");
        var info = provider.GetDevices()[0];
        var noStore = new DeviceOptions { UseCompilationCache = false };
        PortableCodeSet portable;
        DeviceCodeSet deviceCode;
        using (var device = provider.OpenDevice(info, noStore))
        {
            portable = await provider.CompilePortableAsync(new PortableCompilationRequest(device.CreateProfile()!, Workload), TestContext.Current.CancellationToken);
            deviceCode = await provider.CompileDeviceSpecificAsync(device, portable, TestContext.Current.CancellationToken);
        }

        Assert.False(deviceCode.IsEmpty);
        Assert.Equal(ArtifactTiers.Binary, ArtifactDescriptor.Get(deviceCode.Descriptor, ArtifactDescriptorKeys.Tier));
        Assert.Equal(portable.Entries.Count, deviceCode.Entries.Count);
        Assert.All(deviceCode.Entries.Keys, key => Assert.StartsWith(CudaDevice.CubinKeyPrefix, key));

        using var loaded = (CudaDevice)provider.OpenDevice(info, noStore);
        Workload(loaded, new CompilationArtifacts(new MemoryStore([portable.ToArtifactSet(), deviceCode.ToArtifactSet()])), TestContext.Current.CancellationToken);
        Assert.Equal(0, loaded.CubinsLinked);
        Assert.True(loaded.CubinsFromCache > 0);
    }

    [Fact]
    public void BlockMatchesTheCpu()
    {
        var provider = new CudaProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "No CUDA device.");
        var cpuProvider = new Cpu.CpuExecutionProvider();

        var expected = RunBlock(cpuProvider.OpenDevice(cpuProvider.GetDevices()[0]));
        var actual = RunBlock(provider.OpenDevice(provider.GetDevices()[0]));

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-2 + 2e-2 * Math.Abs(expected[i]), $"y[{i}]: {actual[i]} vs {expected[i]}");
        }
    }

    private static void RequireNvrtc() => Assert.SkipWhen(NvrtcTranslator.NvrtcVersion is null, "NVRTC is not installed (Sharplight.Cuda.Runtime).");

    private static Dictionary<string, string> NormMacros() => new()
    {
        ["D"] = "64", ["R1"] = "1", ["S0"] = "64", ["S1"] = "64", ["EPS"] = "1e-6", ["MODE"] = "0", ["HAS_W"] = "1", ["SG"] = "32", ["ROWS_PER_WG"] = "4",
    };

    /// <summary>Runs the MLP block on 16 rows and returns its softmax output.</summary>
    private static float[] RunBlock(IComputeDevice device)
    {
        using (device)
        {
            var graph = BuildBlock();
            using var model = CompiledModel.Compile(graph, new PlacementOptions
            {
                Devices = [new DeviceSlot { Device = device }],
                Compile = new CompileOptions { MaxTokens = 64, ContextLength = 256 },
                ReservePerDevice = 0,
            });
            const int Rows = 16;
            var input = new float[Rows * 256];
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = MathF.Sin(i * 0.37f);
            }

            model.SetInput(graph.Inputs.Single(), System.Runtime.InteropServices.MemoryMarshal.AsBytes(input.AsSpan()));
            var parameters = new RunParameters();
            parameters.Symbols["T"] = Rows;
            model.Run(parameters);
            var output = new float[Rows * 256];
            model.GetOutput<float>("y", output);
            return output;
        }
    }

    /// <summary>The profile of an NVIDIA GPU of the given compute capability as the provider reports it.</summary>
    private static DeviceProfile Profile(int computeCapability) => new()
    {
        Provider = "cuda",
        Name = $"NVIDIA GPU sm_{computeCapability} (profile)",
        Vendor = "NVIDIA",
        Kind = DeviceKind.DiscreteGpu,
        CodeTarget = CudaCodeGenerator.CodeTargetOf(computeCapability),
        SubgroupSize = 32,
        MaxSharedMemory = 48 * 1024,
        Features = [DeviceFeatures.Float16, DeviceFeatures.SubgroupOps, DeviceFeatures.IntDot],
    };

    private static void Workload(IComputeDevice device, CompilationArtifacts artifacts, CancellationToken cancellationToken)
    {
        foreach (var (graph, reduced) in new[] { (BuildBlock(), false), (BuildAttention(), true), (BuildExperts(), false), (BuildConvolution(), true) })
        {
            using var model = CompiledModel.Compile(graph, new PlacementOptions
            {
                Devices = [new DeviceSlot { Device = device }],
                Compile = new CompileOptions { MaxTokens = 64, ContextLength = 256, CacheFor = artifacts.For, AllowReducedPrecision = reduced },
                ReservePerDevice = 0,
            });
        }
    }

    /// <summary>A 3x3 convolution of 64 to 128 channels on a 32 x 32 map, compiled with reduced precision.</summary>
    private static ComputeGraph BuildConvolution()
    {
        var tg = new TG("cuda-conv");
        var x = tg.InF("x", Dim.Of(1), 64, 32, 32);
        var attributes = Attributes.From(("strides", new long[] { 1, 1 }), ("pads", new long[] { 1, 1, 1, 1 }), ("dilations", new long[] { 1, 1 }), ("groups", 1L));
        tg.B.Output("y", tg.B.Emit1("Conv", [x, tg.RandW("w", DataType.F16, 0.1f, 128, 64, 3, 3)], attributes));
        return tg.B.Build();
    }

    /// <summary>An expert-indexed matmul over 8 Q4_K experts, 2 slots per token.</summary>
    private static ComputeGraph BuildExperts()
    {
        var tg = new TG("cuda-experts");
        var x = tg.InF("x", TG.T, 256);
        var ids = tg.In("ids", DataType.I32, TG.T, 2);
        tg.B.Output("y", tg.B.Emit1("MatMulId", [x, tg.RandW("experts", DataType.Q4_K, 0.3f, 8, 128, 256), ids]));
        return tg.B.Build();
    }

    /// <summary>Bidirectional attention of the token rows over 64 keys, compiled with reduced precision allowed.</summary>
    private static ComputeGraph BuildAttention()
    {
        const int Heads = 4;
        const int HeadSize = 64;
        var tg = new TG("cuda-attention");
        var q = tg.InF("q", TG.T, Heads, HeadSize);
        var k = tg.InF("k", Dim.Of(64), Heads, HeadSize);
        var v = tg.InF("v", Dim.Of(64), Heads, HeadSize);
        tg.B.Output("o", tg.B.Emit1("Attention", [q, k, v], Attributes.From(("scale", 0.125f))));
        return tg.B.Build();
    }

    /// <summary>A pre-norm MLP block over f16, q8_0, q4_K and q6_K weights, with a softmax and a GELU.</summary>
    private static ComputeGraph BuildBlock()
    {
        const int Hidden = 256;
        const int Ffn = 512;
        var tg = new TG("cuda-block");
        var x = tg.InF("x", TG.T, Hidden);
        var norm = tg.B.Emit1("RmsNorm", [x, tg.RandW("norm", DataType.F32, 1f, Hidden)], Attributes.From(("eps", 1e-6f)));
        var gate = tg.B.MatMul(norm, tg.RandW("gate", DataType.Q4_K, 0.3f, Ffn, Hidden));
        var up = tg.B.MatMul(norm, tg.RandW("up", DataType.F16, 0.3f, Ffn, Hidden));
        var hidden = tg.B.Mul(tg.B.Silu(gate), up);
        var down = tg.B.MatMul(hidden, tg.RandW("down", DataType.Q6_K, 0.3f, Hidden, Ffn));
        var mixed = tg.B.MatMul(tg.B.Add(down, x), tg.RandW("out", DataType.Q8_0, 0.3f, Hidden, Hidden));
        tg.B.Output("y", tg.B.Emit1("Softmax", [mixed], Attributes.From(("scale", 1f))));
        tg.B.Output("g", tg.B.Unary("Gelu", mixed));
        return tg.B.Build();
    }
}
