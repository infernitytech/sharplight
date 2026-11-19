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
        Assert.Equal(profile.CodeTarget, portable.Format);
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

    [Fact]
    public void LibraryGemmMatchesTheCpu()
    {
        var provider = new CudaProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "No CUDA device.");
        var info = provider.GetDevices()[0];
        Assert.SkipUnless(info.Features.Contains(DeviceFeatures.LibraryGemm), "cuBLASLt is not installed (Sharplight.Cuda.Blas or SHARPLIGHT_CUDA_BLAS_DIR).");
        var cpuProvider = new Cpu.CpuExecutionProvider();

        // 300 rows (library from 256) of a bf16 projection (stored as f16 for the library); x is rounded to f16 as for the generated GEMMs.
        // The hand-written f16 GEMM is preferred over the library where both exist: switch it off to reach the library.
        var libraryOnly = new DeviceOptions();
        libraryOnly.DisabledFeatures.Add(DeviceFeatures.NativeGemmF16);
        var expected = RunProjection(cpuProvider.OpenDevice(cpuProvider.GetDevices()[0]), 300);
        var actual = RunProjection(provider.OpenDevice(info, libraryOnly), 300);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 3e-2 + 1e-2 * Math.Abs(expected[i]), $"y[{i}]: {actual[i]} vs {expected[i]}");
        }

        // The library launch ran: the same rows without the feature give the generated GEMM's results, which sum in another order.
        var withoutLibrary = new DeviceOptions();
        withoutLibrary.DisabledFeatures.Add(DeviceFeatures.LibraryGemm);
        withoutLibrary.DisabledFeatures.Add(DeviceFeatures.NativeGemmF16);
        var kernels = RunProjection(provider.OpenDevice(info, withoutLibrary), 300);
        Assert.NotEqual(kernels, actual);
    }

    [Theory]
    [InlineData(300, false)]
    [InlineData(700, false)]
    [InlineData(300, true)]
    [InlineData(700, true)]
    public void NativeF16GemmMatchesTheCpu(int rows, bool swiglu)
    {
        var provider = new CudaProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "No CUDA device.");
        var info = provider.GetDevices()[0];
        Assert.SkipUnless(info.Features.Contains(DeviceFeatures.NativeGemmF16), "The GPU has no hand-written f16 GEMM (compute capability below 8.0).");
        var cpuProvider = new Cpu.CpuExecutionProvider();

        // Rows from 256 run the hand-written GEMM (single, or dual with the SiLU·mul epilogue) on the f16 copies of x and W.
        var expected = RunProjection(cpuProvider.OpenDevice(cpuProvider.GetDevices()[0]), rows, swiglu);
        var actual = RunProjection(provider.OpenDevice(info), rows, swiglu);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 3e-2 + 1e-2 * Math.Abs(expected[i]), $"y[{i}]: {actual[i]} vs {expected[i]}");
        }

        // The hand-written launch ran: without it (and without the library) the generated GEMMs sum in another order.
        var generated = new DeviceOptions();
        generated.DisabledFeatures.Add(DeviceFeatures.NativeGemmF16);
        generated.DisabledFeatures.Add(DeviceFeatures.LibraryGemm);
        Assert.NotEqual(RunProjection(provider.OpenDevice(info, generated), rows, swiglu), actual);
    }

    [Theory]
    [InlineData(100, false)]
    [InlineData(300, false)]
    [InlineData(100, true)]
    [InlineData(700, true)]
    public void NativeFp8GemmMatchesTheCpu(int rows, bool swiglu)
    {
        var provider = new CudaProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "No CUDA device.");
        var info = provider.GetDevices()[0];
        Assert.SkipUnless(info.Features.Contains(DeviceFeatures.NativeGemmFp8), "The GPU has no hand-written FP8 GEMM (compute capability below 8.9).");
        var cpuProvider = new Cpu.CpuExecutionProvider();

        // From 65 rows the hand-written GEMM serves (64-row tiles at 100 rows, 128-row tiles at 700 for these shapes), reading the
        // 128-row weight groups and the dense scale copy of the E4M3 activations. The activations are values that the per-128 E4M3
        // quantization represents exactly, so the CPU's f32 product differs by the summation order only.
        var expected = RunProjection(cpuProvider.OpenDevice(cpuProvider.GetDevices()[0]), rows, swiglu, DataType.F8E4M3_B128);
        var actual = RunProjection(provider.OpenDevice(info), rows, swiglu, DataType.F8E4M3_B128);
        double scale = expected.Max(MathF.Abs);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-3 * scale, $"y[{i}]: {actual[i]} vs {expected[i]}");
        }

        // The generated fp8 GEMM on the same layout (feature off: the split rows of the phase-2 kernels) gives the same products
        // within the accumulation order.
        var generated = new DeviceOptions();
        generated.DisabledFeatures.Add(DeviceFeatures.NativeGemmFp8);
        var reference = RunProjection(provider.OpenDevice(info, generated), rows, swiglu, DataType.F8E4M3_B128);
        for (int i = 0; i < reference.Length; i++)
        {
            Assert.True(Math.Abs(reference[i] - actual[i]) <= 1e-4 * scale, $"y[{i}]: {actual[i]} vs the generated GEMM's {reference[i]}");
        }
    }

    [Theory]
    [InlineData(150)]
    [InlineData(700)]
    public void NativeMoeGemmMatchesTheCpu(int tokens)
    {
        var provider = new CudaProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "No CUDA device.");
        var info = provider.GetDevices()[0];
        Assert.SkipUnless(info.Features.Contains(DeviceFeatures.NativeGemmMoe), "The GPU has no hand-written grouped MXFP4 GEMM (compute capability below 8.0).");
        var cpuProvider = new Cpu.CpuExecutionProvider();

        // A gpt-oss-style expert block on MXFP4 experts: 75 / 350 routed rows per expert (one to three 128-row tiles, the last
        // partial), the fused gate / up halves with bias rows and SiLU·mul in the dual store (192 outputs: a partial column tile),
        // the down product with bias and router weight in its store, K = 576 (nine 64-value steps).
        var expected = RunExpertBlock(cpuProvider.OpenDevice(cpuProvider.GetDevices()[0]), tokens);
        var actual = RunExpertBlock(provider.OpenDevice(info), tokens);
        double scale = expected.Max(MathF.Abs);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-2 * scale, $"y[{i}]: {actual[i]} vs {expected[i]}");
        }

        // The generated grouped GEMM decodes the same halves and sums the same 16-value MMAs in the same order: the same outputs.
        var generated = new DeviceOptions();
        generated.DisabledFeatures.Add(DeviceFeatures.NativeGemmMoe);
        var reference = RunExpertBlock(provider.OpenDevice(info, generated), tokens);
        for (int i = 0; i < reference.Length; i++)
        {
            Assert.True(Math.Abs(reference[i] - actual[i]) <= 1e-5 * scale, $"y[{i}]: {actual[i]} vs the generated GEMM's {reference[i]}");
        }
    }

    /// <summary>
    /// The routed-expert block of <see cref="NativeMoeGemmMatchesTheCpu"/> on <paramref name="tokens"/> tokens (8 MXFP4 experts, 4 slots,
    /// hidden size 576, expert size 192): silu(gate + b) · (up + b) of the fused gate / up product, then the down product plus its
    /// bias times the router weight, summed over the slots.
    /// </summary>
    private static float[] RunExpertBlock(IComputeDevice device, int tokens)
    {
        const int E = 8, S = 4, K = 576, H = 192;
        using (device)
        {
            var tg = new TG("cuda-native-moe");
            var x = tg.InF("x", TG.T, K);
            var ids = tg.In("ids", DataType.I32, TG.T, S);
            var weights = tg.InF("weights", TG.T, S, 1);
            Value Bias(string name, int n) => tg.B.Emit1("GatherAxis", [tg.RandW(name, DataType.F32, 0.5f, E, n), ids], Attributes.From(("axis", 0L)));
            var gateUp = tg.B.Add(tg.B.Emit1("MatMulId", [x, tg.RandW("gate_up", DataType.MXFP4, 0.1f, E, 2 * H, K), ids]), Bias("gate_up_b", 2 * H));
            var act = tg.B.Mul(tg.B.Silu(tg.B.Slice(gateUp, 2, 0, H)), tg.B.Slice(gateUp, 2, H, H));
            var down = tg.B.Add(tg.B.Emit1("MatMulId", [act, tg.RandW("down", DataType.MXFP4, 0.1f, E, K, H), ids]), Bias("down_b", K));
            tg.B.Output("y", tg.B.Emit1("ReduceSum", [tg.B.Permute(tg.B.Mul(down, weights), 0, 2, 1)]));
            var graph = tg.B.Build();
            using var model = CompiledModel.Compile(graph, new PlacementOptions
            {
                Devices = [new DeviceSlot { Device = device }],
                Compile = new CompileOptions { MaxTokens = 1024, ContextLength = 1024 },
                ReservePerDevice = 0,
            });
            var input = new float[tokens * K];
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = MathF.Sin(i * 0.37f);
            }

            // Distinct experts per token, skewed toward the low experts so that some see several tiles.
            var routes = new int[tokens * S];
            var gates = new float[tokens * S];
            var rng = new Random(tokens);
            for (int t = 0; t < tokens; t++)
            {
                var chosen = Enumerable.Range(0, E).OrderBy(e => rng.NextDouble() * (1 + e * 0.3)).Take(S).ToArray();
                for (int s = 0; s < S; s++)
                {
                    routes[t * S + s] = chosen[s];
                    gates[t * S + s] = 0.1f + 0.2f * s;
                }
            }

            model.SetInput<float>(x, input);
            model.SetInput<int>(ids, routes);
            model.SetInput<float>(weights, gates);
            var parameters = new RunParameters();
            parameters.Symbols["T"] = tokens;
            model.Run(parameters);
            var output = new float[tokens * K];
            model.GetOutput<float>("y", output);
            return output;
        }
    }

    /// <summary>
    /// y = x · Wᵀ with W [512, 1024] (bf16, or <paramref name="weightType"/>) on <paramref name="rows"/> rows, or with
    /// <paramref name="swiglu"/> the gated pair silu(x · Wgᵀ) · (x · Wuᵀ).
    /// </summary>
    private static float[] RunProjection(IComputeDevice device, int rows, bool swiglu = false, DataType? weightType = null)
    {
        using (device)
        {
            var tg = new TG("cuda-library-gemm");
            var x = tg.InF("x", TG.T, 1024);
            var y = tg.B.MatMul(x, tg.RandW("w", weightType ?? DataType.BF16, 0.05f, 512, 1024));
            if (swiglu)
            {
                y = tg.B.Mul(tg.B.Silu(y), tg.B.MatMul(x, tg.RandW("w2", weightType ?? DataType.BF16, 0.05f, 512, 1024)));
            }

            tg.B.Output("y", y);
            var graph = tg.B.Build();
            using var model = CompiledModel.Compile(graph, new PlacementOptions
            {
                Devices = [new DeviceSlot { Device = device }],
                Compile = new CompileOptions { MaxTokens = 1024, ContextLength = 1024 },
                ReservePerDevice = 0,
            });
            var input = new float[rows * 1024];
            for (int i = 0; i < input.Length; i++)
            {
                // FP8 weights: -1, -0.5, 0, 0.5, 1, a 1 in every 128 values, which E4M3 with a per-128 scale represents exactly.
                input[i] = weightType is null ? MathF.Sin(i * 0.37f) : (i % 5 - 2) * 0.5f;
            }

            model.SetInput(graph.Inputs.Single(), System.Runtime.InteropServices.MemoryMarshal.AsBytes(input.AsSpan()));
            var parameters = new RunParameters();
            parameters.Symbols["T"] = rows;
            model.Run(parameters);
            var output = new float[rows * 512];
            model.GetOutput<float>("y", output);
            return output;
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
        Features = [.. CudaDeviceProperties.FeaturesFor(computeCapability)],
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
