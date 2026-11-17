using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Shaders;
using Sharplight.Tensors;
using Sharplight.Vulkan.Tests;

namespace Sharplight.Metal.Tests;

/// <summary>
/// The Metal provider's code generation, which needs no Mac: the shared kernels compile to MSL with their threadgroup size, and the
/// compile-only device of a Metal profile lowers a transformer block and compiles every kernel of it.
/// </summary>
public sealed class MetalCodeTests
{
    [Fact]
    public void LibraryKernelCompilesToMslWithItsThreadgroupSize()
    {
        var compiler = new SlangCompiler();
        var macros = new Dictionary<string, string>
        {
            ["D"] = "64", ["R1"] = "1", ["S0"] = "64", ["S1"] = "64", ["EPS"] = "1e-6", ["MODE"] = "0", ["HAS_W"] = "1", ["SG"] = "32", ["ROWS_PER_WG"] = "4",
        };
        var kernel = compiler.CompileLibrary("norm.slang", "main", ShaderTarget.Metal, macros);

        var source = MetalKernelSource.Parse(kernel);

        Assert.Equal(128u, source.Threads);
        Assert.Equal((128u, 1u, 1u), (source.ThreadsX, source.ThreadsY, source.ThreadsZ));
        Assert.Contains("[[kernel]]", source.Source);
        Assert.Contains("[[buffer(0)]]", source.Source);
        Assert.Contains("simd_sum", source.Source);
        Assert.StartsWith("main", source.Function);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ProfileDeviceCompilesEveryKernelOfATransformerBlock(bool simdgroupMatrix)
    {
        var provider = new MetalProvider();
        var profile = Profile(simdgroupMatrix);
        Assert.True(provider.CanCompileFor(profile));

        var portable = await provider.CompilePortableAsync(new PortableCompilationRequest(profile, Workload), TestContext.Current.CancellationToken);

        Assert.Equal("metal", provider.PortableCodeFormat);
        Assert.Equal("metal", portable.Format);
        Assert.Equal("metal", portable.Provider);
        Assert.NotEmpty(portable.Entries);
        Assert.All(portable.Entries.Keys, key => Assert.StartsWith(SlangCompiler.KernelKeyPrefix, key));
        var sources = portable.Entries.Values.Select(code => System.Text.Encoding.UTF8.GetString(code.Span)).ToList();
        Assert.All(sources, source => Assert.StartsWith(SlangCompiler.MetalThreadsHeader, source));
        // The prefill matmuls take the 8x8 subgroup-matrix GEMM exactly when the profile has simdgroup matrices.
        Assert.Equal(simdgroupMatrix, sources.Any(source => source.Contains("simdgroup_multiply_accumulate", StringComparison.Ordinal)));
    }

    [Fact]
    public void ProviderIsUnavailableOffMacOS()
    {
        Assert.SkipWhen(OperatingSystem.IsMacOS(), "Runs off macOS.");
        var provider = new MetalProvider();
        Assert.False(provider.CheckAvailability().IsAvailable);
        Assert.Empty(provider.GetDevices());
    }

    [Fact]
    public void ProviderListsAMetal3DeviceOnMacOS()
    {
        Assert.SkipUnless(OperatingSystem.IsMacOS(), "Needs macOS.");
        var provider = new MetalProvider();
        var availability = provider.CheckAvailability();
        Assert.True(availability.IsAvailable, availability.Reason);
        var info = Assert.Single(provider.GetDevices().Take(1));
        Assert.Equal("metal", info.Provider);
        Assert.Contains(DeviceFeatures.Float16, info.Features);
        Assert.Contains(DeviceFeatures.SubgroupOps, info.Features);
        using var device = provider.OpenDevice(info);
        Assert.True(device.GetMemoryInfo().Total > 0);
    }

    [Fact]
    public async Task DeviceStageArchiveServesThePipelinesOfLaterLoads()
    {
        Assert.SkipUnless(OperatingSystem.IsMacOS(), "Needs macOS.");
        var provider = new MetalProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "No usable Metal device.");
        var info = provider.GetDevices()[0];
        PortableCodeSet portable;
        DeviceCodeSet deviceCode;
        using (var device = provider.OpenDevice(info))
        {
            portable = await provider.CompilePortableAsync(new PortableCompilationRequest(device.CreateProfile(), Workload), TestContext.Current.CancellationToken);
            deviceCode = await provider.CompileDeviceSpecificAsync(device, portable, TestContext.Current.CancellationToken);
        }

        Assert.False(deviceCode.IsEmpty);
        Assert.Equal(ArtifactTiers.Binary, ArtifactDescriptor.Get(deviceCode.Descriptor, ArtifactDescriptorKeys.Tier));
        Assert.Equal([MetalDevice.BinaryArchiveKey], deviceCode.Entries.Keys);

        using var loaded = (MetalDevice)provider.OpenDevice(info);
        Workload(loaded, new CompilationArtifacts(new MemoryStore([portable.ToArtifactSet(), deviceCode.ToArtifactSet()])), TestContext.Current.CancellationToken);
        Assert.Equal(0, loaded.RejectedDeviceCode);
        Assert.True(loaded.PipelinesFromArchive > 0, $"{loaded.PipelinesFromArchive} of {loaded.PipelinesCreated} pipelines from the archive");
    }

    [Fact]
    public void FailingPipelineFallsBackToThePortableKernels()
    {
        Assert.SkipUnless(OperatingSystem.IsMacOS(), "Needs macOS.");
        var provider = new MetalProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "No usable Metal device.");
        var info = provider.GetDevices()[0];
        Assert.SkipUnless(info.Features.Contains(DeviceFeatures.MatrixF16Small), "The device has no simdgroup matrices (no gemm_sg).");

        var expected = RunBlock((MetalDevice)provider.OpenDevice(info), null, out _);
        // Every gemm_sg pipeline gets invalid MSL: Metal rejects it twice, and the segment is compiled again with the SIMT kernels.
        var actual = RunBlock((MetalDevice)provider.OpenDevice(info), kernel => kernel.EntryPoint == "gemm_sg", out int fallbacks);

        Assert.Equal(1, fallbacks);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-2 + 2e-2 * Math.Abs(expected[i]), $"y[{i}]: {actual[i]} vs {expected[i]}");
        }
    }

    /// <summary>Runs the MLP block on 16 rows and returns its softmax output and the device's pipeline fallbacks.</summary>
    private static float[] RunBlock(MetalDevice device, Func<CompiledKernel, bool>? invalidSource, out int fallbacks)
    {
        using (device)
        {
            device.InvalidSourceFor = invalidSource;
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
            fallbacks = device.PipelineFallbacks;
            return output;
        }
    }

    /// <summary>The profile of an Apple GPU as the provider reports it.</summary>
    private static DeviceProfile Profile(bool simdgroupMatrix) => new()
    {
        Provider = "metal",
        Name = "Apple GPU (profile)",
        Vendor = "Apple",
        Kind = DeviceKind.IntegratedGpu,
        CodeTarget = "metal",
        SubgroupSize = 32,
        MaxSharedMemory = 32768,
        Features = simdgroupMatrix ? [DeviceFeatures.Float16, DeviceFeatures.SubgroupOps, DeviceFeatures.MatrixF16Small, DeviceFeatures.IntDot] : [DeviceFeatures.Float16, DeviceFeatures.SubgroupOps],
        ConcurrentWorkgroups = 64,
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

    /// <summary>A 3x3 convolution of 64 to 128 channels on a 32 x 32 map, compiled with reduced precision (the implicit GEMM <c>conv_cm</c>).</summary>
    private static ComputeGraph BuildConvolution()
    {
        var tg = new TG("metal-conv");
        var x = tg.InF("x", Dim.Of(1), 64, 32, 32);
        var attributes = Attributes.From(("strides", new long[] { 1, 1 }), ("pads", new long[] { 1, 1, 1, 1 }), ("dilations", new long[] { 1, 1 }), ("groups", 1L));
        tg.B.Output("y", tg.B.Emit1("Conv", [x, tg.RandW("w", DataType.F16, 0.1f, 128, 64, 3, 3)], attributes));
        return tg.B.Build();
    }

    /// <summary>An expert-indexed matmul over 8 Q4_K experts, 2 slots per token (the grouped GEMM above the matvec rows).</summary>
    private static ComputeGraph BuildExperts()
    {
        var tg = new TG("metal-experts");
        var x = tg.InF("x", TG.T, 256);
        var ids = tg.In("ids", DataType.I32, TG.T, 2);
        tg.B.Output("y", tg.B.Emit1("MatMulId", [x, tg.RandW("experts", DataType.Q4_K, 0.3f, 8, 128, 256), ids]));
        return tg.B.Build();
    }

    /// <summary>
    /// Bidirectional attention of the token rows over 64 keys, compiled with reduced precision allowed (f16 products): the flash
    /// kernel <c>fa</c>, on 8x8 fragments with simdgroup matrices.
    /// </summary>
    private static ComputeGraph BuildAttention()
    {
        const int Heads = 4;
        const int HeadSize = 64;
        var tg = new TG("metal-attention");
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
        var tg = new TG("metal-block");
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
