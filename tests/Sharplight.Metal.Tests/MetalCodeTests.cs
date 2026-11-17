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

    [Fact]
    public async Task ProfileDeviceCompilesEveryKernelOfATransformerBlock()
    {
        var provider = new MetalProvider();
        var profile = Profile();
        Assert.True(provider.CanCompileFor(profile));

        var portable = await provider.CompilePortableAsync(new PortableCompilationRequest(profile, Workload), TestContext.Current.CancellationToken);

        Assert.Equal("metal", provider.PortableCodeFormat);
        Assert.Equal("metal", portable.Format);
        Assert.Equal("metal", portable.Provider);
        Assert.NotEmpty(portable.Entries);
        Assert.All(portable.Entries.Keys, key => Assert.StartsWith(SlangCompiler.KernelKeyPrefix, key));
        Assert.All(portable.Entries.Values, code => Assert.StartsWith(SlangCompiler.MetalThreadsHeader, System.Text.Encoding.UTF8.GetString(code.Span)));
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

    /// <summary>The profile of an Apple GPU as the provider reports it.</summary>
    private static DeviceProfile Profile() => new()
    {
        Provider = "metal",
        Name = "Apple GPU (profile)",
        Vendor = "Apple",
        Kind = DeviceKind.IntegratedGpu,
        CodeTarget = "metal",
        SubgroupSize = 32,
        MaxSharedMemory = 32768,
        Features = [DeviceFeatures.Float16, DeviceFeatures.SubgroupOps],
        ConcurrentWorkgroups = 64,
    };

    private static void Workload(IComputeDevice device, CompilationArtifacts artifacts, CancellationToken cancellationToken)
    {
        using var model = CompiledModel.Compile(BuildBlock(), new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            Compile = new CompileOptions { MaxTokens = 64, ContextLength = 256, CacheFor = artifacts.For },
            ReservePerDevice = 0,
        });
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
