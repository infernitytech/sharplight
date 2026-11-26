using System.Text;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Sharplight.Vulkan.Tests;

namespace Sharplight.Cuda.Tests;

/// <summary>
/// The kernels that quantize activations to a low-precision float format (the FP4 GEMM's E2M1 rows, the FP8 GEMMs' E4M3 rows) serve
/// only weights whose format quantizes activations (<see cref="WeightInfo.Activations"/>): a weight-only checkpoint (ModelOpt
/// <c>W4A16_NVFP4</c>, an FP8 scheme without <c>input_activations</c>) keeps full-precision activations, whatever the device offers.
/// Checked on the code a Blackwell profile with the block-scaled MMAs compiles for a dense projection at prompt and decode sizes.
/// </summary>
public sealed class WeightOnlyActivationTests
{
    private const int Hidden = 1024;

    [Theory]
    [InlineData(ActivationScheme.Unspecified, false, true)]
    [InlineData(ActivationScheme.Quantized, false, true)]
    [InlineData(ActivationScheme.Unquantized, false, false)]
    [InlineData(ActivationScheme.Unquantized, true, true)]
    public async Task Nvfp4ActivationsAreQuantizedOnlyWhereTheFormatSaysSo(ActivationScheme scheme, bool reducedPrecision, bool expected)
        => Assert.Equal(expected, await CompilesEntry(DataType.NVFP4_GS, scheme, reducedPrecision, "quantize_fp4"));

    [Theory]
    [InlineData(ActivationScheme.Unspecified, false, true)]
    [InlineData(ActivationScheme.Quantized, false, true)]
    [InlineData(ActivationScheme.Unquantized, false, false)]
    [InlineData(ActivationScheme.Unquantized, true, true)]
    public async Task Fp8ActivationsAreQuantizedOnlyWhereTheFormatSaysSo(ActivationScheme scheme, bool reducedPrecision, bool expected)
        => Assert.Equal(expected, await CompilesEntry(DataType.F8E4M3_B128, scheme, reducedPrecision, "quantize_fp8"));

    /// <summary>Whether the PTX compiled for x + W(RmsNorm(x)) with <paramref name="type"/> weights of <paramref name="scheme"/> has a kernel named <paramref name="entry"/>.</summary>
    private static async Task<bool> CompilesEntry(DataType type, ActivationScheme scheme, bool reducedPrecision, string entry)
    {
        Assert.SkipWhen(NvrtcTranslator.NvrtcVersion is null, "NVRTC is not installed (Sharplight.Cuda.Runtime).");
        var provider = new CudaProvider();
        var profile = CudaCodeTests.Profile(120, 99 * 1024, DeviceFeatures.BlockScaledFp8, DeviceFeatures.BlockScaledFp4);
        var graph = Projection(type, scheme);
        var portable = await provider.CompilePortableAsync(new PortableCompilationRequest(profile, (device, artifacts, cancellationToken) =>
        {
            using var model = CompiledModel.Compile(graph, new PlacementOptions
            {
                Devices = [new DeviceSlot { Device = device }],
                Compile = new CompileOptions { MaxTokens = 256, ContextLength = 256, CacheFor = artifacts.For, AllowReducedPrecision = reducedPrecision },
                ReservePerDevice = 0,
            });
        }) { DeviceOptions = new DeviceOptions { UseCompilationCache = false } }, TestContext.Current.CancellationToken);

        Assert.NotEmpty(portable.Entries);
        return portable.Entries.Values.Any(code => Encoding.UTF8.GetString(code.Span).Contains($".entry {entry}", StringComparison.Ordinal));
    }

    private static ComputeGraph Projection(DataType type, ActivationScheme scheme)
    {
        var tg = new TG($"weight-only-{type}-{scheme}");
        var x = tg.InF("x", TG.T, Hidden);
        const string name = "w";
        tg.Weights[name] = new WeightInfo
        {
            Name = name, Type = type, Shape = new Shape(Hidden, Hidden), Source = new MemSource(Rnd.WeightBytes(type, (long)Hidden * Hidden, 7, 0.05f)),
            Activations = scheme,
        };
        var y = tg.B.MatMul(tg.B.RmsNorm(x, null, 1e-5f), tg.B.Weight(name));
        tg.B.Output("y", tg.B.Add(x, y));
        return tg.B.Build();
    }
}
