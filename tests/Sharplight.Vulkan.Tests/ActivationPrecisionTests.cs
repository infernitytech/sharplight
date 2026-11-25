using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// The activation-range probe and the bf16 activation choice (<see cref="ActivationRangeProbe"/>, <see cref="ActivationPrecision"/>):
/// a small convolution stack whose middle activations reach ~1e5 (beyond f16's 65504, as in SDXL's VAE decoder) is flagged and, with
/// <see cref="ActivationPrecision.BFloat16"/>, matches the f32 CPU oracle under reduced precision; the same stack at normal scale is
/// not flagged.
/// </summary>
public sealed class ActivationPrecisionTests
{
    private const int Channels = 64;
    private const int Size = 16;

    /// <summary>
    /// x -> conv -> × <paramref name="gain"/> -> conv -> × 1 / <paramref name="gain"/>: the second convolution reads activations of
    /// magnitude ~<paramref name="gain"/>; the output stays of order one.
    /// </summary>
    private static (ComputeGraph Graph, Value X) Stack(float gain)
    {
        var tg = new TG();
        var x = tg.InF("x", 1, Channels, Size, Size);
        var wide = tg.B.Scale(Conv(tg, x), gain, name: "wide");
        tg.B.Output("y", tg.B.Scale(Conv(tg, wide), 1 / gain));
        return (tg.B.Build(), x);
    }

    private static Value Conv(TG tg, Value x)
    {
        var w = tg.RandW("w" + tg.Weights.Count, DataType.F32, 1f / MathF.Sqrt(Channels * 9), Channels, Channels, 3, 3);
        var attrs = Attributes.From(("strides", new long[] { 1, 1 }), ("pads", new long[] { 1, 1, 1, 1 }), ("dilations", new long[] { 1, 1 }), ("groups", 1L));
        return tg.B.Emit1("Conv", [x, w, tg.RandW("b" + tg.Weights.Count, DataType.F32, 0.1f, Channels)], attrs);
    }

    private static CompiledModel Compile(IComputeDevice device, ComputeGraph graph, bool reduced, ActivationPrecision precision)
        => CompiledModel.Compile(graph, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            Compile = new CompileOptions { MaxTokens = 1, AllowReducedPrecision = reduced, ActivationPrecision = precision },
            ReservePerDevice = 0,
        });

    private static float[] Run(IComputeDevice device, ComputeGraph graph, Value x, float[] input, bool reduced, ActivationPrecision precision)
    {
        using var compiled = Compile(device, graph, reduced, precision);
        compiled.SetInput<float>(x, input);
        compiled.Run(new RunParameters());
        var y = new float[Channels * Size * Size];
        compiled.GetOutput<float>("y", y);
        return y;
    }

    private static ActivationRange Probe(ComputeGraph graph, Value x, float[] input)
    {
        using var compiled = Compile(Gpu.Device!, ActivationRangeProbe.Instrument(graph), reduced: false, ActivationPrecision.Half);
        compiled.SetInput<float>(x, input);
        var parameters = new RunParameters();
        compiled.Run(parameters);
        return ActivationRangeProbe.Read(compiled, parameters);
    }

    [Fact]
    public void OverflowingActivationsAreFlagged()
    {
        Gpu.Require();
        var (graph, x) = Stack(2e5f);
        var range = Probe(graph, x, Rnd.F(Channels * Size * Size, 31));
        Assert.False(range.FitsHalf, $"max |x| {range.MaxMagnitude}");
        Assert.Equal(ActivationPrecision.BFloat16, range.Precision);
        Assert.Equal("wide", range.Location);
    }

    [Fact]
    public void NormalActivationsAreNotFlagged()
    {
        Gpu.Require();
        var (graph, x) = Stack(4f);
        var range = Probe(graph, x, Rnd.F(Channels * Size * Size, 31));
        Assert.True(range.FitsHalf, $"max |x| {range.MaxMagnitude} at {range.Location}");
        Assert.Equal(ActivationPrecision.Half, range.Precision);
        Assert.InRange(range.MaxMagnitude, 1f, ActivationRange.HalfSafeLimit);
    }

    /// <summary>
    /// The flagged stack with bf16 activations against the f32 CPU oracle at the bf16 tolerance; with f16 activations (where the device
    /// computes them) its output is not finite or far off.
    /// </summary>
    [Fact]
    public void BFloat16ActivationsMatchF32()
    {
        Gpu.Require();
        var (graph, x) = Stack(2e5f);
        var input = Rnd.F(Channels * Size * Size, 33);
        var reference = Run(Gpu.Cpu, graph, x, input, reduced: false, ActivationPrecision.Half);
        var wide = Run(Gpu.Device!, graph, x, input, reduced: true, ActivationPrecision.BFloat16);
        float scale = reference.Max(MathF.Abs);
        for (int i = 0; i < reference.Length; i++)
        {
            Assert.True(MathF.Abs(wide[i] - reference[i]) <= 4e-2f * scale + 4e-2f * MathF.Abs(reference[i]),
                $"y[{i}] = {wide[i]}, f32 {reference[i]} (output scale {scale})");
        }

        if (Gpu.Device!.Info.Features.Contains(DeviceFeatures.MatrixF16))
        {
            var half = Run(Gpu.Device!, graph, x, input, reduced: true, ActivationPrecision.Half);
            double worst = half.Zip(reference, (h, r) => float.IsFinite(h) ? Math.Abs(h - r) : double.PositiveInfinity).Max();
            Assert.True(worst > 0.5 * scale, $"f16 activations unexpectedly fit: worst error {worst}, output scale {scale}");
        }
    }
}
