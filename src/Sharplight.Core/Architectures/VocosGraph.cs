using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The Vocos mel vocoder (charactr's <c>vocos-mel-24khz</c>) as a graph over the mel frames <c>[T, mels]</c>: the input convolution and LayerNorm,
/// ConvNeXt blocks (depthwise convolution, LayerNorm, pointwise expansion with GELU, projection, layer scale, residual), the final LayerNorm, and the
/// ISTFT head (a linear layer to log-magnitude and phase, the magnitude clipped at 100) up to the windowed inverse-DFT frames <c>[T, n_fft]</c> that the
/// host overlap-adds (<see cref="OverlapAdd"/>, centered). Weights under the given prefix: <c>backbone.embed</c>, <c>backbone.norm</c>,
/// <c>backbone.convnext.{i}.{dwconv,norm,pwconv1,pwconv2,gamma}</c>, <c>backbone.final_layer_norm</c>, <c>head.out</c>.
/// </summary>
internal static class VocosGraph
{
    /// <summary>The graph name.</summary>
    public const string GraphName = "vocos";

    /// <summary>The mel frames [<see cref="WellKnown.TokenSymbol"/>, mels].</summary>
    public const string MelInput = "mel";

    /// <summary>The windowed inverse-DFT frames [T, n_fft].</summary>
    public const string FramesOutput = "frames";

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Looks up the weights.</param>
    /// <param name="prefix">The weights' prefix (e.g. <c>vocoder.</c>).</param>
    /// <param name="mels">Mel bins.</param>
    /// <param name="layers">ConvNeXt blocks.</param>
    /// <param name="nFft">The STFT size.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, string prefix, int mels, int layers, int nFft)
    {
        var b = new GraphBuilder(GraphName, weights);
        var mel = b.Input(MelInput, InputRole.Other, DataType.F32, new Shape(Dim.Sym(WellKnown.TokenSymbol), mels));
        string p = prefix + "backbone.";
        var x = b.Emit1("Conv1d", [mel, b.Weight(p + "embed.weight"), b.Weight(p + "embed.bias")], Attributes.From(("pad_left", 3L), ("pad_right", 3L)));
        x = Norm(b, x, p + "norm");
        long dim = x.Shape[1].Fixed;
        for (int i = 0; i < layers; i++)
        {
            string m = $"{p}convnext.{i}.";
            using var _ = b.Group(m.TrimEnd('.'));
            var y = b.Emit1("Conv1d", [x, b.Weight(m + "dwconv.weight"), b.Weight(m + "dwconv.bias")],
                Attributes.From(("pad_left", 3L), ("pad_right", 3L), ("groups", dim)));
            y = Norm(b, y, m + "norm");
            y = b.Linear(b.Unary("Gelu", b.Linear(y, m + "pwconv1.weight", m + "pwconv1.bias")), m + "pwconv2.weight", m + "pwconv2.bias");
            x = b.Add(x, b.Mul(y, b.Weight(m + "gamma")));
        }

        x = Norm(b, x, p + "final_layer_norm");
        x = b.Linear(x, prefix + "head.out.weight", prefix + "head.out.bias");
        int bins = (nFft / 2) + 1;
        var magnitude = b.Emit1("Clip", [b.Unary("Exp", b.Slice(x, 1, 0, bins))], Attributes.From(("min", float.NegativeInfinity), ("max", 100f)));
        var phase = b.Slice(x, 1, bins, bins);
        var real = b.Mul(magnitude, b.Unary("Cos", phase));
        var imag = b.Mul(magnitude, b.Unary("Sin", phase));
        var stacked = b.Concat(2, b.Reshape(real, 0, bins, 1), b.Reshape(imag, 0, bins, 1));
        var inverse = b.Emit1("Dft", [stacked], Attributes.From(("length", (long)nFft), ("out_bins", (long)nFft), ("inverse", true), ("real_output", true)));
        var window = b.Constant("vocos.window", DataType.F32, new Shape(nFft), KokoroBlocks.Bytes(Window.Hann(nFft)));
        b.Output(FramesOutput, b.Mul(b.Reshape(inverse, 0, nFft), window));
        return b.Build();
    }

    private static Value Norm(GraphBuilder b, Value x, string module)
        => b.Emit1("LayerNorm", [x, b.Weight(module + ".weight"), b.Weight(module + ".bias")], Attributes.From(("eps", 1e-6f)));
}
