using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the EnCodec encoder (SEANet convolutions with causal reflect padding, residual units, a two-layer LSTM bottleneck, then the residual vector quantizer
/// searching the nearest codebook entry per stage) and decoder (the mirror with transposed convolutions) as graphs of generic ops over time-major signals with a
/// symbolic length <c>T</c>. The audio length must be a multiple of the samples per frame (the client pads), so no strided convolution needs extra padding.
/// </summary>
internal static class EncodecGraph
{
    /// <summary>Builds the encoder graph: <c>audio</c> [T, 1] to the per-codebook codes.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildEncoder(EncodecParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(AudioCodecGraphs.EncodeGraph, weights);
        var x = b.Input(AudioCodecGraphs.AudioInput, InputRole.Media, DataType.F32, new Shape(Dim.Sym(WellKnown.TokenSymbol), 1));
        int layer = 0;
        x = Conv(b, p, x, $"encoder.layers.{layer++}", p.KernelSize);
        long scale = 1;
        for (int i = p.Ratios.Length - 1; i >= 0; i--)
        {
            for (int j = 0; j < p.ResidualLayers; j++)
            {
                x = Residual(b, p, x, $"encoder.layers.{layer++}", [(long)Math.Pow(p.DilationGrowth, j), 1]);
            }

            long ratio = p.Ratios[i];
            x = Conv(b, p, b.Unary("Elu", x), $"encoder.layers.{layer + 1}", (int)(2 * ratio), stride: ratio);
            layer += 2;
            scale *= 2;
        }

        x = Lstm(b, p, x, $"encoder.layers.{layer++}");
        x = Conv(b, p, b.Unary("Elu", x), $"encoder.layers.{layer + 1}", p.LastKernelSize);

        var residual = x;
        for (int q = 0; q < p.Codebooks; q++)
        {
            using var _ = b.Group($"quantizer.{q}");
            var embed = b.Weight($"quantizer.layers.{q}.codebook.embed");
            // The distance as the reference writes it (|x|² - 2 x·e + |e|², rounded in the same order), so that near ties break the same way.
            var cross = b.Scale(b.MatMul(residual, embed), 2f);
            var norm = b.Reshape(b.Unary("ReduceSum", b.Unary("Square", embed)), 1, p.CodebookSize);
            var distance = b.Add(b.Sub(b.Unary("ReduceSum", b.Unary("Square", residual)), cross), norm);
            var index = b.Unary("ArgMax", b.Unary("Neg", distance));
            b.Output(AudioCodecGraphs.CodesName(q), index);
            if (q + 1 < p.Codebooks)
            {
                residual = b.Sub(residual, b.Gather(embed, b.Reshape(index, 0)));
            }
        }

        return b.Build();
    }

    /// <summary>Builds the decoder graph: the codes of the codebooks (each masked in or out) to <c>audio</c> [T · samples per frame, 1].</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildDecoder(EncodecParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(AudioCodecGraphs.DecodeGraph, weights);
        var frames = Dim.Sym(WellKnown.TokenSymbol);
        var mask = b.Input(AudioCodecGraphs.MaskInput, InputRole.Other, DataType.F32, new Shape(p.Codebooks));
        Value? latent = null;
        for (int q = 0; q < p.Codebooks; q++)
        {
            var codes = b.Input(AudioCodecGraphs.CodesName(q), InputRole.TokenIds, DataType.I32, new Shape(frames));
            var vectors = b.Mul(b.Gather(b.Weight($"quantizer.layers.{q}.codebook.embed"), codes), b.Slice(mask, 0, q, 1));
            latent = latent is null ? vectors : b.Add(latent, vectors);
        }

        int layer = 0;
        var x = Conv(b, p, latent!, $"decoder.layers.{layer++}", p.KernelSize);
        x = Lstm(b, p, x, $"decoder.layers.{layer++}");
        foreach (long ratio in p.Ratios)
        {
            x = b.Unary("Elu", x);
            x = b.Emit1("ConvTranspose1d", [x, b.Weight($"decoder.layers.{layer + 1}.weight"), b.Weight($"decoder.layers.{layer + 1}.bias")],
                Attributes.From(("stride", ratio), ("crop_left", p.Causal ? 0L : ratio - (ratio / 2)), ("crop_right", p.Causal ? ratio : ratio / 2)));
            layer += 2;
            for (int j = 0; j < p.ResidualLayers; j++)
            {
                x = Residual(b, p, x, $"decoder.layers.{layer++}", [(long)Math.Pow(p.DilationGrowth, j), 1]);
            }
        }

        x = Conv(b, p, b.Unary("Elu", x), $"decoder.layers.{layer + 1}", p.LastKernelSize);
        b.Output(AudioCodecGraphs.AudioOutput, x);
        return b.Build();
    }

    private static Value Residual(GraphBuilder b, EncodecParameters p, Value x, string name, long[] dilations)
    {
        var y = Conv(b, p, b.Unary("Elu", x), name + ".block.1", p.ResidualKernelSize, dilation: dilations[0]);
        y = Conv(b, p, b.Unary("Elu", y), name + ".block.3", 1, dilation: dilations[1]);
        return b.Add(p.ConvShortcut ? Conv(b, p, x, name + ".shortcut", 1) : x, y);
    }

    // The causal convolution with reflect padding: padding_total = (kernel - 1) · dilation + 1 - stride on the left, none on the right.
    private static Value Conv(GraphBuilder b, EncodecParameters p, Value x, string name, int kernel, long stride = 1, long dilation = 1)
    {
        long total = (((kernel - 1) * dilation) + 1) - stride;
        long right = p.Causal ? 0 : total / 2;
        var attributes = Attributes.From(("stride", stride), ("dilation", dilation), ("pad_left", total - right), ("pad_right", right), ("pad_mode", total > 0 ? "reflect" : "zeros"));
        return b.Emit1("Conv1d", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")], attributes);
    }

    // The recurrent bottleneck with a residual connection around all LSTM layers.
    private static Value Lstm(GraphBuilder b, EncodecParameters p, Value x, string name)
    {
        var h = x;
        for (int l = 0; l < p.LstmLayers; l++)
        {
            string n = $"{name}.lstm.";
            var gates = b.Linear(h, n + $"weight_ih_l{l}", n + $"bias_l{l}");
            h = b.Emit1("Lstm", [gates, b.Weight(n + $"weight_hh_l{l}")], Attributes.From(("reverse", false)));
        }

        return b.Add(h, x);
    }
}
