using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the Descript Audio Codec as graphs of generic ops over time-major signals with a symbolic length <c>T</c>: the encoder (strided convolutions with dilated residual
/// units and snake activations), the residual vector quantizer with factorized codes (each stage projects the residual to the low-dimensional code space, picks the entry
/// of the largest cosine similarity and subtracts the projected-back entry), and the decoder (transposed convolutions). The audio length must be a multiple of the samples
/// per frame (the client pads).
/// </summary>
internal static class DacGraph
{
    private const float SnakeEpsilon = 1e-9f;

    /// <summary>Builds the encoder graph: <c>audio</c> [T, 1] to the per-codebook codes.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildEncoder(DacParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(AudioCodecGraphs.EncodeGraph, weights);
        var x = b.Input(AudioCodecGraphs.AudioInput, InputRole.Media, DataType.F32, new Shape(Dim.Sym(WellKnown.TokenSymbol), 1));
        x = Conv(b, x, "encoder.conv1", 3);
        for (int i = 0; i < p.DownsamplingRatios.Length; i++)
        {
            using var _ = b.Group($"encoder.block.{i}");
            string n = $"encoder.block.{i}.";
            long stride = p.DownsamplingRatios[i];
            x = Residual(b, x, n + "res_unit1.", 1);
            x = Residual(b, x, n + "res_unit2.", 3);
            x = Residual(b, x, n + "res_unit3.", 9);
            x = Snake(b, x, n + "snake1.alpha");
            long pad = (stride + 1) / 2;
            x = Conv(b, x, n + "conv1", pad, pad, stride);
        }

        x = Snake(b, x, "encoder.snake1.alpha");
        x = Conv(b, x, "encoder.conv2", 1);

        var residual = x;
        for (int q = 0; q < p.Codebooks; q++)
        {
            using var _ = b.Group($"quantizer.{q}");
            string n = $"quantizer.quantizers.{q}.";
            var latent = Conv(b, residual, n + "in_proj", 0);
            var codebook = b.L2Norm(b.Weight(n + "codebook.weight"), 1e-12f);
            var index = b.Unary("ArgMax", b.MatMul(latent, codebook));
            b.Output(AudioCodecGraphs.CodesName(q), index);
            if (q + 1 < p.Codebooks)
            {
                var entries = b.Gather(b.Weight(n + "codebook.weight"), b.Reshape(index, 0));
                residual = b.Sub(residual, Conv(b, entries, n + "out_proj", 0));
            }
        }

        return b.Build();
    }

    /// <summary>Builds the decoder graph: the codes of the codebooks (each masked in or out) to <c>audio</c> [T · samples per frame, 1].</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildDecoder(DacParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(AudioCodecGraphs.DecodeGraph, weights);
        var frames = Dim.Sym(WellKnown.TokenSymbol);
        var mask = b.Input(AudioCodecGraphs.MaskInput, InputRole.Other, DataType.F32, new Shape(p.Codebooks));
        Value? latent = null;
        for (int q = 0; q < p.Codebooks; q++)
        {
            string n = $"quantizer.quantizers.{q}.";
            var codes = b.Input(AudioCodecGraphs.CodesName(q), InputRole.TokenIds, DataType.I32, new Shape(frames));
            var projected = Conv(b, b.Gather(b.Weight(n + "codebook.weight"), codes), n + "out_proj", 0);
            projected = b.Mul(projected, b.Slice(mask, 0, q, 1));
            latent = latent is null ? projected : b.Add(latent, projected);
        }

        var x = Conv(b, latent!, "decoder.conv1", 3);
        for (int i = 0; i < p.UpsamplingRatios.Length; i++)
        {
            using var _ = b.Group($"decoder.block.{i}");
            string n = $"decoder.block.{i}.";
            long stride = p.UpsamplingRatios[i];
            x = Snake(b, x, n + "snake1.alpha");
            long crop = (stride + 1) / 2;
            x = b.Emit1("ConvTranspose1d", [x, b.Weight(n + "conv_t1.weight"), b.Weight(n + "conv_t1.bias")],
                Attributes.From(("stride", stride), ("crop_left", crop), ("crop_right", crop)));
            x = Residual(b, x, n + "res_unit1.", 1);
            x = Residual(b, x, n + "res_unit2.", 3);
            x = Residual(b, x, n + "res_unit3.", 9);
        }

        x = Snake(b, x, "decoder.snake1.alpha");
        x = Conv(b, x, "decoder.conv2", 3);
        b.Output(AudioCodecGraphs.AudioOutput, b.Unary("Tanh", x));
        return b.Build();
    }

    private static Value Residual(GraphBuilder b, Value x, string name, long dilation)
    {
        var y = Conv(b, Snake(b, x, name + "snake1.alpha"), name + "conv1", (6 * dilation) / 2, dilation: dilation);
        y = Conv(b, Snake(b, y, name + "snake2.alpha"), name + "conv2", 0);
        return b.Add(x, y);
    }

    private static Value Snake(GraphBuilder b, Value x, string alpha) => SnakeActivation.Apply(b, x, b.Weight(alpha), SnakeEpsilon);

    private static Value Conv(GraphBuilder b, Value x, string name, long padLeft, long? padRight = null, long stride = 1, long dilation = 1)
        => b.Emit1("Conv1d", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")],
            Attributes.From(("stride", stride), ("dilation", dilation), ("pad_left", padLeft), ("pad_right", padRight ?? padLeft)));
}
