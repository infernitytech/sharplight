using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the SNAC decoder as a graph of generic ops over time-major signals with a symbolic number of coarse-codebook frames <c>T</c>: the codebook lookups with
/// their output projections and nearest-neighbor upsampling to the finest rate, the depthwise input convolutions, the decoder blocks (snake, transposed
/// convolution, noise injection, three dilated residual units) and the output convolution with tanh. The random noise of the blocks comes in as inputs
/// (<see cref="NoiseInput"/>) so a decode is reproducible.
/// </summary>
internal static class SnacGraph
{
    /// <summary>The graph name.</summary>
    public const string GraphName = "codec_decode";

    /// <summary>The audio output, f32 <c>[samples, 1]</c>.</summary>
    public const string AudioOutput = "audio";

    private const float SnakeEpsilon = 1e-9f;

    /// <summary>The input of the codes of codebook <paramref name="index"/>, i32 <c>[T · stride₀ / stride_i]</c>.</summary>
    /// <param name="index">The codebook.</param>
    public static string CodesInput(int index) => $"codes.{index}";

    /// <summary>The input of the noise of decoder block <paramref name="index"/>, f32 <c>[samples of the block's output, 1]</c> (standard normal).</summary>
    /// <param name="index">The block.</param>
    public static string NoiseInput(int index) => $"noise.{index}";

    /// <summary>Builds the graph reading the codec's weights as <c>{prefix}name</c>.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by its full name.</param>
    /// <param name="prefix">The prefix of the codec's weights in the model.</param>
    public static ComputeGraph Build(SnacParameters p, Func<string, WeightInfo?> weights, string prefix)
    {
        var b = new GraphBuilder(GraphName, weights);
        Value W(string name) => b.Weight(prefix + name);
        var frames = Dim.Sym(WellKnown.TokenSymbol);
        long coarse = p.VqStrides[0];

        Value? latent = null;
        for (int i = 0; i < p.VqStrides.Length; i++)
        {
            var codes = b.Input(CodesInput(i), InputRole.TokenIds, DataType.I32, new Shape(frames * (coarse / p.VqStrides[i])));
            var vectors = b.Gather(W($"quantizer.quantizers.{i}.codebook.weight"), codes);
            var projected = Conv(b, vectors, prefix + $"quantizer.quantizers.{i}.out_proj", 0);
            var upsampled = b.Emit1("Interpolate1d", [projected], Attributes.From(("mode", "nearest"), ("up", p.VqStrides[i]), ("down", 1L)));
            latent = latent is null ? upsampled : b.Add(latent, upsampled);
        }

        var x = Conv(b, latent!, prefix + "decoder.model.0", 3, groups: p.LatentDim);
        x = Conv(b, x, prefix + "decoder.model.1", 0);
        long scale = coarse;
        for (int i = 0; i < p.DecoderRates.Length; i++)
        {
            using var _ = b.Group($"decoder.block.{i}");
            long stride = p.DecoderRates[i];
            long outChannels = p.DecoderDim >> (i + 1);
            string n = prefix + $"decoder.model.{i + 2}.block.";
            x = SnakeActivation.Apply(b, x, W($"decoder.model.{i + 2}.block.0.alpha"), SnakeEpsilon);
            long crop = (stride + 1) / 2;
            x = b.Emit1("ConvTranspose1d", [x, b.Weight(n + "1.weight"), b.Weight(n + "1.bias")],
                Attributes.From(("stride", stride), ("crop_left", crop), ("crop_right", crop), ("output_padding", stride % 2)));
            scale *= stride;
            int unit = 2;
            if (p.Noise)
            {
                var noise = b.Input(NoiseInput(i), InputRole.Other, DataType.F32, new Shape(frames * scale, 1));
                x = b.Add(x, b.Mul(noise, Conv(b, x, n + "2.linear", 0, bias: false)));
                unit = 3;
            }

            long[] dilations = [1, 3, 9];
            for (int u = 0; u < dilations.Length; u++)
            {
                string r = n + $"{unit + u}.block.";
                var y = SnakeActivation.Apply(b, x, b.Weight(r + "0.alpha"), SnakeEpsilon);
                y = Conv(b, y, r + "1", ((7 - 1) * dilations[u]) / 2, dilation: dilations[u], groups: p.Depthwise ? outChannels : 1);
                y = SnakeActivation.Apply(b, y, b.Weight(r + "2.alpha"), SnakeEpsilon);
                y = Conv(b, y, r + "3", 0);
                x = b.Add(x, y);
            }
        }

        int tail = p.DecoderRates.Length + 2;
        x = SnakeActivation.Apply(b, x, W($"decoder.model.{tail}.alpha"), SnakeEpsilon);
        x = Conv(b, x, prefix + $"decoder.model.{tail + 1}", 3);
        b.Output(AudioOutput, b.Unary("Tanh", x));
        return b.Build();
    }

    private static Value Conv(GraphBuilder b, Value x, string name, long pad, long dilation = 1, long groups = 1, bool bias = true)
    {
        var weight = b.Weight(name + ".weight");
        Value[] inputs = bias && b.HasWeight(name + ".bias") ? [x, weight, b.Weight(name + ".bias")] : [x, weight];
        return b.Emit1("Conv1d", inputs, Attributes.From(("pad_left", pad), ("pad_right", pad), ("dilation", dilation), ("groups", groups)));
    }
}
