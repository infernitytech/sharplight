using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the Higgs Audio v2 tokenizer as graphs of generic ops over time-major signals with a symbolic length <c>T</c>. The encoder fuses two views of the audio per 25 Hz frame:
/// a snake-activated convolutional acoustic encoder (DAC-style) and the layer-averaged hidden states of a HuBERT encoder (run on the audio resampled to 16 kHz, kept every
/// other frame) through a small convolutional semantic encoder; a linear layer fuses the two and a residual vector quantizer with Euclidean codebooks and linear projections
/// picks the codes. The decoder sums the projected codebook entries, projects them to the acoustic width and runs the transposed-convolution acoustic decoder (no output
/// activation). The audio length must be a multiple of the samples per frame (the client pads).
/// </summary>
internal static class HiggsAudioCodecGraph
{
    private const float SnakeEpsilon = 1e-9f;

    /// <summary>The weight of the resampling kernel (the adapter derives it).</summary>
    public const string ResampleKernel = "resample.kernel";

    /// <summary>Builds the encoder graph: <c>audio</c> [T, 1] to the per-codebook codes.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="kernel">The resampling kernel from the codec's rate to the semantic rate.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildEncoder(HiggsAudioCodecParameters p, SincResampleKernel kernel, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(AudioCodecGraphs.EncodeGraph, weights);
        var audio = b.Input(AudioCodecGraphs.AudioInput, InputRole.Media, DataType.F32, new Shape(Dim.Sym(WellKnown.TokenSymbol), 1));

        Value semantic;
        using (b.Group("semantic"))
        {
            var resampled = b.Emit1("Conv1d", [audio, b.Weight(ResampleKernel)],
                Attributes.From(("stride", (long)kernel.OriginalFrequency), ("pad_left", (long)kernel.Width), ("pad_right", (long)kernel.Width)));
            // [frames, new rate] -> the resampled signal [frames · new rate, 1] (the channels are the interleaved phases).
            var signal = b.Emit1("Reshape", [resampled], Attributes.From(("shape", new long[] { 0, 1 }), ("outer_symbol", (resampled.Shape[0] * kernel.NewFrequency).Expression.ToString())));
            semantic = Hubert(b, p, signal);
            semantic = SemanticEncoder(b, p, semantic);
        }

        Value acoustic;
        using (b.Group("acoustic"))
        {
            acoustic = AcousticEncoder(b, p, audio);
        }

        var fused = Linear(b, b.Concat(1, acoustic, semantic), "fc");
        var residual = fused;
        for (int q = 0; q < p.Codebooks; q++)
        {
            using var _ = b.Group($"quantizer.{q}");
            string n = $"quantizer.quantizers.{q}.";
            var latent = Linear(b, residual, n + "project_in");
            var embed = b.Weight(n + "codebook.embed");
            var squares = b.Emit1("Reduce", [b.Unary("Square", embed)], Attributes.From(("mode", "sum"), ("axes", new long[] { 1 }), ("keepdims", false)));
            var score = b.Sub(b.Scale(b.MatMul(latent, embed), 2f), squares);
            var index = b.Unary("ArgMax", score);
            b.Output(AudioCodecGraphs.CodesName(q), index);
            if (q + 1 < p.Codebooks)
            {
                residual = b.Sub(residual, Linear(b, b.Gather(embed, b.Reshape(index, 0)), n + "project_out"));
            }
        }

        return b.Build();
    }

    /// <summary>Builds the decoder graph: the codes of the codebooks (each masked in or out) to <c>audio</c> [T · samples per frame, 1].</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildDecoder(HiggsAudioCodecParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(AudioCodecGraphs.DecodeGraph, weights);
        var frames = Dim.Sym(WellKnown.TokenSymbol);
        var mask = b.Input(AudioCodecGraphs.MaskInput, InputRole.Other, DataType.F32, new Shape(p.Codebooks));
        Value? latent = null;
        for (int q = 0; q < p.Codebooks; q++)
        {
            string n = $"quantizer.quantizers.{q}.";
            var codes = b.Input(AudioCodecGraphs.CodesName(q), InputRole.TokenIds, DataType.I32, new Shape(frames));
            var projected = Linear(b, b.Gather(b.Weight(n + "codebook.embed"), codes), n + "project_out");
            projected = b.Mul(projected, b.Slice(mask, 0, q, 1));
            latent = latent is null ? projected : b.Add(latent, projected);
        }

        var x = Linear(b, latent!, "fc2");
        x = Conv(b, x, "acoustic_decoder.conv1", 3);
        for (int i = 0; i < p.UpsamplingRatios.Length; i++)
        {
            using var _ = b.Group($"acoustic_decoder.block.{i}");
            string n = $"acoustic_decoder.block.{i}.";
            long stride = p.UpsamplingRatios[i];
            x = Snake(b, x, n + "snake1.alpha");
            long crop = (stride + 1) / 2;
            x = b.Emit1("ConvTranspose1d", [x, b.Weight(n + "conv_t1.weight"), b.Weight(n + "conv_t1.bias")],
                Attributes.From(("stride", stride), ("crop_left", crop), ("crop_right", crop - (stride % 2))));
            x = Residual(b, x, n + "res_unit1.", 1);
            x = Residual(b, x, n + "res_unit2.", 3);
            x = Residual(b, x, n + "res_unit3.", 9);
        }

        x = Snake(b, x, "acoustic_decoder.snake1.alpha");
        x = Conv(b, x, "acoustic_decoder.conv2", 3);
        b.Output(AudioCodecGraphs.AudioOutput, x);
        return b.Build();
    }

    private static Value AcousticEncoder(GraphBuilder b, HiggsAudioCodecParameters p, Value x)
    {
        x = Conv(b, x, "acoustic_encoder.conv1", 3);
        for (int i = 0; i < p.DownsamplingRatios.Length; i++)
        {
            using var _ = b.Group($"acoustic_encoder.block.{i}");
            string n = $"acoustic_encoder.block.{i}.";
            long stride = p.DownsamplingRatios[i];
            x = Residual(b, x, n + "res_unit1.", 1);
            x = Residual(b, x, n + "res_unit2.", 3);
            x = Residual(b, x, n + "res_unit3.", 9);
            x = Snake(b, x, n + "snake1.alpha");
            long pad = (stride + 1) / 2;
            x = Conv(b, x, n + "conv1", pad, pad, stride);
        }

        x = Snake(b, x, "acoustic_encoder.snake1.alpha");
        return Conv(b, x, "acoustic_encoder.conv2", 1);
    }

    // HuBERT on the 16 kHz signal [T, 1] padded by 160 samples on each side: the mean of the embedding output (after the positional convolution and the norm) and of every
    // layer's output, every other frame.
    private static Value Hubert(GraphBuilder b, HiggsAudioCodecParameters p, Value x)
    {
        var h = p.Hubert;
        float eps = h.Eps;
        for (int i = 0; i < h.ConvStrides.Length; i++)
        {
            string n = $"semantic.encoder.conv.{i}.";
            Value[] inputs = b.HasWeight(n + "bias") ? [x, b.Weight(n + "weight"), b.Weight(n + "bias")] : [x, b.Weight(n + "weight")];
            var attributes = i == 0
                ? Attributes.From(("stride", h.ConvStrides[i]), ("pad_left", 160L), ("pad_right", 160L))
                : Attributes.From(("stride", h.ConvStrides[i]));
            x = b.Emit1("Conv1d", inputs, attributes);
            if (b.HasWeight(n + "norm.weight"))
            {
                x = ChannelNorm(b, x, n + "norm", eps);
            }

            x = b.Unary("Gelu", x);
        }

        x = LayerNorm(b, x, "semantic.encoder.projection_norm", eps);
        x = Linear(b, x, "semantic.encoder.projection");
        long kernel = b.Weight("semantic.encoder.position_conv.weight").Shape[2].Fixed;
        long pad = kernel / 2;
        var position = b.Emit1("Conv1d", [x, b.Weight("semantic.encoder.position_conv.weight"), b.Weight("semantic.encoder.position_conv.bias")],
            Attributes.From(("pad_left", pad), ("pad_right", kernel % 2 == 0 ? pad - 1 : pad), ("groups", (long)h.PositionGroups)));
        x = LayerNorm(b, b.Add(x, b.Unary("Gelu", position)), "semantic.encoder.norm", eps);
        var sum = x;
        int headDim = h.Hidden / h.Heads;
        float scale = 1f / MathF.Sqrt(headDim);
        for (int il = 0; il < h.Layers; il++)
        {
            using var _ = b.Group($"semantic.encoder.layers.{il}");
            string n = $"semantic.encoder.layers.{il}.";
            var q = b.Reshape(Linear(b, x, n + "attention.q"), 0, h.Heads, headDim);
            var k = b.Reshape(Linear(b, x, n + "attention.k"), 0, h.Heads, headDim);
            var v = b.Reshape(Linear(b, x, n + "attention.v"), 0, h.Heads, headDim);
            var attention = b.Emit1("Attention", [q, k, v], Attributes.From(("scale", scale), ("causal", false)));
            var o = Linear(b, b.Reshape(attention, 0, h.Hidden), n + "attention.o");
            x = LayerNorm(b, b.Add(x, o), n + "attention_norm", eps);
            var ffn = Linear(b, b.Unary("Gelu", Linear(b, x, n + "mlp.up")), n + "mlp.down");
            x = LayerNorm(b, b.Add(x, ffn), n + "mlp_norm", eps);
            sum = b.Add(sum, x);
        }

        return b.Scale(sum, 1f / (h.Layers + 1));
    }

    // The semantic encoder over the HuBERT features. Keeping every SemanticDownsample-th frame before its first convolution equals running that convolution with a stride and dilation
    // of SemanticDownsample over all frames.
    private static Value SemanticEncoder(GraphBuilder b, HiggsAudioCodecParameters p, Value x)
    {
        long same = p.ConvKernel / 2;
        long step = p.SemanticDownsample;
        x = b.Emit1("Conv1d", [x, b.Weight("encoder_semantic.conv.weight")],
            Attributes.From(("stride", step), ("dilation", step), ("pad_left", same * step), ("pad_right", same * step)));
        for (int i = 0; i < p.SemanticStrides.Length; i++)
        {
            string n = $"encoder_semantic.conv_blocks.{i}.";
            for (int j = 0; j < p.Dilations.Length; j++)
            {
                long dilation = p.Dilations[j];
                long padding = (p.UnitKernel - 1) / 2 * dilation;
                var y = b.Unary("Elu", x);
                y = b.Emit1("Conv1d", [y, b.Weight($"{n}res_units.{j}.conv1.weight")], Attributes.From(("dilation", dilation), ("pad_left", padding), ("pad_right", padding)));
                y = b.Emit1("Conv1d", [b.Unary("Elu", y), b.Weight($"{n}res_units.{j}.conv2.weight")]);
                x = b.Add(x, y);
            }

            long stride = p.SemanticStrides[i];
            long kernel = stride == 1 ? 3 : 2 * stride;
            long left = (kernel - 1) / 2;
            x = b.Emit1("Conv1d", [x, b.Weight(n + "conv.weight"), b.Weight(n + "conv.bias")], Attributes.From(("stride", stride), ("pad_left", left), ("pad_right", left)));
        }

        return x;
    }

    private static Value Residual(GraphBuilder b, Value x, string name, long dilation)
    {
        var y = Conv(b, Snake(b, x, name + "snake1.alpha"), name + "conv1", (6 * dilation) / 2, dilation: dilation);
        y = Conv(b, Snake(b, y, name + "snake2.alpha"), name + "conv2", 0);
        return b.Add(x, y);
    }

    private static Value Snake(GraphBuilder b, Value x, string alpha) => SnakeActivation.Apply(b, x, b.Weight(alpha), SnakeEpsilon);

    private static Value Linear(GraphBuilder b, Value x, string name) => b.Linear(x, name + ".weight", name + ".bias");

    private static Value LayerNorm(GraphBuilder b, Value x, string module, float eps)
        => b.Emit1("LayerNorm", [x, b.Weight(module + ".weight"), b.Weight(module + ".bias")], Attributes.From(("eps", eps)));

    // A group norm with one group per channel over the time-major signal [T, C].
    private static Value ChannelNorm(GraphBuilder b, Value x, string module, float eps)
    {
        var time = Attributes.From(("mode", "mean"), ("axes", new long[] { 0 }), ("keepdims", true));
        var centered = b.Sub(x, b.Emit1("Reduce", [x], time));
        var variance = b.Emit1("Reduce", [b.Unary("Square", centered)], time);
        var normed = b.Mul(centered, b.Unary("Rsqrt", b.Scale(variance, 1f, eps)));
        return b.Add(b.Mul(normed, b.Weight(module + ".weight")), b.Weight(module + ".bias"));
    }

    private static Value Conv(GraphBuilder b, Value x, string name, long padLeft, long? padRight = null, long stride = 1, long dilation = 1)
        => b.Emit1("Conv1d", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")],
            Attributes.From(("stride", stride), ("dilation", dilation), ("pad_left", padLeft), ("pad_right", padRight ?? padLeft)));
}
