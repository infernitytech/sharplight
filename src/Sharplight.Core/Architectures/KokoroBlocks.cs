using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// Building blocks of the Kokoro graphs over time-major signals <c>[T, C]</c>: bidirectional LSTMs, instance and adaptive instance normalization,
/// the adaptive residual blocks of the prosody predictor and decoder, the snake-activated residual blocks of the generator. Style vectors are <c>[1, 128]</c>.
/// </summary>
internal static class KokoroBlocks
{
    private const float NormEps = 1e-5f;

    /// <summary>A 1-D convolution with symmetric zero padding; bias optional.</summary>
    public static Value Conv(GraphBuilder b, Value x, string name, long pad, long dilation = 1, long stride = 1, bool bias = true, string padMode = "zeros")
    {
        var weight = b.Weight(name + ".weight");
        Value[] inputs = bias && b.HasWeight(name + ".bias") ? [x, weight, b.Weight(name + ".bias")] : [x, weight];
        return b.Emit1("Conv1d", inputs, Attributes.From(("pad_left", pad), ("pad_right", pad), ("dilation", dilation), ("stride", stride), ("pad_mode", padMode)));
    }

    /// <summary>One direction pair of an LSTM layer (PyTorch parameter names under <paramref name="name"/>), concatenated to <c>[T, 2H]</c>.</summary>
    public static Value BiLstm(GraphBuilder b, Value x, string name)
    {
        var forward = b.Emit1("Lstm", [b.Linear(x, name + ".weight_ih_l0", name + ".bias_l0"), b.Weight(name + ".weight_hh_l0")], Attributes.From(("reverse", false)));
        var backward = b.Emit1("Lstm", [b.Linear(x, name + ".weight_ih_l0_reverse", name + ".bias_l0_reverse"), b.Weight(name + ".weight_hh_l0_reverse")],
            Attributes.From(("reverse", true)));
        return b.Concat(1, forward, backward);
    }

    /// <summary>Leaky ReLU.</summary>
    public static Value LeakyRelu(GraphBuilder b, Value x, float slope) => b.Emit1("LeakyRelu", [x], Attributes.From(("alpha", slope)));

    /// <summary>Instance normalization without affine parameters: statistics per channel over all frames.</summary>
    public static Value InstanceNorm(GraphBuilder b, Value x)
    {
        var axes = Attributes.From(("mode", "mean"), ("axes", new long[] { 0 }), ("keepdims", true));
        var centered = b.Sub(x, b.Emit1("Reduce", [x], axes));
        var variance = b.Emit1("Reduce", [b.Unary("Square", centered)], axes);
        return b.Mul(centered, b.Unary("Rsqrt", b.Scale(variance, 1f, NormEps)));
    }

    /// <summary>The <c>(gamma, beta)</c> pair a style vector projects to, each <c>[1, channels]</c>.</summary>
    public static (Value Gamma, Value Beta) StyleScaleShift(GraphBuilder b, Value style, string name, long channels)
    {
        var h = b.Linear(style, name + ".fc.weight", name + ".fc.bias");
        return (b.Slice(h, 1, 0, channels), b.Slice(h, 1, channels, channels));
    }

    /// <summary>Adaptive instance normalization: <c>(1 + gamma) · norm(x) + beta</c> with gamma and beta from the style.</summary>
    public static Value AdaIn(GraphBuilder b, Value x, Value style, string name, long channels)
    {
        var (gamma, beta) = StyleScaleShift(b, style, name, channels);
        return b.Add(b.Mul(InstanceNorm(b, x), b.Scale(gamma, 1f, 1f)), beta);
    }

    /// <summary>Adaptive layer normalization over channels (the duration encoder): <c>(1 + gamma) · layer_norm(x) + beta</c>.</summary>
    public static Value AdaLayerNorm(GraphBuilder b, Value x, Value style, string name, long channels)
    {
        var (gamma, beta) = StyleScaleShift(b, style, name, channels);
        var normalized = b.Emit1("LayerNorm", [x], Attributes.From(("eps", NormEps)));
        return b.Add(b.Mul(normalized, b.Scale(gamma, 1f, 1f)), beta);
    }

    /// <summary><paramref name="style"/> repeated to one row per frame of <paramref name="like"/> (<c>[T, width]</c>).</summary>
    public static Value StyleRows(GraphBuilder b, Value style, Value like)
        => b.Add(b.Scale(b.Slice(like, 1, 0, 1), 0f), style);

    /// <summary>Nearest-neighbor upsampling by two along time.</summary>
    public static Value Upsample2(GraphBuilder b, Value x)
        => b.Emit1("Interpolate1d", [x], Attributes.From(("mode", "nearest"), ("up", 2L), ("down", 1L)));

    /// <summary>
    /// An adaptive residual block (<c>AdainResBlk1d</c>): norm, leaky ReLU, optional depthwise transposed-convolution upsampling, convolution, norm, leaky ReLU,
    /// convolution, plus the (upsampled, projected) input, scaled by <c>1/√2</c>.
    /// </summary>
    public static Value AdainResBlock(GraphBuilder b, Value x, Value style, string name, long dimIn, long dimOut, bool upsample)
    {
        var r = LeakyRelu(b, AdaIn(b, x, style, name + ".norm1", dimIn), 0.2f);
        if (upsample)
        {
            r = b.Emit1("ConvTranspose1d", [r, b.Weight(name + ".pool.weight"), b.Weight(name + ".pool.bias")],
                Attributes.From(("stride", 2L), ("crop_left", 1L), ("crop_right", 1L), ("output_padding", 1L), ("groups", dimIn)));
        }

        r = Conv(b, r, name + ".conv1", 1);
        r = Conv(b, LeakyRelu(b, AdaIn(b, r, style, name + ".norm2", dimOut), 0.2f), name + ".conv2", 1);
        var shortcut = upsample ? Upsample2(b, x) : x;
        if (dimIn != dimOut)
        {
            shortcut = Conv(b, shortcut, name + ".conv1x1", 0, bias: false);
        }

        return b.Scale(b.Add(r, shortcut), 1f / MathF.Sqrt(2f));
    }

    /// <summary>The generator's residual block (<c>AdaINResBlock1</c>): three rounds of norm, snake, dilated convolution, norm, snake, convolution, each added to its input.</summary>
    public static Value SnakeResBlock(GraphBuilder b, Value x, Value style, string name, long channels, long kernel, long[] dilations)
    {
        for (int i = 0; i < dilations.Length; i++)
        {
            var xt = SnakeActivation.Apply(b, AdaIn(b, x, style, $"{name}.adain1.{i}", channels), b.Weight($"{name}.alpha1.{i}"));
            xt = Conv(b, xt, $"{name}.convs1.{i}", ((kernel * dilations[i]) - dilations[i]) / 2, dilations[i]);
            xt = SnakeActivation.Apply(b, AdaIn(b, xt, style, $"{name}.adain2.{i}", channels), b.Weight($"{name}.alpha2.{i}"));
            xt = Conv(b, xt, $"{name}.convs2.{i}", (kernel - 1) / 2);
            x = b.Add(xt, x);
        }

        return x;
    }

    /// <summary>Little-endian bytes of <paramref name="values"/>.</summary>
    public static byte[] Bytes(float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
