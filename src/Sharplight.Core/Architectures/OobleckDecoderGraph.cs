using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// The decoder of the Oobleck audio autoencoder (Stable Audio's VAE, diffusers <c>AutoencoderOobleck</c>; ACE-Step and Stable Audio Open use it) over time-major signals:
/// an input convolution, upsampling blocks (snake, transposed convolution, three dilated residual units) and an output snake and convolution. Weights under a prefix
/// (<c>{prefix}conv1</c>, <c>{prefix}block.{i}.{snake1, conv_t1, res_unit{1,2,3}.{snake1, conv1, snake2, conv2}}</c>, <c>{prefix}snake1</c>, <c>{prefix}conv2</c>) with the weight
/// normalization folded and the snake parameters flattened to <c>[C]</c> (log-scale <c>alpha</c> and <c>beta</c>).
/// </summary>
internal static class OobleckDecoderGraph
{
    private const float SnakeEpsilon = 1e-9f;

    /// <summary>Emits the decoder: latents <c>[T, C]</c> to audio <c>[T · product of the strides, channels]</c>.</summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="x">The latents.</param>
    /// <param name="prefix">The weight prefix of the decoder (<c>vae.decoder.</c>).</param>
    /// <param name="upsampling">The upsampling strides, deep to shallow.</param>
    public static Value Build(GraphBuilder b, Value x, string prefix, IReadOnlyList<long> upsampling)
    {
        x = Conv(b, x, prefix + "conv1", 3);
        for (int i = 0; i < upsampling.Count; i++)
        {
            using var _ = b.Group($"{prefix}block.{i}");
            string n = $"{prefix}block.{i}.";
            long stride = upsampling[i];
            x = Snake(b, x, n + "snake1");
            long crop = (stride + 1) / 2;
            x = b.Emit1("ConvTranspose1d", [x, b.Weight(n + "conv_t1.weight"), b.Weight(n + "conv_t1.bias")],
                Attributes.From(("stride", stride), ("crop_left", crop), ("crop_right", crop)));
            x = Residual(b, x, n + "res_unit1.", 1);
            x = Residual(b, x, n + "res_unit2.", 3);
            x = Residual(b, x, n + "res_unit3.", 9);
        }

        x = Snake(b, x, prefix + "snake1");
        return Conv(b, x, prefix + "conv2", 3);
    }

    private static Value Residual(GraphBuilder b, Value x, string name, long dilation)
    {
        var y = Conv(b, Snake(b, x, name + "snake1"), name + "conv1", 3 * dilation, dilation);
        y = Conv(b, Snake(b, y, name + "snake2"), name + "conv2", 0);
        return b.Add(x, y);
    }

    // x + sin²(exp(alpha) · x) / (exp(beta) + 1e-9) with per-channel log-scale parameters.
    private static Value Snake(GraphBuilder b, Value x, string name)
    {
        var alpha = b.Unary("Exp", b.Weight(name + ".alpha"));
        var inverse = b.Unary("Recip", b.Scale(b.Unary("Exp", b.Weight(name + ".beta")), 1f, SnakeEpsilon));
        return b.Add(x, b.Mul(inverse, b.Unary("Square", b.Unary("Sin", b.Mul(alpha, x)))));
    }

    private static Value Conv(GraphBuilder b, Value x, string name, long pad, long dilation = 1)
    {
        var inputs = b.HasWeight(name + ".bias") ? new[] { x, b.Weight(name + ".weight"), b.Weight(name + ".bias") } : [x, b.Weight(name + ".weight")];
        return b.Emit1("Conv1d", inputs, Attributes.From(("stride", 1L), ("dilation", dilation), ("pad_left", pad), ("pad_right", pad)));
    }
}
