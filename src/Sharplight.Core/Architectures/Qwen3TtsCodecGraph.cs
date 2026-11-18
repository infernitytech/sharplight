using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the decoder of the Qwen3-TTS 12 Hz codec as a graph of generic ops over time-major signals: the split residual quantizer (the first codebook and the rest each project their
/// summed entries), a causal convolution, the sliding-window transformer (RMS norm, rotary attention, SwiGLU, layer scale), the ConvNeXt upsampling stages, and the snake-activated
/// vocoder of transposed convolutions and dilated residual units. The encoder is Mimi's graph (<see cref="MimiGraph.BuildEncoder"/>).
/// </summary>
internal static class Qwen3TtsCodecGraph
{
    private const float SnakeEpsilon = 1e-9f;

    /// <summary>Builds the decoder graph: the codes of the codebooks (each masked in or out) to <c>audio</c> [T · samples per frame, 1].</summary>
    /// <param name="p">The decoder hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    /// <param name="name">The graph's name.</param>
    public static ComputeGraph BuildDecoder(Qwen3TtsCodecParameters p, Func<string, WeightInfo?> weights, string name = AudioCodecGraphs.DecodeGraph)
    {
        var b = new GraphBuilder(name, weights);
        var frames = Dim.Sym(WellKnown.TokenSymbol);
        var mask = b.Input(AudioCodecGraphs.MaskInput, InputRole.Other, DataType.F32, new Shape(p.Quantizers));
        Value? first = null;
        Value? rest = null;
        for (int q = 0; q < p.Quantizers; q++)
        {
            bool isFirst = q < p.SemanticQuantizers;
            string embed = isFirst ? $"decoder.quantizer.first.layers.{q}.embed" : $"decoder.quantizer.rest.layers.{q - p.SemanticQuantizers}.embed";
            var codes = b.Input(AudioCodecGraphs.CodesName(q), InputRole.TokenIds, DataType.I32, new Shape(frames));
            var vectors = b.Mul(b.Gather(b.Weight(embed), codes), b.Slice(mask, 0, q, 1));
            if (isFirst)
            {
                first = first is null ? vectors : b.Add(first, vectors);
            }
            else
            {
                rest = rest is null ? vectors : b.Add(rest, vectors);
            }
        }

        var x = b.Add(Pointwise(b, first!, "decoder.quantizer.first.output_proj.weight"), Pointwise(b, rest!, "decoder.quantizer.rest.output_proj.weight"));
        x = Conv(b, x, "decoder.pre_conv", 3);
        x = Transformer(b, p, x);
        for (int i = 0; i < p.UpsamplingRatios.Length; i++)
        {
            long factor = p.UpsamplingRatios[i];
            x = b.Emit1("ConvTranspose1d", [x, b.Weight($"decoder.upsample.{i}.0.weight"), b.Weight($"decoder.upsample.{i}.0.bias")],
                Attributes.From(("stride", factor), ("crop_left", 0L), ("crop_right", 0L)));
            x = ConvNeXt(b, p, x, $"decoder.upsample.{i}.1");
        }

        x = Conv(b, x, "decoder.decoder.0", 7);
        for (int i = 0; i < p.UpsampleRates.Length; i++)
        {
            string n = $"decoder.decoder.{i + 1}.block";
            long rate = p.UpsampleRates[i];
            x = Snake(b, x, n + ".0");
            x = b.Emit1("ConvTranspose1d", [x, b.Weight(n + ".1.weight"), b.Weight(n + ".1.bias")],
                Attributes.From(("stride", rate), ("crop_left", 0L), ("crop_right", rate)));
            foreach (var (unit, dilation) in new[] { (2, 1L), (3, 3L), (4, 9L) })
            {
                string u = $"{n}.{unit}";
                var y = Conv(b, Snake(b, x, u + ".act1"), u + ".conv1", 7, dilation);
                y = Conv(b, Snake(b, y, u + ".act2"), u + ".conv2", 1);
                x = b.Add(x, y);
            }
        }

        int tail = p.UpsampleRates.Length + 1;
        x = Conv(b, Snake(b, x, $"decoder.decoder.{tail}"), $"decoder.decoder.{tail + 1}", 7);
        b.Output(AudioCodecGraphs.AudioOutput, b.Emit1("Clip", [x], Attributes.From(("min", -1f), ("max", 1f))));
        return b.Build();
    }

    private static Value Transformer(GraphBuilder b, Qwen3TtsCodecParameters p, Value x)
    {
        var positions = b.Input(AudioCodecGraphs.PositionsInput, InputRole.Positions, DataType.I32, new Shape(x.Shape[0], 1));
        var spans = b.Input(AudioCodecGraphs.SpansInput, InputRole.RowSpans, DataType.I32, new Shape(x.Shape[0], 2));
        var rope = Attributes.From(("dims", (long)p.HeadDim), ("base", p.RopeBase), ("style", "neox"));
        var attention = Attributes.From(("scale", 1f / MathF.Sqrt(p.HeadDim)), ("causal", true), ("window", (long)p.Window));
        const string Prefix = "decoder.pre_transformer";
        x = b.Add(b.MatMul(x, b.Weight(Prefix + ".input_proj.weight")), b.Weight(Prefix + ".input_proj.bias"));
        for (int il = 0; il < p.Layers; il++)
        {
            using var _ = b.Group($"{Prefix}.layers.{il}");
            string n = $"{Prefix}.layers.{il}.";
            var h = b.RmsNorm(x, b.Weight(n + "input_layernorm.weight"), p.Eps);
            var q = b.Emit1("Rope", [b.Reshape(b.MatMul(h, b.Weight(n + "self_attn.q_proj.weight")), 0, p.Heads, p.HeadDim), positions], rope);
            var k = b.Emit1("Rope", [b.Reshape(b.MatMul(h, b.Weight(n + "self_attn.k_proj.weight")), 0, p.Heads, p.HeadDim), positions], rope);
            var v = b.Reshape(b.MatMul(h, b.Weight(n + "self_attn.v_proj.weight")), 0, p.Heads, p.HeadDim);
            var attended = b.Emit1("Attention", [q, k, v, spans], attention);
            var o = b.MatMul(b.Reshape(attended, 0, p.Heads * p.HeadDim), b.Weight(n + "self_attn.o_proj.weight"));
            x = b.Add(x, b.Mul(o, b.Weight(n + "self_attn_layer_scale.scale")));
            h = b.RmsNorm(x, b.Weight(n + "post_attention_layernorm.weight"), p.Eps);
            var gated = b.Mul(b.Silu(b.MatMul(h, b.Weight(n + "mlp.gate_proj.weight"))), b.MatMul(h, b.Weight(n + "mlp.up_proj.weight")));
            x = b.Add(x, b.Mul(b.MatMul(gated, b.Weight(n + "mlp.down_proj.weight")), b.Weight(n + "mlp_layer_scale.scale")));
        }

        x = b.RmsNorm(x, b.Weight(Prefix + ".norm.weight"), p.Eps);
        return b.Add(b.MatMul(x, b.Weight(Prefix + ".output_proj.weight")), b.Weight(Prefix + ".output_proj.bias"));
    }

    private static Value ConvNeXt(GraphBuilder b, Qwen3TtsCodecParameters p, Value x, string name)
    {
        var h = b.Emit1("Conv1d", [x, b.Weight(name + ".dwconv.weight"), b.Weight(name + ".dwconv.bias")],
            Attributes.From(("stride", 1L), ("dilation", 1L), ("pad_left", 6L), ("pad_right", 0L), ("groups", (long)p.Latent)));
        h = TowerBlocks.LayerNorm(b, h, name + ".norm", 1e-6f);
        h = b.Add(b.MatMul(h, b.Weight(name + ".pwconv1.weight")), b.Weight(name + ".pwconv1.bias"));
        h = b.Add(b.MatMul(b.Unary("Gelu", h), b.Weight(name + ".pwconv2.weight")), b.Weight(name + ".pwconv2.bias"));
        return b.Add(x, b.Mul(h, b.Weight(name + ".gamma")));
    }

    // SnakeBeta: x + sin²(x · e^alpha) / (e^beta + 1e-9), with alpha and beta stored as logarithms.
    private static Value Snake(GraphBuilder b, Value x, string name)
    {
        var alpha = b.Unary("Exp", b.Weight(name + ".alpha"));
        var inverse = b.Unary("Recip", b.Scale(b.Unary("Exp", b.Weight(name + ".beta")), 1f, SnakeEpsilon));
        return b.Add(x, b.Mul(inverse, b.Unary("Square", b.Unary("Sin", b.Mul(alpha, x)))));
    }

    private static Value Pointwise(GraphBuilder b, Value x, string weight) => b.Emit1("Conv1d", [x, b.Weight(weight)]);

    // The causal convolution with zero padding on the left ((kernel - 1) · dilation); stride 1.
    private static Value Conv(GraphBuilder b, Value x, string name, int kernel, long dilation = 1)
    {
        long total = (kernel - 1) * dilation;
        return b.Emit1("Conv1d", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")], Attributes.From(("stride", 1L), ("dilation", dilation), ("pad_left", total), ("pad_right", 0L)));
    }
}
