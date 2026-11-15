using Sharplight.Graph;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the text-encoder graph of a T5 encoder (T5 v1.1 / Flan-T5, as FLUX and SD3 use it): token embeddings, pre-norm RMS
/// layers of bidirectional self-attention with T5's bucketed relative position bias and a (gated) feed-forward, the final norm.
/// One padded prompt of the fixed context length per run, no attention mask (the padding tokens take part, as in the diffusers
/// pipelines). Weights in the encoder-decoder vocabulary of <see cref="EncoderDecoderAdapter"/> under a prefix.
/// </summary>
internal static class T5TextGraphBuilder
{
    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The encoder hyper-parameters.</param>
    /// <param name="prefix">Canonical weight prefix of the encoder (<c>text_encoder_2.</c>).</param>
    /// <param name="name">Graph name.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, T5TextHyperParameters hp, string prefix, string name = DiffusionGraphs.TextEncode2)
    {
        var b = new GraphBuilder(name, weights);
        int rows = hp.ContextLength;
        int inner = hp.HeadCount * hp.HeadDim;
        var tokens = b.Input(DiffusionGraphs.Tokens, InputRole.TokenIds, DataType.I32, new Shape(rows));
        var x = b.Gather(b.Weight(prefix + "token_embedding.weight"), tokens);
        var shared = weights(prefix + "encoder.attention.relative_bias.weight") is { } table
            ? RelativePositionBias.FromBuckets(table, hp.MaxDistance, bidirectional: true)
            : null;
        for (int layer = 0; layer < hp.LayerCount; layer++)
        {
            using var _ = b.Group($"layers.{layer}");
            string p = $"{prefix}encoder.layers.{layer}.";
            var bias = weights(p + "attention.relative_bias.weight") is { } own ? RelativePositionBias.FromBuckets(own, hp.MaxDistance, bidirectional: true) : shared;
            var h = b.RmsNorm(x, b.Weight(p + "attention_norm.weight"), hp.NormEps);
            var q = b.Reshape(b.Linear(h, p + "attention.q.weight"), rows, hp.HeadCount, hp.HeadDim);
            var k = b.Reshape(b.Linear(h, p + "attention.k.weight"), rows, hp.HeadCount, hp.HeadDim);
            var v = b.Reshape(b.Linear(h, p + "attention.v.weight"), rows, hp.HeadCount, hp.HeadDim);
            var attended = b.Emit1("Attention", [q, k, v], Attributes.From([("scale", 1f), ("causal", false), .. bias?.Attributes ?? []]));
            x = b.Add(x, b.Linear(b.Reshape(attended, rows, inner), p + "attention.o.weight"));
            h = b.RmsNorm(x, b.Weight(p + "mlp_norm.weight"), hp.NormEps);
            var up = b.Linear(h, p + "mlp.up.weight");
            var activated = b.HasWeight(p + "mlp.gate.weight") ? b.Mul(Activate(b, b.Linear(h, p + "mlp.gate.weight"), hp.Activation), up) : Activate(b, up, hp.Activation);
            x = b.Add(x, b.Linear(activated, p + "mlp.down.weight"));
        }

        b.Output(DiffusionGraphs.Hidden, b.RmsNorm(x, b.Weight(prefix + "encoder.output_norm.weight"), hp.NormEps));
        return b.Build();
    }

    private static Value Activate(GraphBuilder b, Value x, string activation) => activation switch
    {
        "relu" => b.Unary("Relu", x),
        "gelu" => b.Unary("Gelu", x),
        "gelu_tanh" => b.Unary("GeluTanh", x),
        "silu" => b.Silu(x),
        _ => throw new NotSupportedException($"T5 text encoder: unknown activation '{activation}'."),
    };
}
