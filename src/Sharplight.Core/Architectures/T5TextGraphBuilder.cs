using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the text-encoder graph of a T5 encoder (T5 v1.1 / Flan-T5, as FLUX and SD3 use it): token embeddings, pre-norm RMS
/// layers of bidirectional self-attention with T5's bucketed relative position bias (a [heads, L, L] bias gathered from the bucket
/// table, added to batched per-head score products) and a (gated) feed-forward, the final norm.
/// One padded prompt of the fixed context length per run; without a mask input the padding tokens take part (the FLUX / SD3
/// pipelines), with a mask input (the <c>maskInput</c> of <see cref="Build"/>: 1 for prompt tokens, 0 for padding) no token attends to the
/// padding (the Wan pipelines). Weights in the encoder-decoder vocabulary of <see cref="EncoderDecoderAdapter"/> under a prefix.
/// </summary>
internal static class T5TextGraphBuilder
{
    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The encoder hyper-parameters.</param>
    /// <param name="prefix">Canonical weight prefix of the encoder (<c>text_encoder_2.</c>).</param>
    /// <param name="name">Graph name.</param>
    /// <param name="maskInput">Name of an f32 [L] key-mask input (1 = token, 0 = padding), or <see langword="null"/> for none.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, T5TextHyperParameters hp, string prefix, string name = DiffusionGraphs.TextEncode2,
        string? maskInput = null)
    {
        var b = new GraphBuilder(name, weights);
        int rows = hp.ContextLength;
        var tokens = b.Input(DiffusionGraphs.Tokens, InputRole.TokenIds, DataType.I32, new Shape(rows));
        Value? keyMask = null;
        if (maskInput is not null)
        {
            // (mask − 1) · 10⁹: 0 for tokens, a large negative score for padding keys, broadcast over heads and query rows.
            var mask = b.Input(maskInput, InputRole.Other, DataType.F32, new Shape(rows));
            keyMask = b.Reshape(b.Scale(mask, 1e9f, -1e9f), 1, 1, rows);
        }

        var x = b.Gather(b.Weight(prefix + "token_embedding.weight"), tokens);
        b.Output(DiffusionGraphs.Hidden, Encode(b, x, keyMask, hp, prefix));
        return b.Build();
    }

    /// <summary>
    /// The encoder layers and the final norm over embedded rows <paramref name="x"/> [<see cref="T5TextHyperParameters.ContextLength"/>, hidden]
    /// (token embeddings, or the patch embeddings of a time-series encoder), or over a batch of sequences [batch (symbolic), rows, hidden] that attend
    /// within themselves.
    /// </summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="x">The embedded rows.</param>
    /// <param name="keyMask">An additive key bias [1, 1, rows] (0 for keys that take part, a large negative score for masked ones; [batch, 1, 1, rows] for a
    /// batch), or <see langword="null"/>.</param>
    /// <param name="hp">The encoder hyper-parameters (<see cref="T5TextHyperParameters.ContextLength"/>: the rows).</param>
    /// <param name="prefix">Canonical weight prefix of the encoder.</param>
    public static Value Encode(GraphBuilder b, Value x, Value? keyMask, T5TextHyperParameters hp, string prefix)
    {
        string sharedTable = prefix + "encoder.attention.relative_bias.weight";
        var sharedBias = b.HasWeight(sharedTable) ? WithKeyBias(b, PositionBias(b, sharedTable, hp, "relative_bias"), keyMask) : keyMask;
        for (int layer = 0; layer < hp.LayerCount; layer++)
        {
            using var _ = b.Group($"layers.{layer}");
            string p = $"{prefix}encoder.layers.{layer}.";
            // The key mask is folded into the position bias once (shared table) or per layer (per-layer tables).
            var bias = b.HasWeight(p + "attention.relative_bias.weight")
                ? WithKeyBias(b, PositionBias(b, p + "attention.relative_bias.weight", hp, $"relative_bias_{layer}"), keyMask)
                : sharedBias;

            var h = b.RmsNorm(x, b.Weight(p + "attention_norm.weight"), hp.NormEps);
            var attended = Attention(b, Linear(b, h, p + "attention.q.weight"), Linear(b, h, p + "attention.k.weight"), Linear(b, h, p + "attention.v.weight"), bias, hp);
            x = b.Add(x, Linear(b, attended, p + "attention.o.weight"));
            h = b.RmsNorm(x, b.Weight(p + "mlp_norm.weight"), hp.NormEps);
            var up = Linear(b, h, p + "mlp.up.weight");
            var activated = b.HasWeight(p + "mlp.gate.weight") ? b.Mul(Activate(b, Linear(b, h, p + "mlp.gate.weight"), hp.Activation), up) : Activate(b, up, hp.Activation);
            x = b.Add(x, Linear(b, activated, p + "mlp.down.weight"));
        }

        return b.RmsNorm(x, b.Weight(prefix + "encoder.output_norm.weight"), hp.NormEps);
    }

    private static Value WithKeyBias(GraphBuilder b, Value bias, Value? keyBias) => keyBias is null ? bias : b.Add(bias, keyBias);

    /// <summary>
    /// T5's bucketed relative position bias of the whole prompt, [heads, L, L] (query rows, key columns): the bucket table [buckets,
    /// heads] gathered by the constant bucket index of every key − query distance.
    /// </summary>
    private static Value PositionBias(GraphBuilder b, string table, T5TextHyperParameters hp, string name)
    {
        int rows = hp.ContextLength;
        var weight = b.Weight(table);
        int buckets = (int)weight.Shape[0].Fixed;
        var indices = new int[rows * rows];
        for (int query = 0; query < rows; query++)
        {
            for (int key = 0; key < rows; key++)
            {
                indices[query * rows + key] = RelativePositionBias.Bucket(key - query, buckets, hp.MaxDistance, bidirectional: true);
            }
        }

        var index = b.Constant(name + "_buckets", DataType.I32, new Shape((long)rows * rows), MemoryMarshal.AsBytes(indices.AsSpan()).ToArray());
        var gathered = b.Reshape(b.Gather(weight, index), rows, rows, hp.HeadCount);
        return b.Contiguous(b.Permute(gathered, 2, 0, 1));
    }

    /// <summary>
    /// Unscaled bidirectional attention with an additive per-head bias, written out as batched products over the heads (the bias
    /// table of T5-XXL, 64 heads × 257 distances, is too large for the attention kernels' constant tables).
    /// </summary>
    private static Value Attention(GraphBuilder b, Value q, Value k, Value v, Value? bias, T5TextHyperParameters hp)
    {
        int rows = hp.ContextLength;
        if (q.Rank == 3)
        {
            return BatchAttention(b, q, k, v, bias, hp);
        }

        Value Heads(Value x) => b.Contiguous(b.Permute(b.Reshape(x, rows, hp.HeadCount, hp.HeadDim), 1, 0, 2));
        var scores = b.Emit1("BatchMatMul", [Heads(q), Heads(k)], Attributes.From(("trans_b", true)));
        if (bias is not null)
        {
            scores = b.Add(scores, bias);
        }

        var probabilities = b.Emit1("Softmax", [scores]);
        var attended = b.Emit1("BatchMatMul", [probabilities, Heads(v)]);
        return b.Reshape(b.Contiguous(b.Permute(attended, 1, 0, 2)), rows, hp.HeadCount * hp.HeadDim);
    }

    /// <summary><see cref="Attention"/> over a batch of sequences [batch, rows, heads · head dim]; the bias broadcasts to [batch, heads, rows, rows].</summary>
    private static Value BatchAttention(GraphBuilder b, Value q, Value k, Value v, Value? bias, T5TextHyperParameters hp)
    {
        int rows = hp.ContextLength;
        Value Heads(Value x) => b.Contiguous(b.Permute(b.Reshape(x, 0, rows, hp.HeadCount, hp.HeadDim), 0, 2, 1, 3));
        var scores = b.Emit1("BatchMatMul", [Heads(q), Heads(k)], Attributes.From(("trans_b", true)));
        if (bias is not null)
        {
            scores = b.Add(scores, bias);
        }

        var probabilities = b.Emit1("Softmax", [scores]);
        var attended = b.Emit1("BatchMatMul", [probabilities, Heads(v)]);
        return b.Reshape(b.Contiguous(b.Permute(attended, 0, 2, 1, 3)), 0, rows, hp.HeadCount * hp.HeadDim);
    }

    /// <summary>
    /// A projection whose input stays f32 in the integer-dot kernels of quantized weights: T5 activations carry outliers of 10⁴ and
    /// more that int8 activation quantization flattens (a Q8_0 T5-XXL loses a fifth of its output otherwise).
    /// </summary>
    private static Value Linear(GraphBuilder b, Value x, string weight)
        => b.Emit1("MatMul", [x, b.Weight(weight)], Attributes.From((StandardOps.F32ActivationsAttribute, true)));

    private static Value Activate(GraphBuilder b, Value x, string activation) => activation switch
    {
        "relu" => b.Unary("Relu", x),
        "gelu" => b.Unary("Gelu", x),
        "gelu_tanh" => b.Unary("GeluTanh", x),
        "silu" => b.Silu(x),
        _ => throw new NotSupportedException($"T5 text encoder: unknown activation '{activation}'."),
    };
}
