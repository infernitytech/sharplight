using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the encoder of a T5Gemma model as a text-encoder graph over one padded prompt: the token embeddings scaled by <c>√width</c>, Gemma 2 layers (RMS norms before and
/// after the attention and the feed-forward layer, rotary embeddings, soft-capped bidirectional attention that skips the padding keys, a GELU-gated feed-forward layer)
/// and the final norm. The norms' weights already hold Gemma's <c>1 + w</c>.
/// </summary>
/// <remarks>
/// Weights under the prefix: <c>token_embd.weight</c>, <c>layers.{i}.{attention_pre_norm, attention.{q,k,v,o}, attention_post_norm, mlp_pre_norm, mlp.{gate,up,down},
/// mlp_post_norm}.weight</c>, <c>norm.weight</c> and, with <c>padding</c>, the learned row that replaces the padding rows of the output.
/// </remarks>
internal static class T5GemmaEncoderGraph
{
    /// <summary>Builds the graph: tokens i32 <c>[rows]</c> and the prompt mask f32 <c>[rows]</c> (1 for prompt tokens, 0 for padding) to the rows f32 <c>[rows, width]</c>.</summary>
    /// <param name="name">The graph name.</param>
    /// <param name="weights">Finds a weight by name.</param>
    /// <param name="p">The encoder's hyper-parameters.</param>
    /// <param name="prefix">The weights' prefix (e.g. <c>text.</c>).</param>
    /// <param name="rows">The fixed prompt length.</param>
    /// <param name="tokensInput">The name of the token input.</param>
    /// <param name="maskInput">The name of the mask input.</param>
    /// <param name="hiddenOutput">The name of the output.</param>
    /// <param name="padding">The weight of the learned padding row (the padding rows of the output become it), or null to keep the encoder's rows.</param>
    public static ComputeGraph Build(string name, Func<string, WeightInfo?> weights, T5GemmaEncoderParameters p, string prefix, int rows, string tokensInput, string maskInput,
        string hiddenOutput, string? padding)
    {
        var b = new GraphBuilder(name, weights);
        var tokens = b.Input(tokensInput, InputRole.TokenIds, DataType.I32, new Shape(rows));
        var mask = b.Input(maskInput, InputRole.Other, DataType.F32, new Shape(rows));
        var positions = b.Constant("prompt_positions", DataType.I32, new Shape(rows, 1), MemoryMarshal.AsBytes(Enumerable.Range(0, rows).ToArray().AsSpan()).ToArray());
        var keyBias = b.Scale(mask, 1e9f, -1e9f);
        var rope = Attributes.From(("dims", (long)p.HeadDim), ("base", p.RopeBase), ("style", "neox"));
        var x = b.Mul(b.Gather(b.Weight(prefix + "token_embd.weight"), tokens), b.Constant("embedding_scale", DataType.F32, new Shape(1),
            MemoryMarshal.AsBytes(new[] { MathF.Sqrt(p.Hidden) }.AsSpan()).ToArray()));
        for (int il = 0; il < p.Layers; il++)
        {
            using var _ = b.Group($"{prefix}layers.{il}");
            string n = $"{prefix}layers.{il}.";
            var h = b.RmsNorm(x, b.Weight(n + "attention_pre_norm.weight"), p.NormEps);
            var q = b.Emit1("Rope", [Heads(b, h, n + "attention.q.weight", p.Heads, p.HeadDim), positions], rope);
            var k = b.Emit1("Rope", [Heads(b, h, n + "attention.k.weight", p.KvHeads, p.HeadDim), positions], rope);
            var v = Heads(b, h, n + "attention.v.weight", p.KvHeads, p.HeadDim);
            var attributes = new List<(string, object)> { ("scale", 1f / MathF.Sqrt(p.QueryScale)), ("causal", false) };
            if (p.Softcap > 0)
            {
                attributes.Add(("softcap", p.Softcap));
            }

            // A window at least as long as the prompt changes nothing (the GPU kernels do not combine windows with the padding mask).
            if (il < p.SlidingLayers.Length && p.SlidingLayers[il] != 0 && p.SlidingWindow > 0 && p.SlidingWindow < rows)
            {
                attributes.Add(("window", (long)p.SlidingWindow));
            }

            var attended = b.Emit1("Attention", [q, k, v, keyBias], Attributes.From([.. attributes]));
            var o = b.MatMul(b.Reshape(attended, 0, p.Heads * p.HeadDim), b.Weight(n + "attention.o.weight"));
            x = b.Add(x, b.RmsNorm(o, b.Weight(n + "attention_post_norm.weight"), p.NormEps));

            h = b.RmsNorm(x, b.Weight(n + "mlp_pre_norm.weight"), p.NormEps);
            var gated = b.Mul(b.Unary("GeluTanh", b.MatMul(h, b.Weight(n + "mlp.gate.weight"))), b.MatMul(h, b.Weight(n + "mlp.up.weight")));
            var mlp = b.MatMul(gated, b.Weight(n + "mlp.down.weight"));
            x = b.Add(x, b.RmsNorm(mlp, b.Weight(n + "mlp_post_norm.weight"), p.NormEps));
        }

        x = b.RmsNorm(x, b.Weight(prefix + "norm.weight"), p.NormEps);
        if (padding is not null)
        {
            // Padding rows take the learned padding row: x · mask + padding · (1 − mask).
            var column = b.Reshape(mask, rows, 1);
            x = b.Add(b.Mul(x, column), b.Mul(b.Reshape(b.Weight(padding), 1, p.Hidden), b.Scale(column, -1f, 1f)));
        }

        b.Output(hiddenOutput, x);
        return b.Build();
    }

    private static Value Heads(GraphBuilder b, Value x, string weight, int heads, int headDim) => b.Reshape(b.MatMul(x, b.Weight(weight)), 0, heads, headDim);
}
