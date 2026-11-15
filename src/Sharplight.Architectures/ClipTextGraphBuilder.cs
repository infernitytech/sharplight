using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the text-encoder graph (<see cref="DiffusionGraphs.TextEncode"/>) of a CLIP text transformer: token and position embeddings,
/// pre-norm causal self-attention layers with a quick-GELU / GELU MLP, and the conditioning taken from the last (or a skipped-to
/// earlier) layer, optionally through the final LayerNorm. One padded prompt of the fixed context length per run.
/// </summary>
internal static class ClipTextGraphBuilder
{
    private const float QuickGeluScale = 1.702f;

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The encoder hyper-parameters.</param>
    /// <param name="prefix">Canonical weight prefix of the encoder (<c>text_encoder.</c>).</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, ClipTextHyperParameters hp, string prefix)
    {
        var b = new GraphBuilder(DiffusionGraphs.TextEncode, weights);
        int rows = hp.ContextLength;
        int headDim = hp.HiddenSize / hp.HeadCount;
        var tokens = b.Input(DiffusionGraphs.Tokens, InputRole.TokenIds, DataType.I32, new Shape(rows));
        var spans = new int[rows * 2];
        for (int row = 0; row < rows; row++)
        {
            spans[row * 2 + 1] = rows;
        }

        var span = b.Constant("prompt_spans", DataType.I32, new Shape(rows, 2), MemoryMarshal.AsBytes(spans.AsSpan()).ToArray());
        var x = b.Add(b.Gather(b.Weight(prefix + "token_embedding.weight"), tokens), b.Weight(prefix + "position_embedding.weight"));
        int lastLayer = hp.LayerCount - Math.Max(1, hp.ClipSkip);
        for (int layer = 0; layer <= lastLayer; layer++)
        {
            using var _ = b.Group($"layers.{layer}");
            string p = $"{prefix}layers.{layer}.";
            var h = DiffusionBlocks.LayerNorm(b, x, p + "norm1", hp.NormEps);
            var q = b.Reshape(b.Linear(h, p + "attn.q.weight", p + "attn.q.bias"), rows, hp.HeadCount, headDim);
            var k = b.Reshape(b.Linear(h, p + "attn.k.weight", p + "attn.k.bias"), rows, hp.HeadCount, headDim);
            var v = b.Reshape(b.Linear(h, p + "attn.v.weight", p + "attn.v.bias"), rows, hp.HeadCount, headDim);
            var attended = b.Emit1("Attention", [q, k, v, span], Attributes.From(("scale", 1f / MathF.Sqrt(headDim)), ("causal", true)));
            x = b.Add(x, b.Linear(b.Reshape(attended, rows, hp.HiddenSize), p + "attn.o.weight", p + "attn.o.bias"));
            var up = b.Linear(DiffusionBlocks.LayerNorm(b, x, p + "norm2", hp.NormEps), p + "mlp.up.weight", p + "mlp.up.bias");
            var activated = hp.Activation == "quick_gelu" ? b.Mul(up, b.Sigmoid(b.Scale(up, QuickGeluScale))) : b.Unary("Gelu", up);
            x = b.Add(x, b.Linear(activated, p + "mlp.down.weight", p + "mlp.down.bias"));
        }

        var hidden = hp.FinalNorm ? DiffusionBlocks.LayerNorm(b, x, prefix + "final_norm", hp.NormEps) : x;
        b.Output(DiffusionGraphs.Hidden, hidden);
        return b.Build();
    }
}
