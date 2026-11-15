using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The <see cref="WellKnown.ImageEncodeGraph"/> of vision transformers over the host-patchified image: patch embedding, optional class and register
/// tokens, position table, optional pre-norm (<c>embedding_norm</c>), the layer stack, <c>output_norm</c> on every row, then the pooling and an optional
/// <c>projection.weight</c> into the embedding space (CLIP). Weights carry the tower prefix of <see cref="TowerOptions.Prefix"/>:
/// <c>patch_embed.{weight,bias}</c>, optional <c>class_embedding</c>, <c>register_embedding.weight</c>, <c>position_embedding.weight</c>
/// ([class row +] grid² rows), and for attention pooling <c>head.probe</c>, <c>head.attention.{q,k,v,o}</c>, <c>head.norm</c>, <c>head.mlp.{up,down}</c>.
/// Outputs <see cref="WellKnown.PooledOutput"/> and, with a <c>classifier.out.weight</c>, <see cref="WellKnown.LogitsOutput"/>.
/// </summary>
internal static class ImageTowerGraph
{
    /// <summary>Builds the graph.</summary>
    /// <param name="name">The graph name.</param>
    /// <param name="src">The model source (weights).</param>
    /// <param name="o">The tower hyper-parameters.</param>
    public static ComputeGraph Build(string name, ModelSource src, ImageTowerOptions o)
    {
        var t = o.Tower;
        string p = t.Prefix;
        int grid = o.ImageSize / o.Patch;
        int n = grid * grid;
        var b = new GraphBuilder(name, src.Weight);
        var patches = b.Input("patches", InputRole.Media, DataType.F32, new Shape(n, 3 * o.Patch * o.Patch));
        bool classToken = b.HasWeight(p + "class_embedding");
        Value x;
        using (b.Group(p + "embedding"))
        {
            x = b.Linear(patches, p + "patch_embed.weight", p + "patch_embed.bias", "patch_embd");
            if (classToken)
            {
                x = b.Concat(0, b.Reshape(b.Weight(p + "class_embedding"), 1, t.Hidden), x);
            }

            x = b.Add(x, b.Weight(p + "position_embedding.weight"), "inp_pos_embd");
            if (o.Registers > 0)
            {
                x = b.Concat(0, b.Slice(x, 0, 0, 1), b.Weight(p + "register_embedding.weight"), b.Slice(x, 0, 1, n));
            }

            if (b.HasWeight(p + "embedding_norm.weight"))
            {
                x = TowerBlocks.LayerNorm(b, x, p + "embedding_norm", t.Eps);
            }
        }

        int rows = (classToken ? 1 : 0) + o.Registers + n;
        x = TowerBlocks.Layers(b, x, t, rows, causal: false);
        using (b.Group(p + "output"))
        {
            x = TowerBlocks.LayerNorm(b, x, p + "output_norm", t.Eps);
            Value pooled;
            switch (o.Pooling)
            {
                case "mean":
                {
                    var patchRows = b.Slice(x, 0, rows - n, n);
                    var sum = b.Emit1("ReduceSum", [b.Contiguous(b.Permute(patchRows, 1, 0))]);
                    pooled = b.Reshape(b.Scale(sum, 1f / n), 1, t.Hidden);
                    break;
                }

                case "attention_head":
                    pooled = AttentionPool(b, x, t, rows);
                    break;
                default:
                    pooled = b.Slice(x, 0, 0, 1);
                    break;
            }

            if (b.HasWeight(p + "projection.weight"))
            {
                pooled = b.Linear(pooled, p + "projection.weight", p + "projection.bias");
            }

            b.Output(WellKnown.PooledOutput, pooled);
            if (b.HasWeight(p + "classifier.out.weight"))
            {
                b.Output(WellKnown.LogitsOutput, b.Linear(pooled, p + "classifier.out.weight", p + "classifier.out.bias", "logits"));
            }
        }

        return b.Build();
    }

    // SigLIP's pooling head: one learned probe attends over all rows, then a residual MLP behind a LayerNorm.
    private static Value AttentionPool(GraphBuilder b, Value x, TowerOptions t, int rows)
    {
        string p = t.Prefix + "head.";
        var probe = b.Reshape(b.Weight(p + "probe"), 1, t.Hidden);
        var q = b.Reshape(b.Linear(probe, p + "attention.q.weight", p + "attention.q.bias"), 1, t.Heads, t.HeadSize);
        var k = b.Reshape(b.Linear(x, p + "attention.k.weight", p + "attention.k.bias"), rows, t.Heads, t.HeadSize);
        var v = b.Reshape(b.Linear(x, p + "attention.v.weight", p + "attention.v.bias"), rows, t.Heads, t.HeadSize);
        var attn = b.Emit1("Attention", [q, k, v], Attributes.From(("scale", 1f / MathF.Sqrt(t.HeadSize)), ("causal", false)));
        var residual = b.Linear(b.Reshape(attn, 1, t.Hidden), p + "attention.o.weight", p + "attention.o.bias");
        var h = TowerBlocks.LayerNorm(b, residual, p + "norm", t.Eps);
        var mlp = b.Linear(TowerBlocks.Activation(b, b.Linear(h, p + "mlp.up.weight", p + "mlp.up.bias"), t.Activation), p + "mlp.down.weight", p + "mlp.down.bias");
        return b.Add(residual, mlp);
    }
}
