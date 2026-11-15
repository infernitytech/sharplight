using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The <see cref="WellKnown.TextEncodeGraph"/> of the text tower of a dual-encoder model (CLIP, SigLIP): a fixed number of token rows (the host pads),
/// token and position embedding, the layer stack (causal or bidirectional), <c>final_norm</c> and the row at the input <c>pool_index</c> (the end-of-text
/// token for CLIP, the last row for SigLIP) through an optional <c>projection.{weight,bias}</c>. Weights carry the <c>text.</c> prefix:
/// <c>token_embedding.weight</c>, <c>position_embedding.weight</c>. Outputs <see cref="WellKnown.PooledOutput"/> and, when the weights exist, the
/// zero-shot scale <c>logit_scale</c> (already exponentiated) and <c>logit_bias</c>.
/// </summary>
internal static class TextTowerGraph
{
    /// <summary>Output name of the zero-shot logit scale.</summary>
    public const string LogitScaleOutput = "logit_scale";

    /// <summary>Output name of the zero-shot logit bias.</summary>
    public const string LogitBiasOutput = "logit_bias";

    /// <summary>Name of the input holding the row index to pool.</summary>
    public const string PoolIndexInput = "pool_index";

    /// <summary>Builds the graph.</summary>
    /// <param name="src">The model source (weights).</param>
    /// <param name="t">The transformer layer stack.</param>
    /// <param name="length">The number of token rows.</param>
    /// <param name="causal">Whether attention is causal.</param>
    public static ComputeGraph Build(ModelSource src, TowerOptions t, int length, bool causal)
    {
        string p = t.Prefix;
        var b = new GraphBuilder(WellKnown.TextEncodeGraph, src.Weight);
        var tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(length));
        var poolIndex = b.Input(PoolIndexInput, InputRole.Other, DataType.I32, new Shape(1));
        Value x;
        using (b.Group(p + "embedding"))
        {
            x = b.Gather(b.Weight(p + "token_embedding.weight"), tokens, "inp_embd");
            x = b.Add(x, b.Slice(b.Weight(p + "position_embedding.weight"), 0, 0, length), "pos_embd");
        }

        x = TowerBlocks.Layers(b, x, t, length, causal);
        using (b.Group(p + "output"))
        {
            x = TowerBlocks.LayerNorm(b, x, p + "final_norm", t.Eps);
            var pooled = b.Gather(x, poolIndex, "pooled_row");
            if (b.HasWeight(p + "projection.weight"))
            {
                pooled = b.Linear(pooled, p + "projection.weight", p + "projection.bias");
            }

            b.Output(WellKnown.PooledOutput, pooled);
            if (b.HasWeight(p + "logit_scale"))
            {
                b.Output(LogitScaleOutput, b.Unary("Exp", b.Reshape(b.Weight(p + "logit_scale"), 1)));
            }

            if (b.HasWeight(p + "logit_bias"))
            {
                b.Output(LogitBiasOutput, b.Reshape(b.Weight(p + "logit_bias"), 1));
            }
        }

        return b.Build();
    }
}
