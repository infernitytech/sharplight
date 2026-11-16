using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the MusicGen text encoder (a T5 encoder with bucketed relative position biases, followed by the projection into the decoder's width) and the decoder step
/// (the sum of the codebook embeddings plus the sinusoidal position, pre-norm layers of causal self-attention over a KV cache, cross-attention over the conditioning
/// rows and a GELU MLP, the final norm and one linear head per codebook) as graphs of generic ops. See <see cref="MusicGenerationGraphs"/>.
/// </summary>
internal static class MusicGenGraph
{
    /// <summary>Builds the text graph.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildText(MusicGenParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(MusicGenerationGraphs.TextGraph, weights);
        var rows = Dim.Sym(WellKnown.TokenSymbol);
        var tokens = b.Input(MusicGenerationGraphs.TokensInput, InputRole.TokenIds, DataType.I32, new Shape(rows));
        var x = b.Gather(b.Weight("text.token_embedding.weight"), tokens);
        var bias = RelativePositionBias.FromBuckets(weights("text.relative_bias.weight")!, p.TextMaxDistance, bidirectional: true);
        int inner = p.TextHeads * p.TextHeadDim;
        for (int il = 0; il < p.TextLayers; il++)
        {
            using var _ = b.Group($"text.layers.{il}");
            string n = $"text.layers.{il}.";
            var h = b.RmsNorm(x, b.Weight(n + "attention_norm.weight"), p.TextEps);
            var q = b.Reshape(b.Linear(h, n + "attention.q.weight"), 0, p.TextHeads, p.TextHeadDim);
            var k = b.Reshape(b.Linear(h, n + "attention.k.weight"), 0, p.TextHeads, p.TextHeadDim);
            var v = b.Reshape(b.Linear(h, n + "attention.v.weight"), 0, p.TextHeads, p.TextHeadDim);
            var attention = b.Emit1("Attention", [q, k, v], Attributes.From([("scale", 1f), ("causal", false), .. bias.Attributes]));
            x = b.Add(x, b.Linear(b.Reshape(attention, 0, inner), n + "attention.o.weight"));
            h = b.RmsNorm(x, b.Weight(n + "mlp_norm.weight"), p.TextEps);
            x = b.Add(x, b.Linear(b.Unary("Relu", b.Linear(h, n + "mlp.up.weight")), n + "mlp.down.weight"));
        }

        x = b.RmsNorm(x, b.Weight("text.output_norm.weight"), p.TextEps);
        b.Output(MusicGenerationGraphs.ConditionOutput, b.Linear(x, "text.projection.weight", "text.projection.bias"));
        return b.Build();
    }

    /// <summary>The cache states of the decoder's self-attention.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="options">Adapter options (KV cache type).</param>
    public static StateDefinition[] States(MusicGenParameters p, AdapterOptions options)
    {
        int headDim = p.Hidden / p.Heads;
        var states = new List<StateDefinition>();
        for (int il = 0; il < p.Layers; il++)
        {
            string g = $"decoder.layers.{il}";
            var shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), p.Heads, headDim);
            states.Add(new StateDefinition { Name = $"{g}.k_cache", Type = options.KvCacheTypeFor(headDim), Shape = shape, Kind = StateKind.KeyCache, Group = g });
            states.Add(new StateDefinition { Name = $"{g}.v_cache", Type = options.KvCacheTypeFor(headDim), Shape = shape, Kind = StateKind.ValueCache, Group = g });
        }

        return [.. states];
    }

    /// <summary>Builds a decoder step graph.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    /// <param name="states">The cache states of <see cref="States"/>.</param>
    /// <param name="conditioned">True for the graph with cross-attention, false for the unconditional branch (whose cross-attention output is zero).</param>
    public static ComputeGraph BuildStep(MusicGenParameters p, Func<string, WeightInfo?> weights, StateDefinition[] states, bool conditioned)
    {
        var b = new GraphBuilder(conditioned ? MusicGenerationGraphs.StepGraph : MusicGenerationGraphs.UnconditionalStepGraph, weights);
        var rows = Dim.Sym(WellKnown.TokenSymbol);
        int headDim = p.Hidden / p.Heads;
        float scale = 1f / MathF.Sqrt(headDim);
        Value x = null!;
        for (int q = 0; q < p.Codebooks; q++)
        {
            var codes = b.Input(MusicGenerationGraphs.CodeInput(q), InputRole.TokenIds, DataType.I32, new Shape(rows));
            var embedded = b.Gather(b.Weight($"decoder.embedding.{q}.weight"), codes);
            x = q == 0 ? embedded : b.Add(x, embedded);
        }

        var positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(rows));
        x = b.Add(x, b.Gather(b.Weight("decoder.position_table.weight"), positions));
        Value? condition = conditioned
            ? b.Input(MusicGenerationGraphs.ConditionInput, InputRole.EncoderOutput, DataType.F32, new Shape(Dim.Sym(WellKnown.EncoderSymbol), p.Hidden))
            : null;
        for (int il = 0; il < p.Layers; il++)
        {
            using var _ = b.Group($"decoder.layers.{il}");
            string n = $"decoder.layers.{il}.";
            var h = TowerBlocks.LayerNorm(b, x, n + "self_norm", p.Eps);
            var kc = b.State(states[2 * il]);
            var vc = b.State(states[(2 * il) + 1]);
            b.CacheWrite(kc, b.Reshape(b.Linear(h, n + "self_attention.k.weight"), 0, p.Heads, headDim));
            b.CacheWrite(vc, b.Reshape(b.Linear(h, n + "self_attention.v.weight"), 0, p.Heads, headDim));
            var attention = b.CachedAttention(b.Reshape(b.Linear(h, n + "self_attention.q.weight"), 0, p.Heads, headDim), kc, vc, scale);
            x = b.Add(x, b.Linear(b.Reshape(attention, 0, p.Hidden), n + "self_attention.o.weight"));
            if (condition is not null)
            {
                h = TowerBlocks.LayerNorm(b, x, n + "cross_norm", p.Eps);
                var q = b.Reshape(b.Linear(h, n + "cross_attention.q.weight"), 0, p.Heads, headDim);
                var k = b.Reshape(b.Linear(condition, n + "cross_attention.k.weight"), 0, p.Heads, headDim);
                var v = b.Reshape(b.Linear(condition, n + "cross_attention.v.weight"), 0, p.Heads, headDim);
                var cross = b.Emit1("Attention", [q, k, v], Attributes.From(("scale", scale), ("causal", false)));
                x = b.Add(x, b.Linear(b.Reshape(cross, 0, p.Hidden), n + "cross_attention.o.weight"));
            }

            h = TowerBlocks.LayerNorm(b, x, n + "mlp_norm", p.Eps);
            x = b.Add(x, b.Linear(b.Unary("Gelu", b.Linear(h, n + "mlp.up.weight")), n + "mlp.down.weight"));
        }

        x = TowerBlocks.LayerNorm(b, x, "decoder.output_norm", p.Eps);
        var heads = new Value[p.Codebooks];
        for (int q = 0; q < p.Codebooks; q++)
        {
            heads[q] = b.Linear(x, $"decoder.lm_head.{q}.weight");
        }

        b.Output(MusicGenerationGraphs.LogitsOutput, b.Concat(0, heads));
        return b.Build();
    }
}
