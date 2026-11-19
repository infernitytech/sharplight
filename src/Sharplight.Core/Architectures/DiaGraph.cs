using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits Dia's text encoder (a bidirectional pre-norm transformer over byte tokens: rotary attention without score scaling, SwiGLU) and the decoder step (the sum of the
/// codebook embeddings, pre-norm layers of causal grouped-query self-attention over a KV cache, cross-attention over the encoder rows whose keys and values are projected once
/// per source into states, SwiGLU, the final norm and one head over all codebooks) as graphs of generic ops.
/// </summary>
internal static class DiaGraph
{
    /// <summary>Builds the text encoder graph.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildText(DiaParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(DiaGraphs.TextGraph, weights);
        var rows = Dim.Sym(WellKnown.TokenSymbol);
        var tokens = b.Input(DiaGraphs.TokensInput, InputRole.TokenIds, DataType.I32, new Shape(rows));
        var positions = b.Input(DiaGraphs.PositionsInput, InputRole.Positions, DataType.I32, new Shape(rows, 1));
        var x = b.Gather(b.Weight("encoder.token_embedding.weight"), tokens);
        int inner = p.EncoderHeads * p.EncoderHeadDim;
        for (int il = 0; il < p.EncoderLayers; il++)
        {
            using var _ = b.Group($"encoder.layers.{il}");
            string n = $"encoder.layers.{il}.";
            var h = b.RmsNorm(x, b.Weight(n + "attention_norm.weight"), p.Eps);
            var q = Rotate(b, b.Reshape(b.Linear(h, n + "attention.q.weight"), 0, p.EncoderHeads, p.EncoderHeadDim), positions, p);
            var k = Rotate(b, b.Reshape(b.Linear(h, n + "attention.k.weight"), 0, p.EncoderHeads, p.EncoderHeadDim), positions, p);
            var v = b.Reshape(b.Linear(h, n + "attention.v.weight"), 0, p.EncoderHeads, p.EncoderHeadDim);
            var attention = b.Emit1("Attention", [q, k, v], Attributes.From(("scale", 1f), ("causal", false)));
            x = b.Add(x, b.Linear(b.Reshape(attention, 0, inner), n + "attention.o.weight"));
            h = b.RmsNorm(x, b.Weight(n + "mlp_norm.weight"), p.Eps);
            x = b.Add(x, FeedForward(b, h, n));
        }

        b.Output(DiaGraphs.ConditionOutput, b.RmsNorm(x, b.Weight("encoder.output_norm.weight"), p.Eps));
        return b.Build();
    }

    /// <summary>The states of the decoder: the self-attention caches and the cross-attention keys and values of the current source.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="options">Adapter options (KV cache type).</param>
    public static StateDefinition[] States(DiaParameters p, AdapterOptions options)
    {
        var states = new List<StateDefinition>();
        for (int il = 0; il < p.Layers; il++)
        {
            string g = $"decoder.layers.{il}";
            var shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), p.KvHeads, p.HeadDim);
            states.Add(new StateDefinition { Name = $"{g}.k_cache", Type = options.KvCacheTypeFor(p.HeadDim), Shape = shape, Kind = StateKind.KeyCache, Group = g });
            states.Add(new StateDefinition { Name = $"{g}.v_cache", Type = options.KvCacheTypeFor(p.HeadDim), Shape = shape, Kind = StateKind.ValueCache, Group = g });
            var cross = new Shape(p.MaxText, p.CrossHeads, p.CrossHeadDim);
            states.Add(new StateDefinition { Name = $"{g}.cross_k", Type = DataType.F32, Shape = cross, Kind = StateKind.Other, Group = g });
            states.Add(new StateDefinition { Name = $"{g}.cross_v", Type = DataType.F32, Shape = cross, Kind = StateKind.Other, Group = g });
        }

        return [.. states];
    }

    /// <summary>Builds the decoder step graph.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    /// <param name="states">The states of <see cref="States"/>.</param>
    public static ComputeGraph BuildStep(DiaParameters p, Func<string, WeightInfo?> weights, StateDefinition[] states)
    {
        var b = new GraphBuilder(DiaGraphs.StepGraph, weights);
        var rows = Dim.Sym(WellKnown.TokenSymbol);
        Value x = null!;
        for (int c = 0; c < p.Channels; c++)
        {
            var codes = b.Input(DiaGraphs.CodeInput(c), InputRole.TokenIds, DataType.I32, new Shape(rows));
            var embedded = b.Gather(b.Weight("decoder.embedding.weight"), codes);
            x = c == 0 ? embedded : b.Add(x, embedded);
        }

        var positions = b.Input(DiaGraphs.PositionsInput, InputRole.Positions, DataType.I32, new Shape(rows, 1));
        var condition = b.Input(DiaGraphs.ConditionOutput, InputRole.EncoderOutput, DataType.F32, new Shape(Dim.Sym(WellKnown.EncoderSymbol), p.EncoderHidden));
        var spans = b.Input(DiaGraphs.SpansInput, InputRole.RowSpans, DataType.I32, new Shape(rows, 2));
        int inner = p.Heads * p.HeadDim;
        int crossInner = p.CrossHeads * p.CrossHeadDim;
        for (int il = 0; il < p.Layers; il++)
        {
            using var _ = b.Group($"decoder.layers.{il}");
            string n = $"decoder.layers.{il}.";
            var h = b.RmsNorm(x, b.Weight(n + "attention_norm.weight"), p.Eps);
            var kc = b.State(states[4 * il]);
            var vc = b.State(states[(4 * il) + 1]);
            b.CacheWrite(kc, Rotate(b, b.Reshape(b.Linear(h, n + "attention.k.weight"), 0, p.KvHeads, p.HeadDim), positions, p));
            b.CacheWrite(vc, b.Reshape(b.Linear(h, n + "attention.v.weight"), 0, p.KvHeads, p.HeadDim));
            var query = Rotate(b, b.Reshape(b.Linear(h, n + "attention.q.weight"), 0, p.Heads, p.HeadDim), positions, p);
            var attention = b.CachedAttention(query, kc, vc, 1f);
            x = b.Add(x, b.Linear(b.Reshape(attention, 0, inner), n + "attention.o.weight"));

            h = b.RmsNorm(x, b.Weight(n + "cross_attention_norm.weight"), p.Eps);
            var ck = b.State(states[(4 * il) + 2]);
            var cv = b.State(states[(4 * il) + 3]);
            b.CacheWrite(ck, b.Reshape(b.Linear(condition, n + "cross_attention.k.weight"), 0, p.CrossHeads, p.CrossHeadDim), WellKnown.SourcePastScalar);
            b.CacheWrite(cv, b.Reshape(b.Linear(condition, n + "cross_attention.v.weight"), 0, p.CrossHeads, p.CrossHeadDim), WellKnown.SourcePastScalar);
            var cq = b.Reshape(b.Linear(h, n + "cross_attention.q.weight"), 0, p.CrossHeads, p.CrossHeadDim);
            var cross = b.Emit1("Attention", [cq, ck, cv, spans], Attributes.From(("scale", 1f), ("causal", false)));
            x = b.Add(x, b.Linear(b.Reshape(cross, 0, crossInner), n + "cross_attention.o.weight"));

            h = b.RmsNorm(x, b.Weight(n + "mlp_norm.weight"), p.Eps);
            x = b.Add(x, FeedForward(b, h, n));
        }

        x = b.RmsNorm(x, b.Weight("decoder.output_norm.weight"), p.Eps);
        b.Output(DiaGraphs.LogitsOutput, b.Linear(x, "decoder.lm_head.weight"));
        return b.Build();
    }

    private static Value Rotate(GraphBuilder b, Value heads, Value positions, DiaParameters p)
        => b.Emit1("Rope", [heads, positions], Attributes.From(("dims", (long)heads.Shape[2].Fixed), ("base", p.RopeBase), ("style", "neox")));

    private static Value FeedForward(GraphBuilder b, Value h, string n)
        => b.Linear(b.Mul(b.Silu(b.Linear(h, n + "mlp.gate.weight")), b.Linear(h, n + "mlp.up.weight")), n + "mlp.down.weight");
}
