using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Multi-head latent attention (MLA, DeepSeek-V2 and successors) as a reusable building block. Keys and values are compressed
/// into one latent of <see cref="LatentAttentionParameters.LatentRank"/> dimensions per token plus a rotary key shared by all
/// heads; only that row is cached (<see cref="CreateCache"/>: [ctx, 1, latent + rope], instead of heads · (key + value) entries).
/// </summary>
/// <remarks>
/// Attention runs in the absorbed form for prefill and decode alike: each head's key up-projection is folded into its query
/// (q_latent = W_UK,hᵀ · q_nope through an expert-indexed product with one "expert" per head), so the scores are a plain
/// dot product of [q_latent | q_rope] with the cached [latent | k_rope] rows, a single-key-head (MQA) attention whose values are
/// the latent prefix of the same rows (<c>CachedAttention</c> with <c>v_dims</c>). The attended latent is mapped to each
/// head's value by W_UV,h and then by the output projection. The cache is an ordinary KV state, so the block works with
/// continuous batching (<see cref="MultiSequence"/>) and the slot cache layout unchanged.
/// Weights below the layer prefix: either <c>attention.q.weight</c> [H·(nope + rope), hidden] or the compressed query
/// <c>attention.q_down.weight</c> [q_rank, hidden], <c>attention.q_down_norm.weight</c> [q_rank], <c>attention.q_up.weight</c>
/// [H·(nope + rope), q_rank]; <c>attention.kv_down.weight</c> [latent + rope, hidden] (rows: latent, then the rotary key),
/// <c>attention.kv_down_norm.weight</c> [latent]; <c>attention.k_up.weight</c> [H, latent, nope]; <c>attention.v_up.weight</c>
/// [H, v, latent]; <c>attention.o.weight</c> [hidden, H·v]. Query rows per head are [nope | rope]. Optional: <c>attention.sinks</c> (f32 [H],
/// a learned per-head logit that joins the softmax denominator without a value) and <c>attention.gate.weight</c> [H·v, hidden] or [H, hidden] (sigmoid
/// output gate: every head's value is multiplied by sigmoid of this projection of the layer input, per channel or per head, before the output projection).
/// </remarks>
internal sealed class LatentAttention(LatentAttentionParameters parameters)
{
    /// <summary>The layer dimensions.</summary>
    public LatentAttentionParameters Parameters => parameters;

    /// <summary>The latent cache state of one layer: [ctx, 1, latent + rope] rows of normalized latent and rotated shared key.</summary>
    public StateDefinition CreateCache(string group, DataType type) => new()
    {
        Name = $"{group}.latent_cache",
        Type = AdapterOptions.KvRowType(type, parameters.CacheRowDims),
        Shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), 1, parameters.CacheRowDims),
        Kind = StateKind.KeyCache,
        Group = group,
    };

    /// <summary>
    /// The per-row head indices [T, H] (i32) the absorbed projections select their per-head matrices with; build once per graph
    /// from any [T, ...] input and share it between layers.
    /// </summary>
    public static Value HeadIndices(GraphBuilder b, Value positions, int heads)
    {
        var iota = new byte[heads * 4];
        for (int h = 0; h < heads; h++)
        {
            BitConverter.TryWriteBytes(iota.AsSpan(h * 4, 4), (float)h);
        }

        var zeros = b.Scale(b.Emit1("Cast", [b.Slice(positions, 1, 0, 1)], Attributes.From(("type", "f32"))), 0f);              // [T, 1]
        var rows = b.Add(zeros, b.Constant("latent_head_iota", DataType.F32, new Shape(heads), iota));                           // [T, H]
        return b.Emit1("Cast", [rows], Attributes.From(("type", "i32")), "latent_head_ids");
    }

    /// <summary>Emits the attention of one layer: writes the layer's cache rows and returns the output projection [T, hidden].</summary>
    /// <param name="b">Graph under construction.</param>
    /// <param name="h">Normalized layer input [T, hidden].</param>
    /// <param name="prefix">Weight name prefix of the layer (e.g. <c>layers.3.</c>).</param>
    /// <param name="cache">The layer's latent cache (<see cref="CreateCache"/>).</param>
    /// <param name="headIds">Head indices from <see cref="HeadIndices"/>.</param>
    /// <param name="rotate">Applies the layer's rotary embedding to a [T, heads, rope] tensor.</param>
    /// <param name="temperature">Optional per-row query factor [T, 1, 1] (<see cref="AttentionTemperature"/>), which scales the scores of a row.</param>
    public Value Build(GraphBuilder b, Value h, string prefix, StateDefinition cache, Value headIds, Func<Value, Value> rotate, Value? temperature = null)
    {
        var p = parameters;
        var q = Query(b, h, prefix);                                                                                       // [T, H, nope + rope]
        var queryNope = b.Slice(q, 2, 0, p.NopeDims);
        var queryRope = Magnitude(b, rotate(b.Slice(q, 2, p.NopeDims, p.RopeDims)));

        var compressed = b.Linear(h, prefix + "attention.kv_down.weight", name: $"{prefix}kv_down");                       // [T, latent + rope]
        var latent = b.RmsNorm(b.Slice(compressed, 1, 0, p.LatentRank), b.Weight(prefix + "attention.kv_down_norm.weight"), p.Eps);
        var keyRope = Magnitude(b, rotate(b.Reshape(b.Slice(compressed, 1, p.LatentRank, p.RopeDims), 0, 1, p.RopeDims)));
        var state = b.State(cache);
        b.CacheWrite(state, b.Concat(2, b.Reshape(latent, 0, 1, p.LatentRank), keyRope));

        // Absorbed query: per head W_UK,hᵀ · q_nope through an expert-indexed product (expert = head).
        var queryLatent = b.Emit1("MatMulId", [queryNope, b.Weight(prefix + "attention.k_up.weight"), headIds], null, $"{prefix}q_latent");
        var query = b.Concat(2, queryLatent, queryRope);                                                                   // [T, H, latent + rope]
        if (temperature is not null)
        {
            // The absorbed query is linear in the original one, so tempering it scales the scores exactly as tempering q would.
            query = b.Mul(query, temperature, $"{prefix}q_tempered");
        }

        b.RuntimeScalar(WellKnown.PastScalar);
        var attributes = Attributes.From(("scale", p.Scale), ("offset", WellKnown.PastScalar), ("v_dims", (long)p.LatentRank));
        Value[] inputs = [query, state, state];
        if (b.TryWeight(prefix + "attention.sinks") is { } sinks)
        {
            inputs = [query, state, state, sinks];
            attributes = attributes.With("sinks", true);
        }

        var attended = b.Emit1("CachedAttention", inputs, attributes, $"{prefix}attn");                                    // [T, H, latent]
        var values = b.Emit1("MatMulId", [attended, b.Weight(prefix + "attention.v_up.weight"), headIds], null, $"{prefix}attn_values"); // [T, H, v]
        if (b.TryWeight(prefix + "attention.gate.weight") is not null)
        {
            // One gate per value channel ([H·v, hidden]) or one per head ([H, hidden], Ling 3).
            bool perHead = b.TryWeight(prefix + "attention.gate.weight")!.Shape[0].Fixed == p.Heads;
            var gate = b.Reshape(b.Linear(h, prefix + "attention.gate.weight", name: $"{prefix}attn_gate"), 0, p.Heads, perHead ? 1 : p.ValueDims);
            values = b.Mul(values, b.Sigmoid(gate), $"{prefix}attn_gated");
        }

        return b.Linear(b.Reshape(values, 0, p.Heads * p.ValueDims), prefix + "attention.o.weight", name: $"{prefix}attn_out");
    }

    private Value Query(GraphBuilder b, Value h, string prefix)
    {
        var p = parameters;
        Value q;
        if (p.QueryRank > 0)
        {
            var down = b.RmsNorm(b.Linear(h, prefix + "attention.q_down.weight"), b.Weight(prefix + "attention.q_down_norm.weight"), p.Eps);
            q = b.Linear(down, prefix + "attention.q_up.weight", name: $"{prefix}q");
        }
        else
        {
            q = b.Linear(h, prefix + "attention.q.weight", name: $"{prefix}q");
        }

        return b.Reshape(q, 0, p.Heads, p.NopeDims + p.RopeDims);
    }

    private Value Magnitude(GraphBuilder b, Value rotated) => parameters.RopeMagnitude == 1f ? rotated : b.Scale(rotated, parameters.RopeMagnitude);
}
