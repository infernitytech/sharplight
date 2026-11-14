using Sharplight.Tensors;

namespace Sharplight.Graph;

// Linear attention recurrences over batched sequences with explicit state in and out.
public static partial class StandardOps
{
    /// <summary>
    /// Batched linear attention recurrence (linear / gated / delta / gated delta rule) with the state passed in and out as tensors.
    /// Inputs: q [B, T, Hq·Dk], k [B, T, Hkv·Dk], v [B, T, Hkv·Dv]; then, in this order and only when the attribute of the same name is
    /// set, <c>state</c> [B, Hkv, Dk, Dv] (else zeros), <c>gated</c>: decay [B, T, Hkv] or [B, T, Hkv·Dk] (log decay, per head or per key
    /// dim), <c>delta</c>: beta [B, T, Hkv] or [B, T, 1]. Attributes: <c>q_heads</c>, <c>kv_heads</c> (query head h reads key/value head
    /// h / (Hq/Hkv)), <c>scale</c>. Per token: S = exp(g)·S (gated); v' = beta·(v − Sᵀk) (delta, else v' = v); S += k ⊗ v';
    /// o = scale·Sᵀq. Outputs: o [B, T, Hq·Dv] and the final state [B, Hkv, Dk, Dv], all f32.
    /// </summary>
    public static readonly OpSchema LinearAttention = new("LinearAttention", OpCategory.Composite, 3, 6, (i, a) =>
    {
        var q = i[0].Shape;
        var k = i[1].Shape;
        var v = i[2].Shape;
        long qHeads = a.Get<long>("q_heads");
        long kvHeads = a.Get<long>("kv_heads");
        if (q.Rank != 3 || k.Rank != 3 || v.Rank != 3 || qHeads <= 0 || kvHeads <= 0 || qHeads % kvHeads != 0)
        {
            throw new ArgumentException($"LinearAttention expects q/k/v [B, T, H·D] and q_heads a multiple of kv_heads, got q{q} k{k} v{v}.");
        }

        long dk = k[2].Fixed / kvHeads;
        long dv = v[2].Fixed / kvHeads;
        return [new(DataType.F32, new Shape(q[0], q[1], qHeads * dv)), new(DataType.F32, new Shape(q[0], kvHeads, dk, dv))];
    });
}
