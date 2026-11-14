using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Gated delta-rule linear attention with a decay per key channel (Kimi delta attention, KDA: Kimi-Linear, Ling 3 / Ring-linear) as a reusable
/// token mixer. Per head the state S [D, D] evolves as S = diag(exp(g))·S + k ⊗ (v − Sᵀk)·β and the output is Sᵀq / sqrt(D) (one
/// <c>GatedDeltaRule</c> with a [T, heads, D] decay), followed by a per-head RMS norm gated by sigmoid of a projection of the input.
/// </summary>
/// <remarks>
/// Hyper-parameters (below the architecture prefix): <c>linear_attention.head_count</c>, <c>linear_attention.head_dim</c> (key and value
/// head size), <c>linear_attention.conv_kernel</c>, <c>linear_attention.decay_bound</c> (0, the default: g = a · softplus(f + dt_bias); a
/// negative bound b: g = b · sigmoid(−a · (f + dt_bias)), Ling 3's bounded gate), <c>linear_attention.l2_eps</c> (epsilon of the q/k L2 norms,
/// default 1e-6), <c>norm_eps</c> (epsilon of the output norm).
/// Weights below the layer prefix: <c>linear_attention.{q,k,v}.weight</c> [heads·D, hidden]; <c>linear_attention.conv1d.weight</c> f32
/// [3·heads·D, kernel] (short causal convolutions of q, k and v, in that channel order, SiLU after); the decay input f either from
/// <c>linear_attention.decay.weight</c> [heads·D, hidden] or low-rank from <c>linear_attention.decay_down.weight</c> [r, hidden] and
/// <c>linear_attention.decay_up.weight</c> [heads·D, r]; <c>linear_attention.dt_bias</c> f32 [heads·D]; <c>linear_attention.a</c> f32 [heads]
/// (A = −exp(A_log)); <c>linear_attention.beta.weight</c> [heads, hidden] (β = sigmoid); the output gate from <c>linear_attention.gate.weight</c>
/// [heads·D, hidden] or <c>linear_attention.gate_down.weight</c> / <c>linear_attention.gate_up.weight</c>; <c>linear_attention.norm.weight</c> [D];
/// <c>linear_attention.out.weight</c> [hidden, heads·D].
/// </remarks>
internal sealed class ChannelDeltaAttention
{
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _kernel;
    private readonly float _decayBound;
    private readonly float _l2Eps;
    private readonly float _eps;

    /// <summary>Reads the <c>linear_attention.*</c> hyper-parameters of a source.</summary>
    public ChannelDeltaAttention(ModelSource src)
    {
        _heads = src.Arch("linear_attention.head_count", 0);
        _headDim = src.Arch("linear_attention.head_dim", 0);
        _kernel = src.Arch("linear_attention.conv_kernel", 4);
        _decayBound = src.Arch("linear_attention.decay_bound", 0f);
        _l2Eps = src.Arch("linear_attention.l2_eps", 1e-6f);
        _eps = src.Arch("norm_eps", 1e-6f);
    }

    /// <summary>Whether the layer with weight prefix <paramref name="prefix"/> uses this mixer.</summary>
    public static bool IsPresent(Func<string, WeightInfo?> weight, string prefix) => weight(prefix + "linear_attention.q.weight") is not null
        && weight(prefix + "linear_attention.dt_bias") is not null;

    /// <summary>Declares the convolution window and recurrent state of one layer.</summary>
    /// <exception cref="InvalidDataException">The head count or size is missing.</exception>
    public (StateDefinition Conv, StateDefinition State) DefineStates(string group)
    {
        if (_heads <= 0 || _headDim <= 0)
        {
            throw new InvalidDataException("linear_attention.head_count and linear_attention.head_dim are required by delta-attention layers.");
        }

        return (RecurrentStates.ConvWindow(group, _kernel - 1, 3 * _heads * _headDim), RecurrentStates.Scan(group, _heads, _headDim, _headDim));
    }

    /// <summary>Emits the mixer of one layer on the normalized input <paramref name="h"/> [T, hidden]; returns [T, hidden].</summary>
    public Value Build(GraphBuilder b, Value h, string p, StateDefinition conv, StateDefinition state)
    {
        int width = _heads * _headDim;
        string w = p + "linear_attention.";
        var qkv = b.Concat(1, b.Linear(h, w + "q.weight"), b.Linear(h, w + "k.weight"), b.Linear(h, w + "v.weight"));     // [T, 3·H·D]
        var convState = b.State(conv);
        var mixed = b.Emit("CausalConv1d", [qkv, b.Weight(w + "conv1d.weight"), convState], Attributes.From(("activation", "silu")), writes: [convState])[0];
        var q = b.L2Norm(b.Reshape(b.Slice(mixed, 1, 0, width), 0, _heads, _headDim), _l2Eps);
        var k = b.L2Norm(b.Reshape(b.Slice(mixed, 1, width, width), 0, _heads, _headDim), _l2Eps);
        var v = b.Reshape(b.Slice(mixed, 1, 2 * width, width), 0, _heads, _headDim);

        // Per-channel log decay [T, H, D].
        var f = b.Reshape(b.Add(Projection(b, h, w, "decay"), b.Weight(w + "dt_bias")), 0, _heads, _headDim);
        var a = b.Reshape(b.Weight(w + "a"), _heads, 1);
        var g = _decayBound == 0f
            ? b.Mul(b.Softplus(f), a)
            : b.Scale(b.Sigmoid(b.Mul(f, b.Scale(a, -1f))), _decayBound);
        var beta = b.Sigmoid(b.Linear(h, w + "beta.weight"));                                                              // [T, H]

        var recurrent = b.State(state);
        var o = b.Emit("GatedDeltaRule", [q, k, v, g, beta, recurrent], Attributes.From(("scale", 1f / MathF.Sqrt(_headDim))), $"{p}kda",
            writes: [recurrent])[0];
        var gate = b.Reshape(Projection(b, h, w, "gate"), 0, _heads, _headDim);
        var normed = b.Mul(b.RmsNorm(o, b.Weight(w + "norm.weight"), _eps), b.Sigmoid(gate), $"{p}kda_gated");
        return b.Linear(b.Reshape(normed, 0, width), w + "out.weight", name: $"{p}kda_out");
    }

    // A full-rank projection <kind>.weight or the low-rank pair <kind>_down / <kind>_up.
    private static Value Projection(GraphBuilder b, Value h, string w, string kind)
        => b.HasWeight(w + kind + ".weight") ? b.Linear(h, w + kind + ".weight") : b.Linear(b.Linear(h, w + kind + "_down.weight"), w + kind + "_up.weight");
}
