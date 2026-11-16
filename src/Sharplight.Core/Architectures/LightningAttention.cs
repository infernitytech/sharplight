using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Lightning attention (MiniMax-01): a decay-only linear attention with a fixed decay per head, as a reusable token mixer. Per head the state
/// S [D, D] evolves as S = exp(g)·S + k ⊗ v and the output is Sᵀq without scaling (one <c>GatedDeltaRule</c> with the additive rule, a
/// time-invariant [heads] log decay and β = 1), followed by an RMS norm over all heads gated by sigmoid of a projection of the input.
/// </summary>
/// <remarks>
/// Hyper-parameters (below the architecture prefix): <c>lightning.head_count</c>, <c>lightning.head_dim</c>, <c>norm_eps</c> (epsilon of the
/// output norm). q, k and v pass through SiLU.
/// Weights below the layer prefix: <c>lightning.qkv.weight</c> [heads·3·D, hidden] (per head q, k and v rows in that order);
/// <c>lightning.decay</c> f32 [heads] (the log decay, minus the slope rate of the head); <c>lightning.norm.weight</c> [heads·D];
/// <c>lightning.gate.weight</c> [heads·D, hidden]; <c>lightning.out.weight</c> [hidden, heads·D].
/// </remarks>
internal sealed class LightningAttention
{
    private readonly int _heads;
    private readonly int _headDim;
    private readonly float _eps;

    /// <summary>Reads the <c>lightning.*</c> hyper-parameters of a source.</summary>
    public LightningAttention(ModelSource src)
    {
        _heads = src.Arch("lightning.head_count", 0);
        _headDim = src.Arch("lightning.head_dim", 0);
        _eps = src.Arch("norm_eps", 1e-5f);
    }

    /// <summary>Whether the layer with weight prefix <paramref name="prefix"/> uses this mixer.</summary>
    public static bool IsPresent(Func<string, WeightInfo?> weight, string prefix) => weight(prefix + "lightning.qkv.weight") is not null;

    /// <summary>Declares the recurrent state of one layer.</summary>
    /// <exception cref="InvalidDataException">The head count or size is missing.</exception>
    public StateDefinition DefineState(string group)
    {
        if (_heads <= 0 || _headDim <= 0)
        {
            throw new InvalidDataException("lightning.head_count and lightning.head_dim are required by lightning-attention layers.");
        }

        return RecurrentStates.Scan(group, _heads, _headDim, _headDim);
    }

    /// <summary>Emits the mixer of one layer on the normalized input <paramref name="h"/> [T, hidden]; returns [T, hidden].</summary>
    public Value Build(GraphBuilder b, Value h, string p, StateDefinition state)
    {
        int width = _heads * _headDim;
        string w = p + "lightning.";
        var qkv = b.Reshape(b.Silu(b.Linear(h, w + "qkv.weight")), 0, _heads, 3 * _headDim);                                // [T, H, 3·D]
        var q = b.Contiguous(b.Slice(qkv, 2, 0, _headDim));
        var k = b.Contiguous(b.Slice(qkv, 2, _headDim, _headDim));
        var v = b.Contiguous(b.Slice(qkv, 2, 2 * _headDim, _headDim));
        var ones = new float[_heads];
        Array.Fill(ones, 1f);
        var beta = b.Constant($"{p}lightning_beta", DataType.F32, new Shape(_heads), MemoryMarshal.AsBytes(ones.AsSpan()).ToArray());

        var recurrent = b.State(state);
        var o = b.Emit("GatedDeltaRule", [q, k, v, b.Weight(w + "decay"), beta, recurrent], Attributes.From(("scale", 1f), ("rule", "additive")),
            $"{p}lightning", writes: [recurrent])[0];
        var normed = b.RmsNorm(b.Reshape(o, 0, width), b.Weight(w + "norm.weight"), _eps);
        var gated = b.Mul(normed, b.Sigmoid(b.Linear(h, w + "gate.weight")), $"{p}lightning_gated");
        return b.Linear(gated, w + "out.weight", name: $"{p}lightning_out");
    }
}
