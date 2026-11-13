using Sharplight.Tensors;

namespace Sharplight.Graph;

// State-space (Mamba-2 / SSD) recurrence.
public static partial class StandardOps
{
    /// <summary>
    /// Mamba-2 selective scan (the recurrent form of the state-space duality), one call per layer.
    /// Inputs, in order: <c>x</c> [T, H, P] (post-conv, post-activation, split into H heads of P channels);
    /// <c>dt</c> [T, H] (raw time step, before bias and softplus); <c>dt_bias</c> [H]; <c>A</c> [H] (already negative, A = -exp(A_log));
    /// <c>B</c> [T, G, N]; <c>C</c> [T, G, N] (G groups, head h uses group h / (H/G)); <c>D</c> [H] (skip); <c>state</c> State [H, P, N] f32
    /// (the op reads it, advances it by the T tokens and writes it back: declare it in the node's writes). All f32.
    /// Per token t and head h: <c>dt' = max(softplus(dt[t,h] + dt_bias[h]), dt_min)</c>, <c>S[p,n] = exp(dt'·A[h])·S[p,n] + dt'·x[t,h,p]·B[t,g,n]</c>,
    /// <c>y[t,h,p] = Σₙ S[p,n]·C[t,g,n] + D[h]·x[t,h,p]</c>. Attribute <c>dt_min</c> (float, default 0): lower clamp of the discretised step
    /// (Nemotron-H: time_step_min). Output y [T, H, P] f32. The gated group RMS norm that follows in Mamba-2 blocks
    /// (<c>rmsnorm_groups(y·silu(z))·w</c>) is not part of the op: express it with Silu/Mul, a Reshape to [T, G, H·P/G] and an unweighted RmsNorm.
    /// </summary>
    public static readonly OpSchema SelectiveScan = new("SelectiveScan", OpCategory.Composite, 8, 8, (i, a) =>
    {
        var x = i[0].Shape;
        if (x.Rank != 3) throw new ArgumentException("SelectiveScan x must be [T, H, P].");
        return [new TensorType(DataType.F32, x)];
    });
}
