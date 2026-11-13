using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The Mamba-1 selective state-space mixer (Mamba, Falcon-Mamba, Jamba): input projection to (x, z), causal depthwise convolution with SiLU,
/// data-dependent (dt, B, C) from a low-rank projection of x, selective scan with a per-channel, per-state decay, SiLU gate and output projection.
/// Expressed with the <c>SelectiveScan</c> op in its [H, N] decay form (one head per channel, one row per head).
/// </summary>
/// <remarks>
/// Weights below a layer prefix <c>p</c>: <c>ssm.in.weight</c> [2·inner, hidden] (rows: x | z), <c>ssm.conv1d.weight</c> [inner, K] f32, <c>ssm.conv1d.bias</c> [inner] f32,
/// <c>ssm.x.weight</c> [dt_rank + 2·N, inner] (rows: dt | B | C), <c>ssm.dt.weight</c> [inner, dt_rank], <c>ssm.dt_bias</c> [inner] f32, <c>ssm.a</c> [inner, N] f32 (already negative),
/// <c>ssm.d</c> [inner] f32, <c>ssm.out.weight</c> [hidden, inner]; optional RMS norms (Jamba, Falcon-Mamba) <c>ssm.dt_norm.weight</c> [dt_rank], <c>ssm.b_norm.weight</c> [N],
/// <c>ssm.c_norm.weight</c> [N]. Hyper-parameters: <c>ssm.conv_kernel</c>; the sizes come from the weight shapes.
/// </remarks>
internal sealed class Mamba1Layer
{
    private readonly int _convKernel;
    private readonly float _eps;

    /// <summary>Reads <c>ssm.conv_kernel</c> and the norm epsilon.</summary>
    public Mamba1Layer(ModelSource src, float eps)
    {
        _convKernel = src.Arch("ssm.conv_kernel", 4);
        _eps = eps;
    }

    /// <summary>Whether the layer with weight prefix <paramref name="p"/> is a Mamba-1 layer (it has the low-rank (dt, B, C) projection).</summary>
    public static bool Is(ModelSource src, string p) => src.Weight(p + "ssm.x.weight") is not null;

    /// <summary>Declares the per-layer recurrent states for the layer weights found under <paramref name="p"/>.</summary>
    public (StateDefinition Conv, StateDefinition Scan) DefineStates(ModelSource src, string group, string p)
    {
        var a = src.Weight(p + "ssm.a")!.Shape;
        int inner = (int)a[0].Fixed;
        int stateSize = (int)a[1].Fixed;
        var conv = new StateDefinition { Name = $"{group}.conv_state", Type = DataType.F32, Shape = new Shape(_convKernel - 1, inner), Kind = StateKind.Recurrent, Group = group };
        var scan = new StateDefinition { Name = $"{group}.ssm_state", Type = DataType.F32, Shape = new Shape(inner, 1, stateSize), Kind = StateKind.Recurrent, Group = group };
        return (conv, scan);
    }

    /// <summary>Emits the mixer on the (already normalized) input <paramref name="h"/> [T, hidden].</summary>
    public Value Build(GraphBuilder b, Value h, string p, StateDefinition convState, StateDefinition scanState)
    {
        long inner = b.Weight(p + "ssm.a").Shape[0].Fixed;
        long stateSize = b.Weight(p + "ssm.a").Shape[1].Fixed;
        long dtRank = b.Weight(p + "ssm.dt.weight").Shape[1].Fixed;
        var proj = b.Linear(h, p + "ssm.in.weight");                                    // [T, 2·inner]
        var x = b.Slice(proj, 1, 0, inner);
        var z = b.Slice(proj, 1, inner, inner);
        var cs = b.State(convState);
        var ss = b.State(scanState);
        var conv = b.Emit("CausalConv1d", [x, b.Weight(p + "ssm.conv1d.weight"), cs, b.Weight(p + "ssm.conv1d.bias")],
            Attributes.From(("activation", "silu"), ("bias", true)), writes: [cs])[0];
        var dbc = b.Linear(conv, p + "ssm.x.weight");                                   // [T, dt_rank + 2N]
        var dtLow = Normed(b, b.Slice(dbc, 1, 0, dtRank), p + "ssm.dt_norm.weight");
        var bs = Normed(b, b.Slice(dbc, 1, dtRank, stateSize), p + "ssm.b_norm.weight");
        var cc = Normed(b, b.Slice(dbc, 1, dtRank + stateSize, stateSize), p + "ssm.c_norm.weight");
        var dt = b.Linear(dtLow, p + "ssm.dt.weight");                                  // [T, inner]; the bias and softplus are part of the scan
        var scan = b.Emit("SelectiveScan",
            [b.Reshape(conv, 0, (int)inner, 1), dt, b.Weight(p + "ssm.dt_bias"), b.Weight(p + "ssm.a"),
             b.Reshape(bs, 0, 1, (int)stateSize), b.Reshape(cc, 0, 1, (int)stateSize), b.Weight(p + "ssm.d"), ss],
            Attributes.From(("dt_min", 0f)), $"{p}scan", writes: [ss])[0];              // [T, inner, 1]
        var gated = b.Mul(b.Reshape(scan, 0, (int)inner), b.Silu(z));
        return b.Linear(gated, p + "ssm.out.weight", name: $"{p}ssm_out");
    }

    private Value Normed(GraphBuilder b, Value v, string weightName) => b.TryWeight(weightName) is { } w ? b.RmsNorm(v, w, _eps) : v;
}
