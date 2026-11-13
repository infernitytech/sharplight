using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The PLaMo-2 state-space mixer of the <see cref="HybridAdapter"/>: a Mamba-2 style scan with one decay per head and a single B / C group, whose input
/// projection stores the gate and the scan input interleaved per head, whose causal convolution covers the scan input only, and whose
/// (dt, B, C) come from a projection of the convolved input followed by RMS norms.
/// </summary>
/// <remarks>
/// Weights below a layer prefix <c>p</c>: <c>ssm.in.weight</c> [2·inner, hidden] (per head: gate z (P rows), then scan input x (P rows)), <c>ssm.conv1d.weight</c> [inner, K] f32
/// (no bias), <c>ssm.bcdt.weight</c> [2·N + R, inner] (rows: B | C | dt), <c>ssm.dt_norm.weight</c> [R], <c>ssm.b_norm.weight</c> [N], <c>ssm.c_norm.weight</c> [N],
/// <c>ssm.dt.weight</c> [H, R] with <c>ssm.dt_bias</c> [H], <c>ssm.a</c> [H] f32 (already negative), <c>ssm.d</c> [H] f32, <c>ssm.out.weight</c> [hidden, inner]; inner = H·P.
/// The output is <c>silu(z) · scan(x)</c>. Hyper-parameter: <c>ssm.conv_kernel</c>; the sizes come from the weight shapes.
/// </remarks>
internal sealed class Plamo2Layer
{
    private readonly int _convKernel;
    private readonly float _eps;

    /// <summary>Reads <c>ssm.conv_kernel</c> and the norm epsilon.</summary>
    public Plamo2Layer(ModelSource src, float eps)
    {
        _convKernel = src.Arch("ssm.conv_kernel", 4);
        _eps = eps;
    }

    /// <summary>Whether the layer with weight prefix <paramref name="p"/> has the PLaMo-2 mixer (it has the joint (B, C, dt) projection).</summary>
    public static bool Is(ModelSource src, string p) => src.Weight(p + "ssm.bcdt.weight") is not null;

    /// <summary>Declares the per-layer recurrent states for the layer weights found under <paramref name="p"/>.</summary>
    public (StateDefinition Conv, StateDefinition Scan) DefineStates(ModelSource src, string group, string p)
    {
        int heads = (int)src.Weight(p + "ssm.a")!.Shape[0].Fixed;
        int inner = (int)src.Weight(p + "ssm.conv1d.weight")!.Shape[0].Fixed;
        int stateSize = (int)src.Weight(p + "ssm.b_norm.weight")!.Shape[0].Fixed;
        var conv = new StateDefinition { Name = $"{group}.conv_state", Type = DataType.F32, Shape = new Shape(_convKernel - 1, inner), Kind = StateKind.Recurrent, Group = group };
        var scan = new StateDefinition { Name = $"{group}.ssm_state", Type = DataType.F32, Shape = new Shape(heads, inner / heads, stateSize), Kind = StateKind.Recurrent, Group = group };
        return (conv, scan);
    }

    /// <summary>Emits the mixer on the (already normalized) input <paramref name="h"/> [T, hidden].</summary>
    public Value Build(GraphBuilder b, Value h, string p, StateDefinition convState, StateDefinition scanState)
    {
        int heads = (int)b.Weight(p + "ssm.a").Shape[0].Fixed;
        int inner = (int)b.Weight(p + "ssm.conv1d.weight").Shape[0].Fixed;
        int headDim = inner / heads;
        int stateSize = (int)b.Weight(p + "ssm.b_norm.weight").Shape[0].Fixed;
        int dtRank = (int)b.Weight(p + "ssm.dt_norm.weight").Shape[0].Fixed;
        var perHead = b.Reshape(b.Linear(h, p + "ssm.in.weight"), 0, heads, 2 * headDim);                // [T, H, z | x]
        var z = b.Reshape(b.Slice(perHead, 2, 0, headDim), 0, inner);
        var x = b.Reshape(b.Slice(perHead, 2, headDim, headDim), 0, inner);
        var cs = b.State(convState);
        var ss = b.State(scanState);
        var conv = b.Emit("CausalConv1d", [x, b.Weight(p + "ssm.conv1d.weight"), cs], Attributes.From(("activation", "silu")), writes: [cs])[0];
        var bcdt = b.Linear(conv, p + "ssm.bcdt.weight");                                                // [T, 2N + R]
        var bs = b.RmsNorm(b.Slice(bcdt, 1, 0, stateSize), b.Weight(p + "ssm.b_norm.weight"), _eps);
        var cc = b.RmsNorm(b.Slice(bcdt, 1, stateSize, stateSize), b.Weight(p + "ssm.c_norm.weight"), _eps);
        var dtLow = b.RmsNorm(b.Slice(bcdt, 1, 2 * stateSize, dtRank), b.Weight(p + "ssm.dt_norm.weight"), _eps);
        var dt = b.Linear(dtLow, p + "ssm.dt.weight");                                                   // [T, H]; the bias and softplus are part of the scan
        var scan = b.Emit("SelectiveScan",
            [b.Reshape(conv, 0, heads, headDim), dt, b.Weight(p + "ssm.dt_bias"), b.Weight(p + "ssm.a"),
             b.Reshape(bs, 0, 1, stateSize), b.Reshape(cc, 0, 1, stateSize), b.Weight(p + "ssm.d"), ss],
            Attributes.From(("dt_min", 0f)), $"{p}scan", writes: [ss])[0];                               // [T, H, P]
        var gated = b.Mul(b.Reshape(scan, 0, inner), b.Silu(z));
        return b.Linear(gated, p + "ssm.out.weight", name: $"{p}ssm_out");
    }
}
