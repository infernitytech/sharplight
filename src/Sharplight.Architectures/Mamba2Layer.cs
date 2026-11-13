using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The Mamba-2 state-space mixer shared by the hybrid adapters (<see cref="NemotronHAdapter"/>, <see cref="HybridAdapter"/>): input
/// projection, causal depthwise convolution, selective scan, gated group RMS norm and output projection.
/// </summary>
/// <remarks>
/// Weights below a layer prefix <c>p</c>: <c>p + "ssm.in.weight"</c> [inner + conv_dim + H, hidden] (rows: gate z | conv input x,B,C | dt),
/// <c>ssm.conv1d.weight</c> [conv_dim, K] f32, <c>ssm.conv1d.bias</c> [conv_dim] f32, <c>ssm.dt_bias</c> [H] f32, <c>ssm.a</c> [H] f32 (already negative),
/// <c>ssm.d</c> [H] f32, optional <c>ssm.norm.weight</c> [inner] f32 (gated group RMS norm, G groups; without it the output is just <c>y · silu(z)</c>),
/// <c>ssm.out.weight</c> [hidden, inner]; conv_dim = inner + 2·G·N.
/// </remarks>
internal sealed class Mamba2Layer
{
    private readonly int _convKernel;
    private readonly int _stateSize;
    private readonly int _groups;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly float _eps;
    private readonly float _dtMin;

    /// <summary>Reads the <c>ssm.*</c> hyper-parameters of a source (<c>ssm.conv_kernel</c>, <c>ssm.state_size</c>, <c>ssm.group_count</c>, <c>ssm.head_count</c>, <c>ssm.head_dim</c>, <c>ssm.time_step_min</c>).</summary>
    public Mamba2Layer(ModelSource src, float eps)
    {
        _convKernel = src.Arch("ssm.conv_kernel", 4);
        _stateSize = src.Arch("ssm.state_size", 0);
        _groups = src.Arch("ssm.group_count", 1);
        _heads = src.Arch("ssm.head_count", 0);
        _headDim = src.Arch("ssm.head_dim", 0);
        _dtMin = src.Arch("ssm.time_step_min", 0f);
        _eps = eps;
    }

    private int Inner => _heads * _headDim;

    private int ConvDim => Inner + 2 * _groups * _stateSize;

    /// <summary>Declares the per-layer recurrent states (convolution window and scan state).</summary>
    public (StateDefinition Conv, StateDefinition Scan) DefineStates(string group)
    {
        var conv = new StateDefinition { Name = $"{group}.conv_state", Type = DataType.F32, Shape = new Shape(_convKernel - 1, ConvDim), Kind = StateKind.Recurrent, Group = group };
        var scan = new StateDefinition { Name = $"{group}.ssm_state", Type = DataType.F32, Shape = new Shape(_heads, _headDim, _stateSize), Kind = StateKind.Recurrent, Group = group };
        return (conv, scan);
    }

    /// <summary>Emits the mixer on the (already normalized) input <paramref name="h"/> [T, hidden].</summary>
    public Value Build(GraphBuilder b, Value h, string p, StateDefinition convState, StateDefinition scanState)
    {
        int inner = Inner;
        int convDim = ConvDim;
        int groupWidth = _groups * _stateSize;
        var proj = b.Linear(h, p + "ssm.in.weight");                                   // [T, inner + conv_dim + H]
        var z = b.Slice(proj, 1, 0, inner);
        var xbc = b.Slice(proj, 1, inner, convDim);
        var dt = b.Slice(proj, 1, inner + convDim, _heads);
        var cs = b.State(convState);
        var ss = b.State(scanState);
        var conv = b.Emit("CausalConv1d", [xbc, b.Weight(p + "ssm.conv1d.weight"), cs, b.Weight(p + "ssm.conv1d.bias")],
            Attributes.From(("activation", "silu"), ("bias", true)), writes: [cs])[0];
        var xs = b.Reshape(b.Slice(conv, 1, 0, inner), 0, _heads, _headDim);
        var bs = b.Reshape(b.Slice(conv, 1, inner, groupWidth), 0, _groups, _stateSize);
        var cc = b.Reshape(b.Slice(conv, 1, inner + groupWidth, groupWidth), 0, _groups, _stateSize);
        var scan = b.Emit("SelectiveScan", [xs, dt, b.Weight(p + "ssm.dt_bias"), b.Weight(p + "ssm.a"), bs, cc, b.Weight(p + "ssm.d"), ss],
            Attributes.From(("dt_min", _dtMin)), $"{p}scan", writes: [ss])[0];          // [T, H, P]
        var gated = b.Mul(b.Reshape(scan, 0, inner), b.Silu(z));
        if (b.TryWeight(p + "ssm.norm.weight") is { } normWeight)
        {
            // gated group RMS norm: norm_G(y · silu(z)) · weight
            var normed = b.Reshape(b.RmsNorm(b.Reshape(gated, 0, _groups, inner / _groups), null, _eps), 0, inner);
            gated = b.Mul(normed, normWeight, $"{p}gated_norm");
        }

        return b.Linear(gated, p + "ssm.out.weight", name: $"{p}ssm_out");
    }
}
