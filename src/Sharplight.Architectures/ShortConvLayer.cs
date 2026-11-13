using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Gated short-convolution mixer (LFM2): <c>(b, c, x) = in_proj(h)</c>, <c>y = c · causal_conv(b · x)</c>, <c>out_proj(y)</c>.
/// Weights below a layer prefix: <c>shortconv.in.weight</c> [3·C, hidden], <c>shortconv.conv.weight</c> [C, K] f32 (depthwise, no bias),
/// <c>shortconv.out.weight</c> [hidden, C]. Hyper-parameter <c>shortconv.kernel</c> (K).
/// </summary>
internal sealed class ShortConvLayer
{
    private readonly int _kernel;

    /// <summary>Reads <c>shortconv.kernel</c>.</summary>
    public ShortConvLayer(ModelSource src)
    {
        _kernel = src.Arch("shortconv.kernel", 3);
    }

    /// <summary>Declares the convolution window state of a layer with <paramref name="channels"/> channels.</summary>
    public StateDefinition DefineState(string group, int channels)
        => new() { Name = $"{group}.conv_state", Type = DataType.F32, Shape = new Shape(_kernel - 1, channels), Kind = StateKind.Recurrent, Group = group };

    /// <summary>Emits the mixer on the normalized input <paramref name="h"/> [T, hidden].</summary>
    public Value Build(GraphBuilder b, Value h, string p, StateDefinition convState)
    {
        var proj = b.Linear(h, p + "shortconv.in.weight");                               // [T, 3C]
        long channels = proj.Shape[1].Fixed / 3;
        var gateIn = b.Slice(proj, 1, 0, channels);
        var gateOut = b.Slice(proj, 1, channels, channels);
        var x = b.Slice(proj, 1, 2 * channels, channels);
        var state = b.State(convState);
        var conv = b.Emit("CausalConv1d", [b.Mul(gateIn, x), b.Weight(p + "shortconv.conv.weight"), state],
            Attributes.From(("activation", "none")), writes: [state])[0];
        return b.Linear(b.Mul(gateOut, conv), p + "shortconv.out.weight", name: $"{p}shortconv_out");
    }
}
