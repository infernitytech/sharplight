using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>An activation function of the feed-forward block, built from elementwise graph ops (see <see cref="MlpActivations"/>).</summary>
internal interface IMlpActivation
{
    /// <summary>Applies the activation to <paramref name="x"/>; <paramref name="layerPrefix"/> (<c>layers.{i}.</c>) locates per-layer parameters.</summary>
    public Value Activate(GraphBuilder b, Value x, string layerPrefix);

    /// <summary>Combines the gate and up projections of a gated block (default: <c>activation(gate) * up</c>).</summary>
    public Value ActivateGated(GraphBuilder b, Value gate, Value up, string layerPrefix) => b.Mul(Activate(b, gate, layerPrefix), up);
}
