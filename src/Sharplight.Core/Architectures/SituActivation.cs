using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// The situ activation (Kimi K3): a soft-capped SiLU, situ(x) = β·tanh(x/β)·sigmoid(x). Gated blocks combine it with a soft-capped up
/// projection, situ(gate)·β_lin·tanh(up/β_lin); a linear beta of 0 leaves the up projection as it is.
/// </summary>
/// <param name="beta">β (<c>mlp.situ_beta</c>, positive).</param>
/// <param name="linearBeta">β_lin (<c>mlp.situ_linear_beta</c>; 0 = no cap on the up projection).</param>
internal sealed class SituActivation(float beta, float linearBeta) : IMlpActivation
{
    /// <inheritdoc/>
    public Value Activate(GraphBuilder b, Value x, string layerPrefix) => b.Mul(SoftCap(b, x, beta), b.Sigmoid(x));

    /// <inheritdoc/>
    public Value ActivateGated(GraphBuilder b, Value gate, Value up, string layerPrefix)
        => b.Mul(Activate(b, gate, layerPrefix), linearBeta > 0f ? SoftCap(b, up, linearBeta) : up);

    // cap · tanh(x / cap)
    private static Value SoftCap(GraphBuilder b, Value x, float cap) => b.Scale(b.Unary("Tanh", b.Scale(x, 1f / cap)), cap);
}
