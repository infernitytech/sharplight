using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// xIELU (Apertus): x &gt; 0 gives <c>αp·x² + β·x</c>, otherwise <c>(expm1(min(x, ε)) − x)·αn + β·x</c>. The effective
/// <c>αp</c> and <c>αn</c> are per-layer scalar weights <c>mlp.act.alpha_p</c> / <c>mlp.act.alpha_n</c> ([1], already
/// softplus-transformed by the importer), β and ε come from <c>mlp.xielu_beta</c> / <c>mlp.xielu_eps</c>.
/// </summary>
internal sealed class XieluActivation(float beta, float eps) : IMlpActivation
{
    /// <inheritdoc/>
    public Value Activate(GraphBuilder b, Value x, string layerPrefix)
    {
        var alphaPositive = b.Weight(layerPrefix + "mlp.act.alpha_p");
        var alphaNegative = b.Weight(layerPrefix + "mlp.act.alpha_n");
        var betaX = b.Scale(x, beta);

        var positive = b.Add(b.Mul(b.Unary("Square", x), alphaPositive), betaX);
        var epsilon = b.Constant("xielu_eps", DataType.F32, new Shape(1), BitConverter.GetBytes(eps));
        var expm1 = b.Scale(b.Unary("Exp", b.Binary("Min", x, epsilon)), 1f, -1f);
        var negative = b.Add(b.Mul(b.Sub(expm1, x), alphaNegative), betaX);

        var zero = b.Constant("xielu_zero", DataType.F32, new Shape(1), BitConverter.GetBytes(0f));
        var isPositive = b.Binary("Greater", x, zero);
        return b.Add(negative, b.Mul(isPositive, b.Sub(positive, negative)));
    }
}
