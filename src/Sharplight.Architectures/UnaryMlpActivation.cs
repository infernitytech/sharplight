using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>A parameter-free activation made of one standard op, optionally followed by a square (ReLU²).</summary>
internal sealed class UnaryMlpActivation(string op, bool squared = false) : IMlpActivation
{
    /// <inheritdoc/>
    public Value Activate(GraphBuilder b, Value x, string layerPrefix)
    {
        var y = b.Unary(op, x);
        return squared ? b.Unary("Square", y) : y;
    }
}
