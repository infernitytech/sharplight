namespace Sharplight.Architectures;

/// <summary>The settings of a <see cref="JointSchemaSpanHead"/>.</summary>
/// <param name="Eps">LayerNorm epsilon.</param>
/// <param name="Heads">Attention heads of the layers.</param>
/// <param name="RoutingLayers">Routing layers.</param>
/// <param name="Layers">Joint layers.</param>
/// <param name="OutputEmbedding">Name of the output embedding weight [V, H] whose rows give the lexical vectors.</param>
internal sealed record JointSchemaSpanHeadSettings(float Eps, int Heads, int RoutingLayers, int Layers, string OutputEmbedding);
