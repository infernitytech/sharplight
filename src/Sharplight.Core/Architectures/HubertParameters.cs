namespace Sharplight.Architectures;

/// <summary>The HuBERT encoder inside the codec: a group-normed convolutional feature encoder, a grouped positional convolution and post-norm transformer layers.</summary>
/// <param name="Hidden">Model width.</param>
/// <param name="Layers">Transformer layers.</param>
/// <param name="Heads">Attention heads.</param>
/// <param name="ConvStrides">Stride of each feature convolution.</param>
/// <param name="PositionGroups">Groups of the positional convolution.</param>
/// <param name="Eps">Norm epsilon.</param>
internal sealed record HubertParameters(int Hidden, int Layers, int Heads, long[] ConvStrides, int PositionGroups, float Eps);
