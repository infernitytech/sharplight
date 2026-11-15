namespace Sharplight.Architectures;

/// <summary>Hyper-parameters of a plain transformer tower (the layer stack shared by image and text encoders).</summary>
/// <param name="Layers">Number of layers.</param>
/// <param name="Hidden">Hidden size.</param>
/// <param name="Heads">Attention heads.</param>
/// <param name="Eps">LayerNorm epsilon.</param>
/// <param name="Activation">MLP activation name (see <see cref="TowerBlocks.Activation"/>).</param>
/// <param name="Prefix">Weight name prefix of the tower (empty, <c>vision.</c> or <c>text.</c>).</param>
internal sealed record TowerOptions(int Layers, int Hidden, int Heads, float Eps, string Activation, string Prefix)
{
    /// <summary>Per-head size.</summary>
    public int HeadSize => Hidden / Heads;
}
