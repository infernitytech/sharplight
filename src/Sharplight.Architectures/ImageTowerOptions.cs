namespace Sharplight.Architectures;

/// <summary>Hyper-parameters of an image tower (<see cref="ImageTowerGraph"/>).</summary>
/// <param name="Tower">The transformer layer stack.</param>
/// <param name="Patch">Patch side in pixels.</param>
/// <param name="ImageSize">Side of the square input; the position table has exactly the grid of this size.</param>
/// <param name="Registers">Register tokens inserted after the class token.</param>
/// <param name="Pooling"><c>cls</c> (the class row), <c>mean</c> (the mean of the patch rows) or <c>attention_head</c> (SigLIP's attention pooling with a learned probe).</param>
internal sealed record ImageTowerOptions(TowerOptions Tower, int Patch, int ImageSize, int Registers, string Pooling);
