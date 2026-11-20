namespace Sharplight.Architectures;

/// <summary>What one step of VoxCPM2 generation saw (see <see cref="VoxCpm2Synthesizer.GeneratePatches"/>).</summary>
/// <param name="Index">The step.</param>
/// <param name="LmHidden">The text-semantic model's state (scalar-quantized after the prompt).</param>
/// <param name="ResidualHidden">The residual model's state.</param>
/// <param name="Stop">The stop logits (continue, stop).</param>
/// <param name="Patch">The patch the diffusion produced ([patch, feature] row-major; before any forced patch replaces it).</param>
/// <param name="Embedding">The local encoder's language model row of the patch that feeds the next step.</param>
internal sealed record VoxCpm2Step(int Index, float[] LmHidden, float[] ResidualHidden, float[] Stop, float[] Patch, float[] Embedding);
