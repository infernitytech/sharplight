namespace Sharplight.Architectures;

/// <summary>One generated patch of VoxCPM2 speech.</summary>
/// <param name="Latents">The latent frames ([patch, feature] row-major).</param>
/// <param name="Audio">The samples the AudioVAE decoded from them (the output rate).</param>
internal sealed record VoxCpm2Patch(float[] Latents, float[] Audio);
