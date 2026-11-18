namespace Sharplight.Architectures;

/// <summary>One generated frame of VibeVoice speech.</summary>
/// <param name="Latent">The acoustic latent the diffusion head produced (the model's scaled domain).</param>
/// <param name="Audio">The samples the acoustic decoder made of it (24 kHz).</param>
internal sealed record VibeVoiceFrame(float[] Latent, float[] Audio);
