namespace Sharplight.Architectures;

/// <summary>What one step of VibeVoice generation saw (see <see cref="VibeVoiceSynthesizer.GenerateFrames"/>).</summary>
/// <param name="Index">The step.</param>
/// <param name="Token">The token the step took (the language model's own choice unless forced).</param>
/// <param name="Logits">The language model's logits for the step.</param>
/// <param name="Hidden">The language model's hidden state (the diffusion condition).</param>
/// <param name="NegativeHidden">The context-free run's hidden state when the step asked for a latent, else <see langword="null"/>.</param>
/// <param name="Latent">The latent the diffusion head produced (before any forced one replaces it), or <see langword="null"/>.</param>
internal sealed record VibeVoiceStep(int Index, int Token, float[] Logits, float[] Hidden, float[]? NegativeHidden, float[]? Latent);
