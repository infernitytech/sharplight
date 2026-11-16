namespace Sharplight.Architectures;

/// <summary>An XTTS v2 voice: a built-in speaker (<c>speakers_xtts.pth</c>) or a recording through <see cref="XttsVoiceCloning"/>.</summary>
/// <param name="Latents">The GPT's conditioning latents [32 · width].</param>
/// <param name="Speaker">The HiFi-GAN decoder's speaker embedding [512].</param>
internal sealed record XttsVoice(float[] Latents, float[] Speaker);
