namespace Sharplight.Architectures;

/// <summary>A voice to clone: the codes of the reference recording, its transcript and its loudness.</summary>
/// <param name="Codes">The reference codes [codebooks, frames] row-major.</param>
/// <param name="Frames">The number of frames.</param>
/// <param name="Text">The transcript of the reference.</param>
/// <param name="Rms">The root-mean-square level of the reference recording (before any boost).</param>
internal sealed record OmniVoiceReference(int[] Codes, int Frames, string Text, float Rms);
