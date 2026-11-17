namespace Sharplight.Architectures;

/// <summary>A CosyVoice voice, taken from a recording: its S3 speech tokens, its prompt log-mel and its CAM++ x-vector.</summary>
/// <param name="SpeechTokens">The recording's speech tokens (25 per second), at most half the prompt mel frames.</param>
/// <param name="PromptMel">The recording's Matcha log-mel at 24 kHz [2 · tokens, 80] (time-major).</param>
/// <param name="XVector">The CAM++ x-vector [192].</param>
internal sealed record CosyVoiceVoice(int[] SpeechTokens, float[] PromptMel, float[] XVector);
