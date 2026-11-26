namespace Sharplight.Architectures;

/// <summary>The voice of a reference recording for a Qwen3-TTS voice-cloning model (<see cref="Qwen3TtsSynthesizer.EncodeReference"/>).</summary>
/// <param name="Speaker">The speaker encoder's embedding of the recording (the talker's hidden width).</param>
/// <param name="Frames">The codec frames of the recording (16 codes each) for the in-context prompt, or <see langword="null"/> for the speaker embedding alone.</param>
/// <param name="Text">The words spoken in the recording (required with <paramref name="Frames"/>), or <see langword="null"/>.</param>
internal sealed record Qwen3TtsReference(float[] Speaker, IReadOnlyList<int[]>? Frames, string? Text);
