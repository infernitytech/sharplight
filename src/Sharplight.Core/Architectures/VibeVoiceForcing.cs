namespace Sharplight.Architectures;

/// <summary>Values that replace what VibeVoice generation would choose, for teacher-forced verification against a reference run.</summary>
internal sealed class VibeVoiceForcing
{
    /// <summary>The tokens of the steps, or <see langword="null"/> for the language model's own choice.</summary>
    public IReadOnlyList<int>? Tokens { get; init; }

    /// <summary>The latents that feed the language model and the decoder in place of the diffusion head's, or <see langword="null"/>.</summary>
    public IReadOnlyList<float[]>? Latents { get; init; }

    /// <summary>The initial noise of each frame's diffusion, or <see langword="null"/> for random noise.</summary>
    public IReadOnlyList<float[]>? Noise { get; init; }
}
