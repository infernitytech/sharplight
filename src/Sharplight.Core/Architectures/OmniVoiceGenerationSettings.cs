namespace Sharplight.Architectures;

/// <summary>The settings of OmniVoice's iterative unmasking (the official package's <c>OmniVoiceGenerationConfig</c>).</summary>
/// <param name="Steps">The number of unmasking steps.</param>
/// <param name="GuidanceScale">Classifier-free guidance scale (0 disables the unconditional pass's influence).</param>
/// <param name="TimeShift">Time-step shift: smaller values reveal more positions in the early steps.</param>
/// <param name="LayerPenalty">The confidence penalty per codebook index, which makes earlier codebooks unmask first.</param>
/// <param name="PositionTemperature">Gumbel temperature of the choice of which positions to reveal (0 reveals the most confident).</param>
/// <param name="Denoise">Whether a voice-cloning prompt carries the denoise token.</param>
/// <param name="Seed">The seed of the position noise.</param>
internal sealed record OmniVoiceGenerationSettings(int Steps = 32, float GuidanceScale = 2f, double TimeShift = 0.1, float LayerPenalty = 5f, float PositionTemperature = 5f, bool Denoise = true, long? Seed = null)
{
    /// <summary>Settings that choose deterministically.</summary>
    public static OmniVoiceGenerationSettings Greedy { get; } = new(PositionTemperature: 0f);
}
