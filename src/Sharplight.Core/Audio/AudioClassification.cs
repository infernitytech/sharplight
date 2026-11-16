namespace Sharplight.Audio;

/// <summary>The result of <see cref="IAudioEmbedder.ClassifyZeroShotAsync"/>.</summary>
/// <param name="Predictions">The best classes, highest score first.</param>
public sealed record AudioClassification(IReadOnlyList<AudioLabelScore> Predictions)
{
    /// <summary>The best class, or <see langword="null"/> when no class was requested.</summary>
    public AudioLabelScore? Top => Predictions.Count > 0 ? Predictions[0] : null;
}
