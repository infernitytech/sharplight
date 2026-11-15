namespace Sharplight.Vision;

/// <summary>The result of <see cref="IImageClassifier.ClassifyAsync"/>.</summary>
/// <param name="Predictions">The best classes, highest score first.</param>
public sealed record ImageClassification(IReadOnlyList<ImageLabelScore> Predictions)
{
    /// <summary>The best class, or <see langword="null"/> when no class was requested.</summary>
    public ImageLabelScore? Top => Predictions.Count > 0 ? Predictions[0] : null;
}
