namespace Sharplight.Vision;

/// <summary>Optional limits on the visual token budget per image (output tokens after merging), overriding the model's pixel bounds.</summary>
/// <param name="MinTokens">Fewest tokens an image is scaled up to, or <see langword="null"/> to use the model's bound.</param>
/// <param name="MaxTokens">Most tokens an image is scaled down to, or <see langword="null"/> to use the model's bound.</param>
public sealed record ImageTokenLimits(int? MinTokens = null, int? MaxTokens = null);
