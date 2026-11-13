namespace Sharplight.Vision;

/// <summary>Optional limits on the visual token budget per image (output tokens after merging), overriding the model's pixel bounds.</summary>
public sealed record ImageTokenLimits(int? MinTokens = null, int? MaxTokens = null);
