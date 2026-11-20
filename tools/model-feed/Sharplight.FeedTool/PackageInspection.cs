namespace Sharplight.FeedTool;

/// <summary>What the built <c>.slm</c> says about itself.</summary>
/// <param name="Name">Model name.</param>
/// <param name="Kind">Engine model kind from the description.</param>
/// <param name="Architecture">Architecture, or <see langword="null"/>.</param>
/// <param name="ContextLength">Maximum context length, or <see langword="null"/>.</param>
/// <param name="Parameters">Number of weight elements.</param>
/// <param name="WeightBytes">Weight bytes.</param>
/// <param name="FormatVersion">Container version.</param>
/// <param name="HasChatTemplate">Whether a chat template is embedded.</param>
internal sealed record PackageInspection(string Name, string Kind, string? Architecture, int? ContextLength, long Parameters, long WeightBytes, int FormatVersion, bool HasChatTemplate)
{
    /// <summary>Weight bytes per engine weight type, largest first.</summary>
    public IReadOnlyList<KeyValuePair<string, long>> WeightTypes { get; init; } = [];

    /// <summary>Context memory of the decode graph, or <see langword="null"/> for models without one (encoders, diffusion, ...).</summary>
    public ContextMemory? Context { get; init; }
}
