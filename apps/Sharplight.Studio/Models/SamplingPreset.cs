namespace Sharplight.Studio.Models;

/// <summary>A named set of sampling settings.</summary>
/// <param name="Name">Shown in the preset picker.</param>
/// <param name="Sampling">The settings.</param>
/// <param name="BuiltIn">Shipped with the app (cannot be deleted).</param>
internal sealed record SamplingPreset(string Name, SamplingSettings Sampling, bool BuiltIn = false)
{
    /// <summary>The presets every installation has.</summary>
    public static IReadOnlyList<SamplingPreset> Defaults { get; } =
    [
        new("Balanced", new SamplingSettings(), BuiltIn: true),
        new("Precise", new SamplingSettings { Temperature = 0.2, TopP = 0.9, TopK = 20, MinP = 0.1 }, BuiltIn: true),
        new("Creative", new SamplingSettings { Temperature = 1.0, TopP = 0.98, TopK = 0, MinP = 0.02 }, BuiltIn: true),
        new("Code", new SamplingSettings { Temperature = 0.1, TopP = 0.95, TopK = 20, MinP = 0.05, MaxTokens = 4096 }, BuiltIn: true),
        new("Greedy", new SamplingSettings { Temperature = 0, TopP = 1, TopK = 0, MinP = 0 }, BuiltIn: true),
    ];
}
