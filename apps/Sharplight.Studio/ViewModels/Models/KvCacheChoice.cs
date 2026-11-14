namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A KV cache element type choice.</summary>
/// <param name="Value">The engine's name.</param>
/// <param name="Name">Shown in the picker.</param>
internal sealed record KvCacheChoice(string Value, string Name)
{
    /// <summary>The choices.</summary>
    public static IReadOnlyList<KvCacheChoice> All { get; } =
    [
        new("f16", "F16 (default)"),
        new("q8_0", "Q8_0 (half the memory)"),
        new("q4_0", "Q4_0 (quarter, less accurate)"),
        new("f32", "F32"),
    ];
}
