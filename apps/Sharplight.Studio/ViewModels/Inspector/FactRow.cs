namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>A labelled fact of an overview card.</summary>
/// <param name="Label">What the fact is.</param>
/// <param name="Value">Its value.</param>
internal sealed record FactRow(string Label, string Value);
