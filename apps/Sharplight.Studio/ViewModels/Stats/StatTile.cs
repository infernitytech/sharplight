namespace Sharplight.Studio.ViewModels.Stats;

/// <summary>A key figure on the dashboard.</summary>
/// <param name="Label">What it is.</param>
/// <param name="Value">The figure.</param>
/// <param name="Unit">Unit after the figure.</param>
/// <param name="Icon">Icon resource key.</param>
/// <param name="Hint">Explanation.</param>
internal sealed record StatTile(string Label, string Value, string Unit, string Icon, string Hint);
