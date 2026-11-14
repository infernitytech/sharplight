namespace Sharplight.Studio.Controls;

/// <summary>A bar of a <see cref="BarChart"/>.</summary>
/// <param name="Label">Label under the bar.</param>
/// <param name="Value">Height value.</param>
/// <param name="Secondary">Second stacked value drawn on top (0 for none).</param>
internal readonly record struct ChartBar(string Label, double Value, double Secondary = 0);
