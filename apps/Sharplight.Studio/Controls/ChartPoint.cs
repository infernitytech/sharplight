namespace Sharplight.Studio.Controls;

/// <summary>A point of a <see cref="LineChart"/>: x is a time (Unix seconds) or an index.</summary>
/// <param name="X">Horizontal value.</param>
/// <param name="Y">Vertical value.</param>
internal readonly record struct ChartPoint(double X, double Y);
