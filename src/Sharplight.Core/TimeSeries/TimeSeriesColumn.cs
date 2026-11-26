namespace Sharplight.TimeSeries;

/// <summary>One numeric column of a <see cref="TimeSeriesTable"/>.</summary>
/// <param name="Name">The column header (or <c>column N</c> when the table has no header).</param>
/// <param name="Values">The values, one per row; <see cref="float.NaN"/> where a cell is empty or not a number.</param>
public sealed record TimeSeriesColumn(string Name, float[] Values);
