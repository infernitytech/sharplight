namespace Sharplight.Studio.ViewModels.TimeSeries;

/// <summary>The prediction interval the page shades: a coverage in percent, as the quantile levels that bound it.</summary>
internal static class ForecastInterval
{
    /// <summary>The coverages the page offers, in percent.</summary>
    public static IReadOnlyList<int> Coverages { get; } = [50, 80, 90];

    /// <summary>The coverage chosen by default, in percent.</summary>
    public const int DefaultCoverage = 80;

    /// <summary>The levels bounding an interval of <paramref name="percent"/> coverage: <c>(1 - p) / 2</c> and <c>(1 + p) / 2</c>.</summary>
    /// <param name="percent">The coverage, 1 to 99.</param>
    /// <returns>The lower and the upper level (80 gives 0.1 and 0.9).</returns>
    public static (float Low, float High) Levels(int percent)
    {
        double p = Math.Clamp(percent, 1, 99) / 100.0;
        return ((float)Math.Round((1 - p) / 2, 4), (float)Math.Round((1 + p) / 2, 4));
    }
}
