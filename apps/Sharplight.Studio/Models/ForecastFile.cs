using System.Globalization;
using System.Text;

namespace Sharplight.Studio.Models;

/// <summary>
/// A forecast as the Time series page stores and exports it (<c>forecast.json</c> of a run): the tail of the history that was plotted, the
/// forecast per quantile level and the timestamps that continue the time column, with the CSV rendering.
/// </summary>
internal sealed class ForecastFile
{
    /// <summary>The forecast column's name.</summary>
    public string Column { get; set; } = "";

    /// <summary>The header of the time column the forecast's timestamps continue, or empty.</summary>
    public string TimeHeader { get; set; } = "";

    /// <summary>The last observations that were plotted, oldest first; <see langword="null"/> marks a missing value.</summary>
    public List<float?> History { get; set; } = [];

    /// <summary>The timestamps of <see cref="History"/> when the table has a time column.</summary>
    public List<string>? HistoryTimes { get; set; }

    /// <summary>The quantile levels, ascending.</summary>
    public List<float> Levels { get; set; } = [];

    /// <summary>Per level (in the order of <see cref="Levels"/>), the forecast of every step.</summary>
    public List<float[]> Quantiles { get; set; } = [];

    /// <summary>The point forecast of every step.</summary>
    public float[] Mean { get; set; } = [];

    /// <summary>The timestamps of the forecast steps when the time column continues with a constant step, else <see langword="null"/>.</summary>
    public List<string>? Times { get; set; }

    /// <summary>The lower level of the shaded interval.</summary>
    public float LowLevel { get; set; }

    /// <summary>The upper level of the shaded interval.</summary>
    public float HighLevel { get; set; }

    /// <summary>The model's context length.</summary>
    public int ContextLength { get; set; }

    /// <summary>The model's prediction length.</summary>
    public int PredictionLength { get; set; }

    /// <summary>The forecast steps.</summary>
    public int Horizon => Mean.Length;

    /// <summary>The header of a quantile column, for example <c>q0.1</c>.</summary>
    /// <param name="level">The quantile level.</param>
    public static string LevelHeader(float level) => "q" + level.ToString("0.####", CultureInfo.InvariantCulture);

    /// <summary>
    /// The forecast as CSV: <c>timestamp</c> (the time column's header, when the timestamps continue) or <c>step</c> (1-based), then <c>mean</c> and one
    /// column per quantile level (<c>q0.1</c>, <c>q0.5</c>, ...).
    /// </summary>
    public string ToCsv()
    {
        var text = new StringBuilder();
        bool timed = Times is { Count: > 0 } && Times.Count == Horizon;
        text.Append(timed ? Escape(TimeHeader.Length > 0 ? TimeHeader : "timestamp") : "step").Append(",mean");
        foreach (float level in Levels)
        {
            text.Append(',').Append(LevelHeader(level));
        }

        text.Append('\n');
        for (int step = 0; step < Horizon; step++)
        {
            text.Append(timed ? Escape(Times![step]) : (step + 1).ToString(CultureInfo.InvariantCulture));
            text.Append(',').Append(Number(Mean[step]));
            for (int level = 0; level < Levels.Count; level++)
            {
                text.Append(',').Append(Number(Quantiles[level][step]));
            }

            text.Append('\n');
        }

        return text.ToString();
    }

    private static string Number(float value) => value.ToString("R", CultureInfo.InvariantCulture);

    private static string Escape(string cell)
        => cell.AsSpan().IndexOfAny(",\"\n") >= 0 ? "\"" + cell.Replace("\"", "\"\"", StringComparison.Ordinal) + "\"" : cell;
}
