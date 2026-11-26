using System.Globalization;

namespace Sharplight.Studio.ViewModels.TimeSeries;

/// <summary>One forecast step of the result table: the step or its timestamp, the point forecast and the interval.</summary>
internal sealed class ForecastRowViewModel
{
    /// <param name="step">The 1-based step.</param>
    /// <param name="time">The timestamp of the step, or empty.</param>
    /// <param name="low">The lower bound of the interval.</param>
    /// <param name="mean">The point forecast.</param>
    /// <param name="high">The upper bound of the interval.</param>
    public ForecastRowViewModel(int step, string time, float low, float mean, float high)
    {
        Step = step.ToString(CultureInfo.InvariantCulture);
        Time = time;
        Low = Format(low);
        Mean = Format(mean);
        High = Format(high);
    }

    /// <summary>The 1-based step.</summary>
    public string Step { get; }

    /// <summary>The timestamp of the step, or empty.</summary>
    public string Time { get; }

    /// <summary>The lower bound of the interval.</summary>
    public string Low { get; }

    /// <summary>The point forecast.</summary>
    public string Mean { get; }

    /// <summary>The upper bound of the interval.</summary>
    public string High { get; }

    private static string Format(float value) => value.ToString("G5", CultureInfo.InvariantCulture);
}
