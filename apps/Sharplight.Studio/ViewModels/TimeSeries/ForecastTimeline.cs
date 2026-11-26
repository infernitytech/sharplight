using System.Globalization;

namespace Sharplight.Studio.ViewModels.TimeSeries;

/// <summary>Continues the time column of a table past its last row, when its timestamps are dates with a constant step.</summary>
internal static class ForecastTimeline
{
    /// <summary>
    /// The <paramref name="count"/> timestamps after the last of <paramref name="timestamps"/>. The step is constant when all rows are the same time
    /// apart, or the same number of months apart on the same day of the month (monthly, quarterly, yearly data).
    /// </summary>
    /// <param name="timestamps">The time column's cells, or null.</param>
    /// <param name="count">The steps to continue.</param>
    /// <returns>The timestamps, written as dates (<c>yyyy-MM-dd</c>, with the time of day when the data has one), or null when they do not parse or the step varies.</returns>
    public static List<string>? Continue(IReadOnlyList<string>? timestamps, int count)
    {
        if (timestamps is null || timestamps.Count < 2 || count < 1)
        {
            return null;
        }

        var times = new List<DateTime>(timestamps.Count);
        foreach (string cell in timestamps)
        {
            if (!DateTime.TryParse(cell, CultureInfo.InvariantCulture, DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal, out var time))
            {
                return null;
            }

            times.Add(time);
        }

        var step = times[1] - times[0];
        bool constant = step > TimeSpan.Zero;
        for (int i = 2; i < times.Count && constant; i++)
        {
            constant = times[i] - times[i - 1] == step;
        }

        int months = MonthStep(times);
        if (!constant && months == 0)
        {
            return null;
        }

        string format = times.All(time => time.TimeOfDay == TimeSpan.Zero) ? "yyyy-MM-dd" : "yyyy-MM-dd HH:mm:ss";
        var result = new List<string>(count);
        for (int i = 1; i <= count; i++)
        {
            var next = constant ? times[^1] + (step * i) : times[^1].AddMonths(months * i);
            result.Add(next.ToString(format, CultureInfo.InvariantCulture));
        }

        return result;
    }

    /// <summary>The constant distance in months of consecutive rows on the same day of the month and time of day, or 0.</summary>
    private static int MonthStep(List<DateTime> times)
    {
        int step = MonthIndex(times[1]) - MonthIndex(times[0]);
        if (step <= 0)
        {
            return 0;
        }

        for (int i = 1; i < times.Count; i++)
        {
            if (times[i].Day != times[0].Day || times[i].TimeOfDay != times[0].TimeOfDay || MonthIndex(times[i]) - MonthIndex(times[i - 1]) != step)
            {
                return 0;
            }
        }

        return step;
    }

    private static int MonthIndex(DateTime time) => (time.Year * 12) + time.Month;
}
