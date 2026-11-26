namespace Sharplight.TimeSeries;

/// <summary>Empirical quantiles as <c>torch.quantile</c> computes them (linear interpolation between the sorted values), which the reference pipelines use.</summary>
internal static class TimeSeriesQuantiles
{
    /// <summary>The <paramref name="level"/> quantile of <paramref name="sorted"/> (ascending).</summary>
    public static float Of(ReadOnlySpan<float> sorted, float level)
    {
        double position = level * (double)(sorted.Length - 1);
        int low = (int)Math.Floor(position);
        int high = Math.Min(low + 1, sorted.Length - 1);
        double fraction = position - low;
        return (float)(sorted[low] + (fraction * (sorted[high] - sorted[low])));
    }

    /// <summary>
    /// Quantile levels from the forecasts of other levels, per step: <paramref name="levels"/> that are among <paramref name="native"/> are taken as they are,
    /// any other level interpolates (as Chronos-Bolt's <c>predict_quantiles</c>) between the native forecasts padded with copies of the outermost ones, which
    /// assumes evenly spaced native levels and clamps levels outside them.
    /// </summary>
    /// <param name="native">The native levels, ascending.</param>
    /// <param name="forecasts">Per native level, the forecast of every step.</param>
    /// <param name="levels">The wanted levels.</param>
    public static float[][] Select(IReadOnlyList<float> native, IReadOnlyList<float[]> forecasts, IReadOnlyList<float> levels)
    {
        int horizon = forecasts[0].Length;
        var result = new float[levels.Count][];
        bool subset = levels.All(level => IndexOf(native, level) >= 0);
        var column = new float[native.Count + 2];
        for (int l = 0; l < levels.Count; l++)
        {
            int index = IndexOf(native, levels[l]);
            if (subset)
            {
                result[l] = (float[])forecasts[index].Clone();
                continue;
            }

            result[l] = new float[horizon];
        }

        if (subset)
        {
            return result;
        }

        for (int t = 0; t < horizon; t++)
        {
            column[0] = forecasts[0][t];
            for (int q = 0; q < native.Count; q++)
            {
                column[q + 1] = forecasts[q][t];
            }

            column[^1] = forecasts[^1][t];
            Array.Sort(column);
            for (int l = 0; l < levels.Count; l++)
            {
                result[l][t] = Of(column, levels[l]);
            }
        }

        return result;
    }

    /// <summary>The index of <paramref name="level"/> among <paramref name="levels"/> (within 1e-6), or -1.</summary>
    public static int IndexOf(IReadOnlyList<float> levels, float level)
    {
        for (int i = 0; i < levels.Count; i++)
        {
            if (MathF.Abs(levels[i] - level) < 1e-6f)
            {
                return i;
            }
        }

        return -1;
    }

    /// <summary>Checks requested levels (null: <paramref name="native"/>) and returns them ascending.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A level is not above 0 and below 1.</exception>
    public static IReadOnlyList<float> Requested(IReadOnlyList<float>? levels, IReadOnlyList<float> native)
    {
        if (levels is null || levels.Count == 0)
        {
            return native;
        }

        foreach (float level in levels)
        {
            if (!(level > 0f && level < 1f))
            {
                throw new ArgumentOutOfRangeException(nameof(levels), $"Quantile level {level} is not between 0 and 1.");
            }
        }

        return [.. levels.Order()];
    }

    /// <summary>Parses the comma-separated levels of <see cref="TimeSeriesGraphs.QuantilesKey"/>.</summary>
    public static float[] Parse(string levels)
        => [.. levels.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(level => float.Parse(level, System.Globalization.CultureInfo.InvariantCulture))];
}
