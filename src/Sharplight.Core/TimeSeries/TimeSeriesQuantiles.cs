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

    /// <summary>
    /// chronos-forecasting's <c>interpolate_quantiles</c>: the value at <paramref name="query"/> of the piecewise-linear quantile function through
    /// (<paramref name="levels"/>, <paramref name="values"/>), extended flat to levels 0 and 1 (non-uniform levels allowed).
    /// </summary>
    /// <param name="levels">The levels, ascending, within [0, 1].</param>
    /// <param name="values">The value at every level.</param>
    /// <param name="query">The wanted level.</param>
    public static float Interpolate(ReadOnlySpan<float> levels, ReadOnlySpan<float> values, float query)
    {
        // Padded points: (0, values[0]) before the first level when it is above 0, (1, values[^1]) after the last when it is below 1.
        int front = levels[0] > 0f ? 1 : 0;
        int count = levels.Length + front + (levels[^1] < 1f ? 1 : 0);
        Span<float> x = count <= 256 ? stackalloc float[count] : new float[count];
        Span<float> y = count <= 256 ? stackalloc float[count] : new float[count];
        x[0] = 0f;
        y[0] = values[0];
        x[^1] = 1f;
        y[^1] = values[^1];
        levels.CopyTo(x[front..]);
        values.CopyTo(y[front..]);

        // torch.searchsorted(right=True): the first level above the query, clamped to the last point.
        int upper = 0;
        while (upper < count && x[upper] <= query)
        {
            upper++;
        }

        upper = Math.Min(upper, count - 1);
        int lower = Math.Max(upper - 1, 0);
        float weight = (query - x[lower]) / (x[upper] - x[lower]);
        if (!float.IsFinite(weight))
        {
            weight = 0f;
        }

        return y[lower] + (weight * (y[upper] - y[lower]));
    }

    /// <summary>
    /// Chronos-2's probability mass of each quantile level (the trapezoid rule over the levels padded with 0 and 1: (next − previous) / 2, normalized).
    /// </summary>
    /// <param name="levels">The levels, ascending, strictly between 0 and 1.</param>
    public static float[] ProbabilityMass(IReadOnlyList<float> levels)
    {
        var mass = new float[levels.Count];
        float sum = 0f;
        for (int i = 0; i < mass.Length; i++)
        {
            float previous = i == 0 ? 0f : levels[i - 1];
            float next = i == mass.Length - 1 ? 1f : levels[i + 1];
            mass[i] = (next - previous) / 2f;
            sum += mass[i];
        }

        for (int i = 0; i < mass.Length; i++)
        {
            mass[i] /= sum;
        }

        return mass;
    }

    /// <summary>
    /// chronos-forecasting's <c>weighted_quantile</c>: the quantiles at <paramref name="query"/> of the weighted samples (sorted, their cumulative weights
    /// clamped to [0, 1] as the levels, then <see cref="Interpolate"/>).
    /// </summary>
    /// <param name="samples">The samples (reordered in place).</param>
    /// <param name="weights">The weight of every sample (positive; reordered in place, normalized here).</param>
    /// <param name="query">The wanted levels.</param>
    /// <param name="result">Receives one value per wanted level.</param>
    public static void Weighted(Span<float> samples, Span<float> weights, IReadOnlyList<float> query, Span<float> result)
    {
        float total = 0f;
        foreach (float weight in weights)
        {
            total += weight;
        }

        samples.Sort(weights);
        float cumulative = 0f;
        for (int i = 0; i < weights.Length; i++)
        {
            cumulative += weights[i] / total;
            weights[i] = Math.Clamp(cumulative, 0f, 1f);
        }

        for (int q = 0; q < query.Count; q++)
        {
            result[q] = Interpolate(weights, samples, query[q]);
        }
    }
}
