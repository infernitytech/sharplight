namespace Sharplight.TimeSeries;

/// <summary>Host-side helpers on a series' history shared by the forecasters.</summary>
internal static class TimeSeriesHistory
{
    /// <summary>Rejects an empty history or one without observed values.</summary>
    /// <exception cref="ArgumentException">The history is empty, all missing or holds an infinite value.</exception>
    public static void Validate(ReadOnlySpan<float> history)
    {
        bool observed = false;
        foreach (float value in history)
        {
            if (float.IsInfinity(value))
            {
                throw new ArgumentException("The history holds an infinite value.", nameof(history));
            }

            observed |= !float.IsNaN(value);
        }

        if (!observed)
        {
            throw new ArgumentException("The history has no observed value.", nameof(history));
        }
    }

    /// <summary>
    /// Chronos-Bolt's instance normalization: the mean and the (population) standard deviation of the observed values; a constant series gets
    /// <paramref name="epsilon"/> as its scale.
    /// </summary>
    public static (double Loc, double Scale) InstanceScale(ReadOnlySpan<float> values, float epsilon)
    {
        double sum = 0;
        int count = 0;
        foreach (float value in values)
        {
            if (!float.IsNaN(value))
            {
                sum += value;
                count++;
            }
        }

        if (count == 0)
        {
            return (0, 1);
        }

        double loc = sum / count;
        double squares = 0;
        foreach (float value in values)
        {
            if (!float.IsNaN(value))
            {
                squares += (value - loc) * (value - loc);
            }
        }

        double scale = Math.Sqrt(squares / count);
        return (loc, scale == 0 ? epsilon : scale);
    }

    /// <summary>Chronos' mean scaling: the mean absolute observed value, 1 when that is not positive.</summary>
    public static double MeanAbsoluteScale(ReadOnlySpan<float> values)
    {
        double sum = 0;
        int count = 0;
        foreach (float value in values)
        {
            if (!float.IsNaN(value))
            {
                sum += Math.Abs(value);
                count++;
            }
        }

        double scale = count == 0 ? 0 : sum / count;
        return scale > 0 ? scale : 1;
    }
}
