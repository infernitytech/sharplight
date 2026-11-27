namespace Sharplight.TimeSeries;

/// <summary>Host-side helpers on a series' history shared by the forecasters.</summary>
internal static class TimeSeriesHistory
{
    /// <summary>Rejects an empty history or one without observed values.</summary>
    /// <exception cref="ArgumentException">The history is empty, all missing or holds an infinite value.</exception>
    public static void Validate(ReadOnlySpan<float> history)
    {
        if (Problem(history) is { } problem)
        {
            throw new ArgumentException(problem, nameof(history));
        }
    }

    // Why a history cannot be forecast, or null.
    private static string? Problem(ReadOnlySpan<float> history)
    {
        bool observed = false;
        foreach (float value in history)
        {
            if (float.IsInfinity(value))
            {
                return "The history holds an infinite value.";
            }

            observed |= !float.IsNaN(value);
        }

        return observed ? null : "The history has no observed value.";
    }

    /// <summary>Rejects a batch of univariate histories that is empty or holds an invalid history (see <see cref="Validate(ReadOnlySpan{float})"/>).</summary>
    /// <exception cref="ArgumentException">There is no history, or one is empty, all missing or holds an infinite value (the message names its index).</exception>
    public static void Validate(IReadOnlyList<ReadOnlyMemory<float>> histories)
    {
        ArgumentNullException.ThrowIfNull(histories);
        if (histories.Count == 0)
        {
            throw new ArgumentException("There is no series to forecast.", nameof(histories));
        }

        for (int i = 0; i < histories.Count; i++)
        {
            if (Problem(histories[i].Span) is { } problem)
            {
                throw new ArgumentException($"Series {i}: {problem}", nameof(histories));
            }
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

    /// <summary>
    /// The horizon of a multivariate request: <see cref="TimeSeriesForecastOptions.Horizon"/>, else the length of the known futures of its covariates, else
    /// <paramref name="predictionLength"/>.
    /// </summary>
    public static int Horizon(TimeSeriesForecastRequest request, TimeSeriesForecastOptions? options, int predictionLength)
    {
        if (options?.Horizon is { } horizon)
        {
            return horizon;
        }

        foreach (var covariate in request.Covariates)
        {
            if (!covariate.Future.IsEmpty)
            {
                return covariate.Future.Length;
            }
        }

        return predictionLength;
    }

    /// <summary>Checks a multivariate request for <paramref name="horizon"/> steps.</summary>
    /// <exception cref="ArgumentException">
    /// The request has no target, a target has no observed value, a series holds an infinite value, the histories differ in length, or a known future is not
    /// <paramref name="horizon"/> steps long.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">The horizon is below 1.</exception>
    public static void Validate(TimeSeriesForecastRequest request, int horizon)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(horizon, 1, nameof(horizon));
        if (request.Targets is null || request.Targets.Count == 0)
        {
            throw new ArgumentException("The request has no target series.", nameof(request));
        }

        int length = request.Targets[0].Length;
        foreach (var target in request.Targets)
        {
            Validate(target.Span);
            if (target.Length != length)
            {
                throw new ArgumentException($"The target histories differ in length ({target.Length} and {length}).", nameof(request));
            }
        }

        foreach (var covariate in request.Covariates)
        {
            if (covariate.History.Length != length)
            {
                throw new ArgumentException($"Covariate '{covariate.Name}' has {covariate.History.Length} history values; the targets have {length}.", nameof(request));
            }

            Finite(covariate.History.Span, covariate.Name);
            if (!covariate.Future.IsEmpty)
            {
                if (covariate.Future.Length != horizon)
                {
                    throw new ArgumentException($"Covariate '{covariate.Name}' has {covariate.Future.Length} future values; the horizon is {horizon} steps.", nameof(request));
                }

                Finite(covariate.Future.Span, covariate.Name);
            }
        }
    }

    private static void Finite(ReadOnlySpan<float> values, string name)
    {
        foreach (float value in values)
        {
            if (float.IsInfinity(value))
            {
                throw new ArgumentException($"Covariate '{name}' holds an infinite value.", "request");
            }
        }
    }
}
