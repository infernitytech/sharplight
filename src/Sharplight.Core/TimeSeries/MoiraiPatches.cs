namespace Sharplight.TimeSeries;

/// <summary>
/// The host side of uni2ts' Moirai inputs (<c>MoiraiForecast._convert</c>, <c>Moirai2Forecast._convert</c>): the context left-padded to whole patches, missing
/// values and padding as 0 with a false observation flag (uni2ts' imputation leaves 0 for a univariate series), the packed standard scaler and the time ids.
/// </summary>
internal static class MoiraiPatches
{
    private const double MinimumScale = 1e-5;

    /// <summary>The context left-padded to whole patches of <paramref name="patchSize"/>: the values (0 where missing or padded) and the observation flags.</summary>
    public static (float[] Values, bool[] Observed) Pad(ReadOnlySpan<float> context, int patchSize)
    {
        int length = (context.Length + patchSize - 1) / patchSize * patchSize;
        int offset = length - context.Length;
        var values = new float[length];
        var observed = new bool[length];
        for (int i = 0; i < context.Length; i++)
        {
            if (!float.IsNaN(context[i]))
            {
                values[offset + i] = context[i];
                observed[offset + i] = true;
            }
        }

        return (values, observed);
    }

    /// <summary>
    /// uni2ts' <c>PackedStdScaler</c> over the observed values: the mean and the sample deviation (denominator n − 1, or 1 when that is 0), as
    /// <c>sqrt(variance + 1e-5)</c>, computed in double and rounded to float as the module does.
    /// </summary>
    public static (float Loc, float Scale) Scale(ReadOnlySpan<float> values, ReadOnlySpan<bool> observed)
    {
        double count = 0;
        double sum = 0;
        for (int i = 0; i < values.Length; i++)
        {
            if (observed[i])
            {
                count++;
                sum += values[i];
            }
        }

        double loc = count == 0 ? 0 : sum / count;
        double squares = 0;
        for (int i = 0; i < values.Length; i++)
        {
            if (observed[i])
            {
                squares += (values[i] - loc) * (values[i] - loc);
            }
        }

        double denominator = Math.Max(count - 1, 0);
        double variance = squares / (denominator == 0 ? 1 : denominator);
        return ((float)loc, (float)Math.Sqrt(variance + MinimumScale));
    }

    /// <summary>
    /// The time ids of the patches (<paramref name="observed"/> covers whole patches): the running count of patches since the first one with an observed
    /// value, less one and at least 0, so leading patches without observations share id 0 with the first observed one (uni2ts' <c>_generate_time_id</c>).
    /// </summary>
    public static int[] TimeIds(ReadOnlySpan<bool> observed, int patchSize)
    {
        var ids = new int[observed.Length / patchSize];
        bool seen = false;
        int running = 0;
        for (int patch = 0; patch < ids.Length; patch++)
        {
            seen |= observed.Slice(patch * patchSize, patchSize).Contains(true);
            running += seen ? 1 : 0;
            ids[patch] = Math.Max(running - 1, 0);
        }

        return ids;
    }

    /// <summary>The scaled value as the module computes it in float: <c>(value − loc) / scale</c>.</summary>
    public static float Scaled(float value, float loc, float scale) => (value - loc) / scale;
}
