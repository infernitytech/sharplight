namespace Sharplight.Architectures;

/// <summary>The per-head ALiBi slopes (Press et al.) shared by the decoder and encoder graphs.</summary>
internal static class AlibiSlopes
{
    /// <summary>
    /// Slopes of <paramref name="heads"/> heads for the maximum bias <paramref name="maxBias"/> (8 in the paper): a geometric sequence over the
    /// largest power of two not above the head count, the remaining heads interleaved from the sequence of twice that count. Null when
    /// <paramref name="maxBias"/> is not positive (no ALiBi).
    /// </summary>
    public static float[]? Of(int heads, float maxBias)
    {
        if (maxBias <= 0f)
        {
            return null;
        }

        int powerOfTwo = 1 << (int)Math.Floor(Math.Log2(heads));
        float first = MathF.Pow(2f, -maxBias / powerOfTwo);
        float second = MathF.Pow(2f, -maxBias / 2f / powerOfTwo);
        var slopes = new float[heads];
        for (int head = 0; head < heads; head++)
        {
            slopes[head] = head < powerOfTwo ? MathF.Pow(first, head + 1) : MathF.Pow(second, 2 * (head - powerOfTwo) + 1);
        }

        return slopes;
    }
}
