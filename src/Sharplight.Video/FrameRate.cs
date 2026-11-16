namespace Sharplight.Video;

/// <summary>Converts a frame rate to the exact fraction the encoders and the WebM muxer use for timestamps.</summary>
internal static class FrameRate
{
    /// <summary>Whole rates are exact; others are rounded to thousandths (23.976 becomes 2997 / 125) and reduced.</summary>
    public static (int Numerator, int Denominator) ToFraction(double framesPerSecond)
    {
        if (framesPerSecond == Math.Floor(framesPerSecond) && framesPerSecond <= 100_000)
        {
            return ((int)framesPerSecond, 1);
        }

        int numerator = (int)Math.Clamp(Math.Round(framesPerSecond * 1000), 1, 100_000_000);
        int divisor = Gcd(numerator, 1000);
        return (numerator / divisor, 1000 / divisor);
    }

    private static int Gcd(int a, int b)
    {
        while (b != 0)
        {
            (a, b) = (b, a % b);
        }

        return a;
    }
}
