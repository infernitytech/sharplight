namespace Sharplight.Audio;

/// <summary>Analysis and synthesis windows of signal processing.</summary>
public static class Window
{
    /// <summary>The periodic Hann window <c>0.5 · (1 - cos(2πn / length))</c> (torch's <c>hann_window</c>, the window of STFT / ISTFT).</summary>
    /// <param name="length">The window length.</param>
    public static float[] Hann(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        var window = new float[length];
        for (int n = 0; n < length; n++)
        {
            window[n] = (float)(0.5 * (1 - Math.Cos(2 * Math.PI * n / length)));
        }

        return window;
    }
}
