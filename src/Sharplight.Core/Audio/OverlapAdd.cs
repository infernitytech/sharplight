namespace Sharplight.Audio;

/// <summary>
/// The synthesis half of an inverse short-time Fourier transform on the host: windowed frames are added at their hop positions and divided
/// by the summed squared window (the envelope), with the 'same' padding of Vocos-style vocoders trimmed from both ends. The frames themselves
/// (inverse DFT and window multiplication) come from the model graph.
/// </summary>
public static class OverlapAdd
{
    /// <summary>
    /// Overlap-adds frames and normalizes by the window envelope.
    /// </summary>
    /// <param name="frames">The (already windowed) frames, <paramref name="frameCount"/> × <paramref name="frameLength"/> samples, frame-major.</param>
    /// <param name="frameCount">The number of frames.</param>
    /// <param name="frameLength">The length of a frame (the window length).</param>
    /// <param name="hop">The distance between frame starts.</param>
    /// <param name="window">The window the frames were multiplied with (its square forms the envelope).</param>
    /// <param name="trim">Samples removed from the start and from the end of the result (<c>(frameLength - hop) / 2</c> for 'same' padding).</param>
    /// <returns>The signal of <c>(frameCount - 1) · hop + frameLength - 2 · trim</c> samples.</returns>
    public static float[] Synthesize(ReadOnlySpan<float> frames, int frameCount, int frameLength, int hop, ReadOnlySpan<float> window, int trim)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(frameCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hop);
        if (frames.Length != frameCount * frameLength || window.Length != frameLength)
        {
            throw new ArgumentException("The frames and the window do not match the frame length.");
        }

        int total = frameCount == 0 ? 0 : ((frameCount - 1) * hop) + frameLength;
        if (total <= 2 * trim)
        {
            return [];
        }

        var signal = new double[total];
        var envelope = new double[total];
        for (int f = 0; f < frameCount; f++)
        {
            int start = f * hop;
            for (int n = 0; n < frameLength; n++)
            {
                signal[start + n] += frames[(f * frameLength) + n];
                envelope[start + n] += (double)window[n] * window[n];
            }
        }

        var result = new float[total - (2 * trim)];
        for (int i = 0; i < result.Length; i++)
        {
            double e = envelope[i + trim];
            result[i] = (float)(e > 1e-11 ? signal[i + trim] / e : signal[i + trim]);
        }

        return result;
    }
}
