namespace Sharplight.Audio;

/// <summary>
/// Triangular mel filter bank on the HTK mel scale (mel = 2595 log10(1 + f / 700)), as torchaudio's <c>melscale_fbanks</c> computes it with
/// <c>mel_scale="htk"</c>: the filter edges evenly spaced in mel between the lowest and highest frequency, the FFT bins at
/// <c>k · (sampleRate div 2) / (bins − 1)</c>, optionally Slaney's area normalization (<c>norm="slaney"</c>: each filter scaled by
/// 2 / (upper edge − lower edge)).
/// </summary>
public static class HtkMelFilterBank
{
    /// <summary>Filters [melBins, fftSize / 2 + 1] row-major.</summary>
    /// <param name="melBins">The number of mel channels.</param>
    /// <param name="fftSize">The FFT size.</param>
    /// <param name="sampleRate">The sample rate in Hz.</param>
    /// <param name="minFrequency">The lowest frequency of the filters in Hz.</param>
    /// <param name="maxFrequency">The highest frequency in Hz, or the Nyquist frequency.</param>
    /// <param name="slaneyNorm">Whether to apply Slaney's area normalization.</param>
    public static float[] Create(int melBins, int fftSize, int sampleRate, double minFrequency = 0.0, double? maxFrequency = null, bool slaneyNorm = false)
    {
        static double ToMel(double hz) => 2595.0 * Math.Log10(1.0 + (hz / 700.0));
        static double ToHz(double mel) => 700.0 * (Math.Pow(10, mel / 2595.0) - 1.0);
        int bins = (fftSize / 2) + 1;
        double low = ToMel(minFrequency);
        double high = ToMel(maxFrequency ?? (sampleRate / 2.0));
        var points = new double[melBins + 2];
        for (int i = 0; i < points.Length; i++)
        {
            points[i] = ToHz(low + ((high - low) * i / (melBins + 1)));
        }

        var filters = new float[melBins * bins];
        for (int m = 0; m < melBins; m++)
        {
            double scale = slaneyNorm ? 2.0 / (points[m + 2] - points[m]) : 1.0;
            for (int k = 0; k < bins; k++)
            {
                double frequency = (double)k * (sampleRate / 2) / (bins - 1);
                double down = (frequency - points[m]) / (points[m + 1] - points[m]);
                double up = (points[m + 2] - frequency) / (points[m + 2] - points[m + 1]);
                filters[(m * bins) + k] = (float)(Math.Max(0, Math.Min(down, up)) * scale);
            }
        }

        return filters;
    }
}
