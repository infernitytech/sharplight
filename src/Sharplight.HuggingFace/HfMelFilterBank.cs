namespace Sharplight.HuggingFace;

/// <summary>
/// Triangular mel filter bank on the Slaney mel scale with Slaney (area) normalization, as librosa's <c>filters.mel</c> and transformers'
/// <c>mel_filter_bank(norm="slaney", mel_scale="slaney")</c> compute it for Whisper's feature extractor (computed in double precision,
/// rounded to f32 like the stored filters).
/// </summary>
internal static class HfMelFilterBank
{
    private const double LinearStep = 200.0 / 3.0;
    private const double LogStartHz = 1000.0;
    private static readonly double s_logStep = Math.Log(6.4) / 27.0;

    /// <summary>Filters [melBins, fftSize / 2 + 1] row-major, from 0 Hz to the Nyquist frequency.</summary>
    public static float[] Create(int melBins, int fftSize, int sampleRate)
    {
        int bins = (fftSize / 2) + 1;
        double nyquist = sampleRate / 2.0;
        double melMax = HzToMel(nyquist);
        var edges = new double[melBins + 2];
        for (int i = 0; i < edges.Length; i++)
        {
            edges[i] = MelToHz(melMax * i / (melBins + 1));
        }

        var filters = new float[melBins * bins];
        for (int m = 0; m < melBins; m++)
        {
            double lower = edges[m];
            double center = edges[m + 1];
            double upper = edges[m + 2];
            double norm = 2.0 / (upper - lower);
            for (int k = 0; k < bins; k++)
            {
                double frequency = nyquist * k / (bins - 1);
                double rising = (frequency - lower) / (center - lower);
                double falling = (upper - frequency) / (upper - center);
                filters[(m * bins) + k] = (float)(Math.Max(0.0, Math.Min(rising, falling)) * norm);
            }
        }

        return filters;
    }

    private static double HzToMel(double hz) => hz < LogStartHz ? hz / LinearStep : (LogStartHz / LinearStep) + (Math.Log(hz / LogStartHz) / s_logStep);

    private static double MelToHz(double mel)
    {
        double logStartMel = LogStartHz / LinearStep;
        return mel < logStartMel ? mel * LinearStep : LogStartHz * Math.Exp(s_logStep * (mel - logStartMel));
    }
}
