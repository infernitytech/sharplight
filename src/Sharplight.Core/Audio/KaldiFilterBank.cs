using Sharplight.Models;

namespace Sharplight.Audio;

/// <summary>
/// Kaldi's log mel filter-bank features (feature-extractor <c>type</c> <c>kaldi-fbank</c>), as torchaudio's <c>compliance.kaldi.fbank</c> computes them
/// with its defaults: frames that fit completely (snip edges), each frame's mean removed, pre-emphasis within the frame (the first sample repeated),
/// the Povey window (a symmetric Hann window to the power 0.85), zero padding to the next power of two, the power spectrum, Kaldi's triangular mel
/// bank (mel = 1127 ln(1 + f / 700), the bins between <c>lowFrequency</c> and <c>highFrequency</c>), and the natural logarithm floored at the float
/// epsilon. Optionally each channel's mean over the recording is subtracted (CAMPPlus's front end).
/// </summary>
/// <remarks>
/// Component configuration: <c>sampleRate</c> (default 16000), <c>frameLength</c> and <c>frameShift</c> in samples (defaults 400 and 160),
/// <c>melBins</c> (default 23), <c>lowFrequency</c> (Hz, default 20), <c>highFrequency</c> (Hz; zero or negative: relative to the Nyquist frequency,
/// default 0), <c>preemphasis</c> (default 0.97), <c>subtractMean</c> (default false).
/// </remarks>
internal sealed class KaldiFilterBank : IAudioFeatureExtractor
{
    /// <summary>The feature-extractor type this class implements.</summary>
    public const string Type = "kaldi-fbank";

    // The machine epsilon of float (torch.finfo(torch.float).eps), Kaldi's floor of the energies.
    private const double FloatEpsilon = 1.1920928955078125e-07;

    private readonly int _frameLength;
    private readonly int _fftSize;
    private readonly double _preemphasis;
    private readonly bool _subtractMean;
    private readonly double[] _window;
    private readonly double[] _banks;
    private readonly FourierTransform _transform;

    /// <summary>Creates the extractor from a feature-extractor component's configuration.</summary>
    /// <param name="component">The component (configuration keys in the remarks).</param>
    /// <exception cref="InvalidDataException">The configuration is inconsistent.</exception>
    public KaldiFilterBank(ModelComponent component)
    {
        ArgumentNullException.ThrowIfNull(component);
        SampleRate = component.ConfigValue("sampleRate", 16000);
        _frameLength = component.ConfigValue("frameLength", 400);
        HopLength = component.ConfigValue("frameShift", 160);
        Channels = component.ConfigValue("melBins", 23);
        double low = component.ConfigValue("lowFrequency", 20.0);
        double high = component.ConfigValue("highFrequency", 0.0);
        _preemphasis = component.ConfigValue("preemphasis", 0.97);
        _subtractMean = component.ConfigValue("subtractMean", false);
        double nyquist = SampleRate / 2.0;
        if (high <= 0)
        {
            high += nyquist;
        }

        if (_frameLength <= 0 || HopLength <= 0 || Channels <= 3 || low < 0 || low >= high || high > nyquist)
        {
            throw new InvalidDataException($"kaldi-fbank feature extractor: frameLength {_frameLength}, frameShift {HopLength}, melBins {Channels} "
                + $"and the band {low}..{high} Hz are inconsistent.");
        }

        _fftSize = (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)_frameLength);
        _window = new double[_frameLength];
        for (int i = 0; i < _frameLength; i++)
        {
            double hann = 0.5 - (0.5 * Math.Cos(2 * Math.PI * i / (_frameLength - 1)));
            _window[i] = Math.Pow(hann, 0.85);
        }

        _banks = MelBanks(Channels, _fftSize, SampleRate, low, high);
        _transform = new FourierTransform(_fftSize);
    }

    /// <inheritdoc/>
    public int SampleRate { get; }

    /// <inheritdoc/>
    public int HopLength { get; }

    /// <inheritdoc/>
    public int Channels { get; }

    /// <inheritdoc/>
    public AudioFeatures Extract(ReadOnlySpan<float> samples)
    {
        int frames = samples.Length < _frameLength ? 0 : 1 + ((samples.Length - _frameLength) / HopLength);
        var values = new float[Channels * frames];
        var signal = samples.ToArray();
        int bins = _fftSize / 2;
        Parallel.For(0, frames, () => (Frame: new double[_fftSize], Real: new double[_fftSize], Imaginary: new double[_fftSize], Work: new double[2 * _fftSize]),
            (frame, _, scratch) =>
            {
                var input = scratch.Frame;
                Array.Clear(input);
                int offset = frame * HopLength;
                double mean = 0;
                for (int i = 0; i < _frameLength; i++)
                {
                    mean += signal[offset + i];
                }

                mean /= _frameLength;
                double previous = signal[offset] - mean;
                for (int i = 0; i < _frameLength; i++)
                {
                    double value = signal[offset + i] - mean;
                    input[i] = (value - (_preemphasis * previous)) * _window[i];
                    previous = value;
                }

                _transform.Forward(input, scratch.Real, scratch.Imaginary, scratch.Work);
                for (int m = 0; m < Channels; m++)
                {
                    double energy = 0;
                    for (int k = 0; k < bins; k++)
                    {
                        double weight = _banks[(m * bins) + k];
                        if (weight != 0)
                        {
                            energy += weight * ((scratch.Real[k] * scratch.Real[k]) + (scratch.Imaginary[k] * scratch.Imaginary[k]));
                        }
                    }

                    values[(m * frames) + frame] = (float)Math.Log(Math.Max(energy, FloatEpsilon));
                }

                return scratch;
            }, static _ => { });
        if (_subtractMean && frames > 0)
        {
            for (int m = 0; m < Channels; m++)
            {
                var channel = values.AsSpan(m * frames, frames);
                double sum = 0;
                foreach (float value in channel)
                {
                    sum += value;
                }

                float mean = (float)(sum / frames);
                for (int f = 0; f < frames; f++)
                {
                    channel[f] -= mean;
                }
            }
        }

        return new AudioFeatures(values, Channels, frames);
    }

    private static double MelScale(double frequency) => 1127.0 * Math.Log(1.0 + (frequency / 700.0));

    // Kaldi's triangular bins over the first fftSize / 2 frequency bins (the Nyquist bin gets no weight) [melBins, fftSize / 2].
    private static double[] MelBanks(int melBins, int fftSize, int sampleRate, double low, double high)
    {
        int bins = fftSize / 2;
        double binWidth = sampleRate / (double)fftSize;
        double melLow = MelScale(low);
        double delta = (MelScale(high) - melLow) / (melBins + 1);
        var banks = new double[melBins * bins];
        for (int m = 0; m < melBins; m++)
        {
            double left = melLow + (m * delta);
            double center = melLow + ((m + 1) * delta);
            double right = melLow + ((m + 2) * delta);
            for (int k = 0; k < bins; k++)
            {
                double mel = MelScale(binWidth * k);
                double up = (mel - left) / (center - left);
                double down = (right - mel) / (right - center);
                banks[(m * bins) + k] = Math.Max(0.0, Math.Min(up, down));
            }
        }

        return banks;
    }
}
