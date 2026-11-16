using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>
/// The built-in <see cref="IAudioSeparator"/> over the <see cref="SeparationGraphs"/> contract. The mixture runs in overlapping segments of the model's length (zero-padded around
/// the last one) whose outputs are cross-faded with a triangular weight; per segment the host computes the spectrogram and normalizations, runs the graph, and rebuilds the audio
/// from the spectrogram estimate plus the waveform estimate.
/// </summary>
internal sealed class AudioSeparatorClient : IAudioSeparator
{
    private readonly TaskModel _model;
    private readonly CompiledModel _compiled;
    private readonly Graph.Value _spectrumInput;
    private readonly Graph.Value _waveformInput;
    private readonly SpectralTransform _transform;
    private readonly int _segment;
    private readonly int _hop;
    private readonly int _frames;
    private readonly int _bins;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _disposed;

    public AudioSeparatorClient(TaskModel model)
    {
        _model = model;
        var component = model.Description.Component(ModelComponentRoles.Encoder);
        SampleRate = component.ConfigValue(SeparationGraphs.SampleRateKey, 0);
        Channels = component.ConfigValue(SeparationGraphs.ChannelsKey, 0);
        Stems = component.ConfigValue(SeparationGraphs.StemsKey, "").Split(',');
        _segment = component.ConfigValue(SeparationGraphs.SegmentSamplesKey, 0);
        int fft = component.ConfigValue(SeparationGraphs.FftSizeKey, 0);
        _hop = component.ConfigValue(SeparationGraphs.HopKey, 0);
        _transform = new SpectralTransform(fft, _hop);
        _frames = (_segment + _hop - 1) / _hop;
        _bins = fft / 2;
        _compiled = model.CompileComponent(ModelComponentRoles.Encoder, SeparationGraphs.SeparateGraph);
        _spectrumInput = _compiled.Graph.Inputs.First(v => v.Input!.Name == SeparationGraphs.SpectrumInput);
        _waveformInput = _compiled.Graph.Inputs.First(v => v.Input!.Name == SeparationGraphs.WaveformInput);
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public int SampleRate { get; }

    public int Channels { get; }

    public IReadOnlyList<string> Stems { get; }

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    public Task<IReadOnlyList<SeparatedStem>> SeparateAsync(AudioBuffer mixture, AudioSeparationOptions? options = null, IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(mixture);
        ObjectDisposedException.ThrowIf(_disposed, this);
        float overlap = options?.Overlap ?? 0.25f;
        if (overlap is < 0f or >= 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "The overlap must be at least 0 and below 1.");
        }

        return BackgroundCompute.RunAsync(_gate, () => Separate(mixture.ConvertTo(SampleRate, Channels), overlap, progress, cancellationToken), cancellationToken);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _compiled.Dispose();
        _gate.Dispose();
    }

    private IReadOnlyList<SeparatedStem> Separate(AudioBuffer mixture, float overlap, IProgress<double>? progress, CancellationToken cancellationToken)
    {
        int length = mixture.FrameCount;
        int stems = Stems.Count;
        var channelsOf = new float[Channels][];
        for (int c = 0; c < Channels; c++)
        {
            channelsOf[c] = new float[length];
            for (int i = 0; i < length; i++)
            {
                channelsOf[c][i] = mixture.Samples[(i * Channels) + c];
            }
        }

        var weight = new float[_segment];
        for (int i = 0; i < _segment; i++)
        {
            weight[i] = i < _segment / 2 ? i + 1 : _segment - i;
        }

        float peak = weight.Max();
        for (int i = 0; i < _segment; i++)
        {
            weight[i] /= peak;
        }

        var output = new float[stems * Channels][];
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = new float[length];
        }

        var sums = new float[length];
        int stride = (int)((1 - overlap) * _segment);
        int count = (length + stride - 1) / stride;
        int done = 0;
        for (int offset = 0; offset < length; offset += stride)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int chunk = Math.Min(_segment, length - offset);
            int delta = _segment - chunk;
            var window = new float[Channels][];
            for (int c = 0; c < Channels; c++)
            {
                window[c] = Padded(channelsOf[c], offset, delta);
            }

            var estimate = RunSegment(window);
            int trim = delta / 2;
            for (int s = 0; s < output.Length; s++)
            {
                for (int i = 0; i < chunk; i++)
                {
                    output[s][offset + i] += weight[i] * estimate[s][trim + i];
                }
            }

            for (int i = 0; i < chunk; i++)
            {
                sums[offset + i] += weight[i];
            }

            progress?.Report((double)(++done) / count);
        }

        var result = new SeparatedStem[stems];
        for (int s = 0; s < stems; s++)
        {
            var samples = new float[length * Channels];
            for (int c = 0; c < Channels; c++)
            {
                var source = output[(s * Channels) + c];
                for (int i = 0; i < length; i++)
                {
                    samples[(i * Channels) + c] = source[i] / sums[i];
                }
            }

            result[s] = new SeparatedStem(Stems[s], new AudioBuffer(samples, SampleRate, Channels));
        }

        return result;
    }

    // TensorChunk.padded: the chunk centered in a window of the segment length, with the neighbouring samples where the mixture has them and zeros elsewhere.
    private float[] Padded(float[] signal, int offset, int delta)
    {
        int start = offset - (delta / 2);
        var window = new float[_segment];
        for (int i = 0; i < _segment; i++)
        {
            int j = start + i;
            window[i] = j >= 0 && j < signal.Length ? signal[j] : 0f;
        }

        return window;
    }

    // One segment: [channels][segment] in, [stems · channels][segment] out.
    private float[][] RunSegment(float[][] window)
    {
        int channels = Channels;
        int padLeft = _hop / 2 * 3;
        int padTotal = (_frames * _hop) - _segment;
        var spectrum = new float[2 * channels * _bins * _frames];
        var real = new float[(_bins + 1) * (_transform.Frames(_segment + (2 * padLeft) + padTotal))];
        var imaginary = new float[real.Length];
        int allFrames = _transform.Frames(_segment + (2 * padLeft) + padTotal);
        for (int c = 0; c < channels; c++)
        {
            var padded = SpectralTransform.ReflectPad(window[c], padLeft, padLeft + padTotal);
            _transform.Forward(padded, real, imaginary);
            for (int k = 0; k < _bins; k++)
            {
                for (int t = 0; t < _frames; t++)
                {
                    spectrum[((((2 * c) * _bins) + k) * _frames) + t] = real[(k * allFrames) + t + 2];
                    spectrum[((((2 * c) + 1) * _bins + k) * _frames) + t] = imaginary[(k * allFrames) + t + 2];
                }
            }
        }

        var waveform = new float[channels * _segment];
        for (int c = 0; c < channels; c++)
        {
            window[c].CopyTo(waveform, c * _segment);
        }

        // Normalization (unbiased standard deviation over all channels and bins / samples).
        (double mean, double deviation) = MeanAndDeviation(spectrum);
        (double meanTime, double deviationTime) = MeanAndDeviation(waveform);
        for (int i = 0; i < spectrum.Length; i++)
        {
            spectrum[i] = (float)((spectrum[i] - mean) / (1e-5 + deviation));
        }

        for (int i = 0; i < waveform.Length; i++)
        {
            waveform[i] = (float)((waveform[i] - meanTime) / (1e-5 + deviationTime));
        }

        _compiled.SetInput<float>(_spectrumInput, spectrum);
        _compiled.SetInput<float>(_waveformInput, waveform);
        _compiled.Run(new RunParameters());
        var spectrumOut = new float[Stems.Count * 2 * channels * _bins * _frames];
        var waveformOut = new float[Stems.Count * channels * _segment];
        _compiled.GetOutput<float>(SeparationGraphs.SpectrumOutput, spectrumOut);
        _compiled.GetOutput<float>(SeparationGraphs.WaveformOutput, waveformOut);

        var result = new float[Stems.Count * channels][];
        int frames = _frames;
        int fullFrames = frames + 4;
        int fullBins = _bins + 1;
        int paddedLength = (_hop * frames) + (2 * padLeft);
        var re = new float[fullBins * fullFrames];
        var im = new float[fullBins * fullFrames];
        for (int s = 0; s < Stems.Count; s++)
        {
            for (int c = 0; c < channels; c++)
            {
                Array.Clear(re);
                Array.Clear(im);
                int baseChannel = ((s * channels) + c) * 2;
                for (int k = 0; k < _bins; k++)
                {
                    for (int t = 0; t < frames; t++)
                    {
                        re[(k * fullFrames) + t + 2] = (float)((spectrumOut[(((baseChannel * _bins) + k) * frames) + t] * deviation) + mean);
                        im[(k * fullFrames) + t + 2] = (float)((spectrumOut[((((baseChannel + 1) * _bins) + k) * frames) + t] * deviation) + mean);
                    }
                }

                var time = _transform.Inverse(re, im, fullFrames, paddedLength);
                var signal = new float[_segment];
                int index = (s * channels) + c;
                for (int i = 0; i < _segment; i++)
                {
                    signal[i] = (float)(time[padLeft + i] + (waveformOut[(index * _segment) + i] * deviationTime) + meanTime);
                }

                result[index] = signal;
            }
        }

        return result;
    }

    private static (double Mean, double Deviation) MeanAndDeviation(float[] values)
    {
        double sum = 0;
        foreach (float v in values)
        {
            sum += v;
        }

        double mean = sum / values.Length;
        double squares = 0;
        foreach (float v in values)
        {
            squares += (v - mean) * (v - mean);
        }

        return (mean, Math.Sqrt(squares / (values.Length - 1)));
    }
}
