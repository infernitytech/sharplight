using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Architectures;

/// <summary>
/// Turns a recording into a Chatterbox voice, following <c>ChatterboxTTS.prepare_conditionals</c>: the recording at 24 kHz and resampled to 16 kHz;
/// the voice encoder's speaker embedding of the 16 kHz signal (trimmed of silence below 20 dB, 40 mel energies, 160-frame partials at 1.3 per
/// second, the mean of the normalized partial embeddings, normalized); T3's prompt speech tokens from its first 6 s (at most 150); and the S3Gen
/// reference of its first 10 s: the 24 kHz log-mel, the CAM++ x-vector and the S3 tokens of that segment resampled to 16 kHz (cut to half the
/// mel frames). Not thread-safe; the synthesizer calls it under its request gate.
/// </summary>
internal sealed class ChatterboxVoiceCloning : IDisposable
{
    private const int Rate16 = 16000;
    private const int Rate24 = 24000;
    private const int PromptTokenLimit = 150;
    private const int T3Seconds = 6;
    private const int FlowSeconds = 10;

    private readonly TaskModel _model;
    private readonly IAudioFeatureExtractor _voiceFeatures;
    private readonly IAudioFeatureExtractor _tokenFeatures;
    private readonly IAudioFeatureExtractor _speakerFeatures;
    private readonly IAudioFeatureExtractor _promptFeatures;
    private readonly int _partialFrames;
    private readonly double _partialRate;
    private readonly double _minCoverage;
    private readonly double _trimDecibels;
    private CompiledModel? _voiceEncoder;
    private CompiledModel? _tokenizer;
    private CompiledModel? _speakerEncoder;

    /// <summary>Creates the cloning stages of a loaded Chatterbox model; nothing is compiled until first use.</summary>
    /// <param name="model">The model (with the cloning components).</param>
    public ChatterboxVoiceCloning(TaskModel model)
    {
        _model = model;
        _voiceFeatures = AudioFeatureExtractors.Create(model, ChatterboxAdapter.VoiceFeaturesRole);
        _tokenFeatures = AudioFeatureExtractors.Create(model, ChatterboxAdapter.SpeechTokenFeaturesRole);
        _speakerFeatures = AudioFeatureExtractors.Create(model, ChatterboxAdapter.SpeakerFeaturesRole);
        _promptFeatures = AudioFeatureExtractors.Create(model, ChatterboxAdapter.PromptFeaturesRole);
        var encoder = model.Description.Component(ChatterboxAdapter.VoiceEncoderRole);
        _partialFrames = encoder.ConfigValue("partialFrames", ChatterboxAdapter.VoicePartialFrames);
        _partialRate = encoder.ConfigValue("partialRate", 1.3);
        _minCoverage = encoder.ConfigValue("minCoverage", 0.8);
        _trimDecibels = encoder.ConfigValue("trimDecibels", 20.0);
    }

    /// <summary>Whether <paramref name="model"/> has the components that clone a voice.</summary>
    /// <param name="model">A loaded Chatterbox model.</param>
    public static bool Supports(TaskModel model)
        => model.Description.TryGetComponent(ChatterboxAdapter.VoiceEncoderRole, out _) && model.Description.TryGetComponent(ChatterboxAdapter.SpeechTokenizerRole, out _)
           && model.Description.TryGetComponent(ChatterboxAdapter.SpeakerEncoderRole, out _);

    /// <summary>The voice of a recording.</summary>
    /// <param name="recording">The recording (any sample rate; channels are mixed down).</param>
    /// <param name="cancellationToken">Stops between stages.</param>
    public ChatterboxVoice Clone(AudioBuffer recording, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(recording);
        var mono = recording.Channels == 1 ? recording.Samples : Mix(recording.Samples, recording.Channels);
        var ref24 = AudioResampler.Resample(mono, 1, recording.SampleRate, Rate24);
        var ref16 = AudioResampler.Resample(ref24, 1, Rate24, Rate16);
        cancellationToken.ThrowIfCancellationRequested();
        var speaker = SpeakerEmbedding(ref16);
        cancellationToken.ThrowIfCancellationRequested();
        var promptTokens = SpeechTokens(ref16.AsSpan(0, Math.Min(ref16.Length, T3Seconds * Rate16)), PromptTokenLimit);
        cancellationToken.ThrowIfCancellationRequested();
        var dec24 = ref24.AsSpan(0, Math.Min(ref24.Length, FlowSeconds * Rate24)).ToArray();
        var dec16 = AudioResampler.Resample(dec24, 1, Rate24, Rate16);
        var promptMel = PromptMel(dec24, out int melFrames);
        var xvector = XVector(dec16);
        cancellationToken.ThrowIfCancellationRequested();
        var flowTokens = SpeechTokens(dec16, maxTokens: null);
        if (flowTokens.Length > melFrames / 2)
        {
            flowTokens = flowTokens[..(melFrames / 2)];
        }

        return new ChatterboxVoice(speaker, promptTokens, flowTokens, promptMel, xvector);
    }

    /// <summary>
    /// The voice encoder's speaker embedding of a 16 kHz signal: trimmed of leading and trailing silence, the mel energies, the mean of the
    /// normalized partial embeddings, normalized.
    /// </summary>
    /// <param name="samples16">The 16 kHz signal.</param>
    public float[] SpeakerEmbedding(ReadOnlySpan<float> samples16)
    {
        var trimmed = Trim(samples16, _trimDecibels);
        var features = _voiceFeatures.Extract(trimmed);
        return SpeakerEmbedding(TimeMajor(features), features.Frames);
    }

    /// <summary>The voice encoder's speaker embedding of mel energies [frames, 40] (time-major).</summary>
    /// <param name="mel">The mel energies.</param>
    /// <param name="frames">The frames.</param>
    public float[] SpeakerEmbedding(float[] mel, int frames)
    {
        var partials = PartialEmbeddings(mel, frames);
        int width = partials[0].Length;
        var mean = new float[width];
        foreach (var partial in partials)
        {
            for (int i = 0; i < width; i++)
            {
                mean[i] += partial[i] / partials.Count;
            }
        }

        return Normalized(mean);
    }

    /// <summary>The normalized embeddings of the 160-frame partials of mel energies [frames, 40] (zero frames pad the last partial).</summary>
    /// <param name="mel">The mel energies.</param>
    /// <param name="frames">The frames.</param>
    public IReadOnlyList<float[]> PartialEmbeddings(float[] mel, int frames)
    {
        int bins = mel.Length / frames;
        int step = (int)Math.Round(Rate16 / _partialRate / _partialFrames);
        int windows = Math.Max(frames - _partialFrames + step, 0) / step;
        int remainder = Math.Max(frames - _partialFrames + step, 0) % step;
        if (windows == 0 || (remainder + (_partialFrames - step)) / (double)_partialFrames >= _minCoverage)
        {
            windows++;
        }

        var encoder = _voiceEncoder ??= _model.CompileComponent(ChatterboxAdapter.VoiceEncoderRole);
        var input = new float[_partialFrames * bins];
        var embeddings = new List<float[]>(windows);
        var output = new float[(int)encoder.Graph.Outputs[ChatterboxVoiceEncoderGraph.EmbeddingOutput].Shape[-1].Fixed];
        for (int w = 0; w < windows; w++)
        {
            Array.Clear(input);
            int first = w * step;
            int count = Math.Clamp(frames - first, 0, _partialFrames);
            mel.AsSpan(first * bins, count * bins).CopyTo(input);
            encoder.SetInput<float>(Input(encoder, ChatterboxVoiceEncoderGraph.MelInput), input);
            encoder.Run(new RunParameters());
            encoder.GetOutput<float>(ChatterboxVoiceEncoderGraph.EmbeddingOutput, output);
            embeddings.Add(Normalized(output));
        }

        return embeddings;
    }

    /// <summary>The S3 speech tokens of a 16 kHz signal (25 per second), at most <paramref name="maxTokens"/>.</summary>
    /// <param name="samples16">The 16 kHz signal.</param>
    /// <param name="maxTokens">The most tokens, or <see langword="null"/>.</param>
    public int[] SpeechTokens(ReadOnlySpan<float> samples16, int? maxTokens)
    {
        var features = _tokenFeatures.Extract(samples16);
        int frames = maxTokens is { } limit ? Math.Min(features.Frames, limit * 4) : features.Frames;
        var mel = TimeMajor(features);
        return SpeechTokens(mel.AsSpan(0, frames * features.Channels).ToArray(), frames);
    }

    /// <summary>The S3 speech tokens of log-mel frames [frames, 128] (time-major).</summary>
    /// <param name="mel">The log-mel frames.</param>
    /// <param name="frames">The frames.</param>
    public int[] SpeechTokens(float[] mel, int frames)
    {
        var tokenizer = _tokenizer ??= _model.CompileComponent(ChatterboxAdapter.SpeechTokenizerRole);
        int tokens = (((frames - 1) / 2) / 2) + 1;
        tokenizer.SetInput<float>(Input(tokenizer, S3TokenizerGraph.FeaturesInput), mel);
        var positions = new int[tokens];
        for (int i = 0; i < tokens; i++)
        {
            positions[i] = i;
        }

        tokenizer.SetInput<int>(Input(tokenizer, S3TokenizerGraph.PositionsInput), positions);
        var run = new RunParameters();
        run.Symbols[WellKnown.TokenSymbol] = frames;
        tokenizer.Run(run);
        var projection = new float[tokens * 8];
        tokenizer.GetOutput<float>(S3TokenizerGraph.ProjectionOutput, projection);
        var result = new int[tokens];
        for (int t = 0; t < tokens; t++)
        {
            // FSQ: tanh, × 0.999, round half to even, + 1, eight ternary digits (the first the least significant).
            int index = 0;
            int power = 1;
            for (int d = 0; d < 8; d++)
            {
                float h = MathF.Tanh(projection[(t * 8) + d]) * 0.9990000128746033f;
                index += ((int)MathF.Round(h, MidpointRounding.ToEven) + 1) * power;
                power *= 3;
            }

            result[t] = index;
        }

        return result;
    }

    /// <summary>The CAM++ x-vector of a 16 kHz signal.</summary>
    /// <param name="samples16">The 16 kHz signal.</param>
    public float[] XVector(ReadOnlySpan<float> samples16)
    {
        var features = _speakerFeatures.Extract(samples16);
        return XVector(TimeMajor(features), features.Frames);
    }

    /// <summary>The CAM++ x-vector of mean-normalized filter-bank frames [frames, 80] (time-major).</summary>
    /// <param name="fbank">The filter-bank frames.</param>
    /// <param name="frames">The frames.</param>
    public float[] XVector(float[] fbank, int frames)
    {
        var encoder = _speakerEncoder ??= _model.CompileComponent(ChatterboxAdapter.SpeakerEncoderRole);
        int strided = ((frames - 1) / 2) + 1;
        int segments = (strided + CamPlusGraph.SegmentLength - 1) / CamPlusGraph.SegmentLength;
        var spans = new int[segments * 2];
        for (int s = 0; s < segments; s++)
        {
            spans[2 * s] = s * CamPlusGraph.SegmentLength;
            spans[(2 * s) + 1] = Math.Min(CamPlusGraph.SegmentLength, strided - (s * CamPlusGraph.SegmentLength));
        }

        var index = new int[strided];
        for (int f = 0; f < strided; f++)
        {
            index[f] = f / CamPlusGraph.SegmentLength;
        }

        encoder.SetInput<float>(Input(encoder, CamPlusGraph.FeaturesInput), fbank);
        encoder.SetInput<int>(Input(encoder, CamPlusGraph.SegmentsInput), spans);
        encoder.SetInput<int>(Input(encoder, CamPlusGraph.SegmentIndexInput), index);
        encoder.SetInput<float>(Input(encoder, CamPlusGraph.VarianceFactorInput), [strided / (float)(strided - 1)]);
        var run = new RunParameters();
        run.Symbols[WellKnown.TokenSymbol] = frames;
        run.Symbols[CamPlusGraph.SegmentSymbol] = segments;
        encoder.Run(run);
        var embedding = new float[(int)encoder.Graph.Outputs[CamPlusGraph.EmbeddingOutput].Shape[-1].Fixed];
        encoder.GetOutput<float>(CamPlusGraph.EmbeddingOutput, embedding);
        return embedding;
    }

    /// <summary>The flow prompt's log-mel [frames, 80] (time-major) of a 24 kHz signal.</summary>
    /// <param name="samples24">The 24 kHz signal.</param>
    /// <param name="frames">The frames.</param>
    public float[] PromptMel(ReadOnlySpan<float> samples24, out int frames)
    {
        var features = _promptFeatures.Extract(samples24);
        frames = features.Frames;
        return TimeMajor(features);
    }

    /// <summary>
    /// librosa's <c>effects.trim</c>: frames of 2048 samples every 512 (centered, zero padded) whose RMS lies within <paramref name="topDecibels"/>
    /// of the loudest frame's; the signal from the first such frame's start to the end of the last one.
    /// </summary>
    /// <param name="samples">The signal.</param>
    /// <param name="topDecibels">The threshold below the peak.</param>
    public static float[] Trim(ReadOnlySpan<float> samples, double topDecibels)
    {
        const int FrameLength = 2048;
        const int Hop = 512;
        int frames = 1 + (samples.Length / Hop);
        var rms = new double[frames];
        double peak = 0;
        for (int f = 0; f < frames; f++)
        {
            double sum = 0;
            int start = (f * Hop) - (FrameLength / 2);
            for (int i = Math.Max(0, start); i < Math.Min(samples.Length, start + FrameLength); i++)
            {
                sum += (double)samples[i] * samples[i];
            }

            rms[f] = Math.Sqrt(sum / FrameLength);
            peak = Math.Max(peak, rms[f]);
        }

        double reference = 20 * Math.Log10(Math.Max(1e-5, peak));
        int first = -1;
        int last = -1;
        for (int f = 0; f < frames; f++)
        {
            if ((20 * Math.Log10(Math.Max(1e-5, rms[f]))) - reference > -topDecibels)
            {
                first = first < 0 ? f : first;
                last = f;
            }
        }

        if (first < 0)
        {
            return [];
        }

        int begin = first * Hop;
        int end = Math.Min(samples.Length, (last + 1) * Hop);
        return samples[begin..end].ToArray();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _voiceEncoder?.Dispose();
        _tokenizer?.Dispose();
        _speakerEncoder?.Dispose();
    }

    private static float[] TimeMajor(AudioFeatures features)
    {
        var result = new float[features.Values.Length];
        for (int c = 0; c < features.Channels; c++)
        {
            for (int f = 0; f < features.Frames; f++)
            {
                result[(f * features.Channels) + c] = features.Values[(c * features.Frames) + f];
            }
        }

        return result;
    }

    private static float[] Normalized(ReadOnlySpan<float> values)
    {
        double norm = 0;
        foreach (float v in values)
        {
            norm += (double)v * v;
        }

        float scale = (float)(1.0 / Math.Sqrt(norm));
        var result = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = values[i] * scale;
        }

        return result;
    }

    private static float[] Mix(float[] interleaved, int channels)
    {
        var mono = new float[interleaved.Length / channels];
        for (int i = 0; i < mono.Length; i++)
        {
            float sum = 0;
            for (int c = 0; c < channels; c++)
            {
                sum += interleaved[(i * channels) + c];
            }

            mono[i] = sum / channels;
        }

        return mono;
    }

    private static Value Input(CompiledModel compiled, string name) => compiled.Graph.Inputs.First(v => v.Input!.Name == name);
}
