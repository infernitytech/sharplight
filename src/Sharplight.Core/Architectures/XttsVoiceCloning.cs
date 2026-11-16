using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Architectures;

/// <summary>
/// Turns a recording into an XTTS v2 voice, following <c>Xtts.get_conditioning_latents</c> with the model's settings: the recording at 22.05 kHz
/// (at most 30 s); the speaker embedding of its 16 kHz resampling (the speaker encoder over 64 log-mel channels, L2-normalized); and the GPT's
/// conditioning latents, the mean of the style encoder's 32 latents over consecutive 4 s chunks (chunks shorter than 0.33 s are skipped).
/// Not thread-safe; the synthesizer calls it under its request gate.
/// </summary>
internal sealed class XttsVoiceCloning : IDisposable
{
    private const int Rate22 = 22050;
    private const int Rate16 = 16000;

    private readonly TaskModel _model;
    private readonly IAudioFeatureExtractor _styleFeatures;
    private readonly IAudioFeatureExtractor _speakerFeatures;
    private readonly int _maxSeconds;
    private readonly int _chunkSeconds;
    private CompiledModel? _style;
    private CompiledModel? _speaker;

    /// <summary>Creates the cloning stages of a loaded XTTS model; nothing is compiled until first use.</summary>
    /// <param name="model">The model (with the cloning components).</param>
    public XttsVoiceCloning(TaskModel model)
    {
        _model = model;
        _styleFeatures = AudioFeatureExtractors.Create(model, XttsAdapter.StyleFeaturesRole);
        _speakerFeatures = AudioFeatureExtractors.Create(model, XttsAdapter.SpeakerFeaturesRole);
        var style = model.Description.Component(XttsAdapter.StyleEncoderRole);
        _maxSeconds = style.ConfigValue("maxSeconds", 30);
        _chunkSeconds = style.ConfigValue("chunkSeconds", 4);
    }

    /// <summary>Whether <paramref name="model"/> has the components that clone a voice.</summary>
    /// <param name="model">A loaded XTTS model.</param>
    public static bool Supports(TaskModel model)
        => model.Description.TryGetComponent(XttsAdapter.StyleEncoderRole, out _) && model.Description.TryGetComponent(XttsAdapter.SpeakerEncoderRole, out _);

    /// <summary>The voice of a recording.</summary>
    /// <param name="recording">The recording (any sample rate; channels are mixed down).</param>
    /// <param name="cancellationToken">Stops between stages.</param>
    /// <exception cref="ArgumentException">The recording is shorter than a third of a second.</exception>
    public XttsVoice Clone(AudioBuffer recording, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(recording);
        var mono = recording.ToMono().Samples;
        var audio22 = AudioResampler.Resample(mono, 1, recording.SampleRate, Rate22);
        audio22 = audio22.AsSpan(0, Math.Min(audio22.Length, _maxSeconds * Rate22)).ToArray();
        for (int i = 0; i < audio22.Length; i++)
        {
            audio22[i] = Math.Clamp(audio22[i], -1f, 1f);
        }

        var speaker = SpeakerEmbedding(AudioResampler.Resample(audio22, 1, Rate22, Rate16));
        cancellationToken.ThrowIfCancellationRequested();
        return new XttsVoice(Latents(audio22, cancellationToken), speaker);
    }

    /// <summary>The conditioning latents [32 · width] of a 22.05 kHz signal: the mean over its 4 s chunks.</summary>
    /// <param name="samples22">The signal.</param>
    /// <param name="cancellationToken">Stops between chunks.</param>
    /// <exception cref="ArgumentException">The signal is shorter than a third of a second.</exception>
    public float[] Latents(ReadOnlySpan<float> samples22, CancellationToken cancellationToken = default)
    {
        int step = _chunkSeconds * Rate22;
        float[]? sum = null;
        int chunks = 0;
        for (int start = 0; start < samples22.Length; start += step)
        {
            var chunk = samples22.Slice(start, Math.Min(step, samples22.Length - start));
            if (chunk.Length < Rate22 * 0.33)
            {
                continue;
            }

            cancellationToken.ThrowIfCancellationRequested();
            var features = _styleFeatures.Extract(chunk);
            var latents = StyleLatents(TimeMajor(features), features.Frames);
            sum ??= new float[latents.Length];
            for (int i = 0; i < latents.Length; i++)
            {
                sum[i] += latents[i];
            }

            chunks++;
        }

        if (sum is null)
        {
            throw new ArgumentException("The recording is too short to clone a voice from (at least 0.33 s).", nameof(samples22));
        }

        for (int i = 0; i < sum.Length; i++)
        {
            sum[i] /= chunks;
        }

        return sum;
    }

    /// <summary>The style encoder's 32 latents [32 · width] of one chunk's log-mel [frames, 80] (time-major, before the mel statistics).</summary>
    /// <param name="mel">The log-mel frames.</param>
    /// <param name="frames">The frames.</param>
    public float[] StyleLatents(float[] mel, int frames)
    {
        var style = _style ??= _model.CompileComponent(XttsAdapter.StyleEncoderRole);
        style.SetInput<float>(Input(style, XttsConditioningGraph.MelInput), mel);
        var run = new RunParameters();
        run.Symbols[WellKnown.TokenSymbol] = frames;
        style.Run(run);
        var output = style.Graph.Outputs[XttsConditioningGraph.LatentsOutput];
        var latents = new float[output.Shape[0].Fixed * output.Shape[1].Fixed];
        style.GetOutput<float>(XttsConditioningGraph.LatentsOutput, latents);
        return latents;
    }

    /// <summary>The L2-normalized speaker embedding of a 16 kHz signal.</summary>
    /// <param name="samples16">The signal.</param>
    public float[] SpeakerEmbedding(ReadOnlySpan<float> samples16)
    {
        var features = _speakerFeatures.Extract(samples16);
        return SpeakerEmbedding(TimeMajor(features), features.Frames);
    }

    /// <summary>The L2-normalized speaker embedding of log-mel frames [frames, 64] (time-major).</summary>
    /// <param name="mel">The log-mel frames.</param>
    /// <param name="frames">The frames.</param>
    public float[] SpeakerEmbedding(float[] mel, int frames)
    {
        var speaker = _speaker ??= _model.CompileComponent(XttsAdapter.SpeakerEncoderRole);
        speaker.SetInput<float>(Input(speaker, XttsSpeakerEncoderGraph.MelInput), mel);
        var run = new RunParameters();
        run.Symbols[WellKnown.TokenSymbol] = frames;
        speaker.Run(run);
        var embedding = new float[(int)speaker.Graph.Outputs[XttsSpeakerEncoderGraph.EmbeddingOutput].Shape[-1].Fixed];
        speaker.GetOutput<float>(XttsSpeakerEncoderGraph.EmbeddingOutput, embedding);
        double norm = Math.Sqrt(embedding.Sum(v => (double)v * v));
        for (int i = 0; i < embedding.Length; i++)
        {
            embedding[i] = (float)(embedding[i] / Math.Max(norm, 1e-12));
        }

        return embedding;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _style?.Dispose();
        _speaker?.Dispose();
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

    private static Value Input(CompiledModel compiled, string name) => compiled.Graph.Inputs.First(v => v.Input!.Name == name);
}
