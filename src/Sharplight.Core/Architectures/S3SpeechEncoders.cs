using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Architectures;

/// <summary>
/// The encoders that CosyVoice-family models (CosyVoice 2 and 3, Chatterbox's S3Gen) share to take a voice from a recording: the S3 speech tokens of a
/// 16 kHz signal (the Whisper-style log-mel through the S3 tokenizer graph and its finite scalar quantizer), the CAM++ x-vector (the mean-subtracted
/// Kaldi filter bank through <see cref="CamPlusGraph"/>) and the flow prompt's Matcha log-mel of a 24 kHz signal. The components are found by role;
/// nothing is compiled until first use. Not thread-safe.
/// </summary>
internal sealed class S3SpeechEncoders : IDisposable
{
    /// <summary>The role of the S3 speech tokenizer (speech tokens of a recording).</summary>
    public const string SpeechTokenizerRole = "speech-tokenizer";

    /// <summary>The role of the CAM++ speaker encoder (x-vector of a recording).</summary>
    public const string SpeakerEncoderRole = "speaker-encoder";

    /// <summary>The role of the speech tokenizer's front end (Whisper-style log-mel at 16 kHz).</summary>
    public const string SpeechTokenFeaturesRole = "speech-token-features";

    /// <summary>The role of the speaker encoder's front end (Kaldi filter bank at 16 kHz).</summary>
    public const string SpeakerFeaturesRole = "speaker-features";

    /// <summary>The role of the flow prompt's front end (Matcha log-mel at 24 kHz).</summary>
    public const string PromptFeaturesRole = "prompt-features";

    private readonly TaskModel _model;
    private readonly IAudioFeatureExtractor _tokenFeatures;
    private readonly IAudioFeatureExtractor _speakerFeatures;
    private readonly IAudioFeatureExtractor _promptFeatures;
    private CompiledModel? _tokenizer;
    private CompiledModel? _speakerEncoder;

    /// <summary>Creates the encoders of a loaded model.</summary>
    /// <param name="model">The model (with the speech tokenizer, speaker encoder and front-end components).</param>
    public S3SpeechEncoders(TaskModel model)
    {
        _model = model;
        _tokenFeatures = AudioFeatureExtractors.Create(model, SpeechTokenFeaturesRole);
        _speakerFeatures = AudioFeatureExtractors.Create(model, SpeakerFeaturesRole);
        _promptFeatures = AudioFeatureExtractors.Create(model, PromptFeaturesRole);
    }

    /// <summary>Whether <paramref name="model"/> has the speech tokenizer and the speaker encoder.</summary>
    /// <param name="model">A loaded model.</param>
    public static bool Supports(TaskModel model)
        => model.Description.TryGetComponent(SpeechTokenizerRole, out _) && model.Description.TryGetComponent(SpeakerEncoderRole, out _);

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
        var tokenizer = _tokenizer ??= _model.CompileComponent(SpeechTokenizerRole);
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
        var encoder = _speakerEncoder ??= _model.CompileComponent(SpeakerEncoderRole);
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

    /// <inheritdoc/>
    public void Dispose()
    {
        _tokenizer?.Dispose();
        _speakerEncoder?.Dispose();
    }

    /// <summary>Channel-major features as time-major rows [frames, channels].</summary>
    /// <param name="features">The features.</param>
    public static float[] TimeMajor(AudioFeatures features)
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
