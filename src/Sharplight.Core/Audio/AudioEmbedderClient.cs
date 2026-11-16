using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Tokenizers;

namespace Sharplight.Audio;

/// <summary>The built-in <see cref="IAudioEmbedder"/> over the <see cref="AudioEmbedderGraphs"/> contract.</summary>
internal sealed class AudioEmbedderClient : IAudioEmbedder
{
    private readonly TaskModel _model;
    private readonly IAudioFeatureExtractor _features;
    private readonly CompiledModel _audio;
    private readonly Graph.Value _featuresInput;
    private readonly CompiledModel? _text;
    private readonly Graph.Value? _tokens;
    private readonly Graph.Value? _positions;
    private readonly ITokenizer? _tokenizer;
    private readonly int _clipSamples;
    private readonly int _frames;
    private readonly int _positionOffset;
    private readonly int _maxTokens;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _disposed;

    public AudioEmbedderClient(TaskModel model)
    {
        _model = model;
        var encoder = model.Description.Component(ModelComponentRoles.Encoder);
        SampleRate = encoder.ConfigValue(AudioEmbedderGraphs.SampleRateKey, 0);
        _clipSamples = encoder.ConfigValue(AudioEmbedderGraphs.ClipSamplesKey, 0);
        _frames = encoder.ConfigValue(AudioEmbedderGraphs.FramesKey, 0);
        Dimensions = encoder.ConfigValue(AudioEmbedderGraphs.DimensionsKey, 0);
        _features = AudioFeatureExtractors.Create(model);
        _audio = model.CompileComponent(ModelComponentRoles.Encoder, AudioEmbedderGraphs.AudioGraph);
        _featuresInput = _audio.Graph.Inputs.First(v => v.Input!.Name == AudioEmbedderGraphs.FeaturesInput);
        if (model.Description.TryGetComponent(ModelComponentRoles.TextEncoder, out var text) && text.Graphs.Contains(AudioEmbedderGraphs.TextGraph))
        {
            _positionOffset = text.ConfigValue(AudioEmbedderGraphs.PositionOffsetKey, 0);
            _maxTokens = text.ConfigValue(AudioEmbedderGraphs.MaxTokensKey, 512);
            _tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
            _text = model.CompileComponent(ModelComponentRoles.TextEncoder, AudioEmbedderGraphs.TextGraph);
            _tokens = _text.Graph.Inputs.First(v => v.Input!.Name == AudioEmbedderGraphs.TokensInput);
            _positions = _text.Graph.Inputs.First(v => v.Input!.Name == AudioEmbedderGraphs.PositionsInput);
        }

        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public int Dimensions { get; }

    public int SampleRate { get; }

    public TimeSpan ClipDuration => TimeSpan.FromSeconds((double)_clipSamples / SampleRate);

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    public Task<float[]> EmbedAudioAsync(AudioBuffer audio, AudioEmbeddingOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ObjectDisposedException.ThrowIf(_disposed, this);
        options ??= new AudioEmbeddingOptions();
        return BackgroundCompute.RunAsync(_gate, () => EmbedAudio(audio, options.LongAudio, options.Normalize, cancellationToken), cancellationToken);
    }

    public Task<float[]> EmbedTextAsync(string text, AudioEmbeddingOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        ObjectDisposedException.ThrowIf(_disposed, this);
        RequireText();
        bool normalize = options?.Normalize ?? true;
        return BackgroundCompute.RunAsync(_gate, () => Normalized(EmbedText(text), normalize), cancellationToken);
    }

    public Task<AudioClassification> ClassifyZeroShotAsync(AudioBuffer audio, IReadOnlyList<string> candidates, AudioClassificationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentNullException.ThrowIfNull(candidates);
        ObjectDisposedException.ThrowIf(_disposed, this);
        RequireText();
        options ??= new AudioClassificationOptions();
        return BackgroundCompute.RunAsync(_gate, () =>
        {
            var embedding = Normalized(EmbedAudio(audio, options.LongAudio, cancellationToken, out float scale), true);
            var logits = new float[candidates.Count];
            for (int i = 0; i < logits.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var text = Normalized(EmbedText(candidates[i]), true);
                double dot = 0;
                for (int d = 0; d < text.Length; d++)
                {
                    dot += (double)embedding[d] * text[d];
                }

                logits[i] = (float)(dot * scale);
            }

            return Rank(logits, candidates, options.TopK);
        }, cancellationToken);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _audio.Dispose();
        _text?.Dispose();
        _gate.Dispose();
    }

    private static AudioClassification Rank(float[] logits, IReadOnlyList<string> labels, int topK)
    {
        var scores = new float[logits.Length];
        float max = logits.Length > 0 ? logits.Max() : 0f;
        float sum = 0f;
        for (int i = 0; i < logits.Length; i++)
        {
            scores[i] = MathF.Exp(logits[i] - max);
            sum += scores[i];
        }

        for (int i = 0; i < logits.Length; i++)
        {
            scores[i] /= sum;
        }

        int count = topK <= 0 ? logits.Length : Math.Min(topK, logits.Length);
        var best = Enumerable.Range(0, logits.Length).OrderByDescending(i => scores[i]).ThenBy(i => i).Take(count)
            .Select(i => new AudioLabelScore(i, labels[i], scores[i], logits[i])).ToArray();
        return new AudioClassification(best);
    }

    private static float[] Normalized(float[] vector, bool normalize)
    {
        if (!normalize)
        {
            return vector;
        }

        double sum = 0;
        foreach (float v in vector)
        {
            sum += (double)v * v;
        }

        float inverse = (float)(1.0 / Math.Max(Math.Sqrt(sum), 1e-12));
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] *= inverse;
        }

        return vector;
    }

    private void RequireText()
    {
        if (_text is null)
        {
            throw new NotSupportedException("The model has no text tower.");
        }
    }

    private float[] EmbedAudio(AudioBuffer audio, LongAudioMode mode, bool normalize, CancellationToken cancellationToken)
        => Normalized(EmbedAudio(audio, mode, cancellationToken, out _), normalize);

    private float[] EmbedAudio(AudioBuffer audio, LongAudioMode mode, CancellationToken cancellationToken, out float scale)
    {
        var samples = audio.ToModelInput(SampleRate).Samples;
        int clips = mode == LongAudioMode.Average ? Math.Max(1, (samples.Length + _clipSamples - 1) / _clipSamples) : 1;
        var sum = new float[Dimensions];
        scale = 1f;
        for (int c = 0; c < clips; c++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int from = c * _clipSamples;
            var clip = RepeatPad(samples.AsSpan(from, Math.Min(_clipSamples, samples.Length - from)));
            var embedding = RunAudio(clip, out scale);
            for (int d = 0; d < sum.Length; d++)
            {
                sum[d] += embedding[d] / clips;
            }
        }

        return sum;
    }

    // The reference's "repeatpad": the clip is repeated as often as it fits, then padded with zeros.
    private float[] RepeatPad(ReadOnlySpan<float> samples)
    {
        var clip = new float[_clipSamples];
        if (samples.Length == 0)
        {
            return clip;
        }

        int repeats = Math.Max(1, _clipSamples / samples.Length);
        for (int r = 0; r < repeats && (r + 1) * samples.Length <= _clipSamples; r++)
        {
            samples.CopyTo(clip.AsSpan(r * samples.Length));
        }

        if (samples.Length > _clipSamples)
        {
            samples[.._clipSamples].CopyTo(clip);
        }

        return clip;
    }

    private float[] RunAudio(float[] clip, out float scale)
    {
        var features = _features.Extract(clip);
        var input = new float[_frames * features.Channels];
        for (int t = 0; t < Math.Min(_frames, features.Frames); t++)
        {
            for (int ch = 0; ch < features.Channels; ch++)
            {
                input[(t * features.Channels) + ch] = features.Values[(ch * features.Frames) + t];
            }
        }

        _audio.SetInput<float>(_featuresInput, input);
        _audio.Run(new RunParameters());
        var embedding = new float[Dimensions];
        _audio.GetOutput<float>(AudioEmbedderGraphs.EmbeddingOutput, embedding);
        var logitScale = new float[1];
        scale = 1f;
        if (_audio.Graph.Outputs.ContainsKey(AudioEmbedderGraphs.LogitScaleOutput))
        {
            _audio.GetOutput<float>(AudioEmbedderGraphs.LogitScaleOutput, logitScale);
            scale = logitScale[0];
        }

        return embedding;
    }

    private float[] EmbedText(string text)
    {
        var ids = _tokenizer!.Encode(text);
        if (ids.Length > _maxTokens)
        {
            int eos = _tokenizer.EosId ?? -1;
            ids = ids[.._maxTokens];
            if (eos >= 0 && _tokenizer.AddEos)
            {
                ids[^1] = eos;
            }
        }

        var positions = new int[ids.Length];
        for (int i = 0; i < positions.Length; i++)
        {
            positions[i] = _positionOffset + i;
        }

        _text!.SetInput<int>(_tokens!, ids);
        _text.SetInput<int>(_positions!, positions);
        var parameters = new RunParameters();
        parameters.Symbols[WellKnown.TokenSymbol] = ids.Length;
        _text.Run(parameters);
        var embedding = new float[Dimensions];
        _text.GetOutput<float>(AudioEmbedderGraphs.EmbeddingOutput, embedding);
        return embedding;
    }
}
