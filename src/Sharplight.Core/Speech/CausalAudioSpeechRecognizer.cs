using System.Runtime.CompilerServices;
using Sharplight.Architectures;
using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Tokenizers;
using Sharplight.Vision;

namespace Sharplight.Speech;

/// <summary>
/// The <see cref="ISpeechRecognizer"/> of causal streaming audio language models (Voxtral Realtime, <see cref="AudioTowerParameters.Causal"/>). The
/// recording is padded as the model was trained (silence before it, to a whole number of steps and after it), the causal tower encodes it in one pass
/// into one embedding row per decoder step (80 ms), and the language model then runs one step per row: its input is the previous token's embedding plus
/// the row of the step. Most steps emit the pad token; a word is emitted a fixed number of steps (the delay) after it was spoken, preceded by the word
/// marker. The prompt is the start token and the pad token for the silent prefix and the delay. Recordings longer than the window are cut into
/// consecutive windows; <see cref="TranscribeStreamingAsync"/> yields one segment per window.
/// </summary>
internal sealed class CausalAudioSpeechRecognizer : ISpeechRecognizer
{
    private readonly TaskModel _model;
    private readonly IAudioFeatureExtractor _features;
    private readonly AudioPromptSettings _prompt;
    private readonly LlmModel _llm;
    private readonly ITokenizer _tokenizer;
    private readonly IServiceProvider? _services;
    private readonly int _maxFrames;
    private readonly int _framesPerToken;
    private readonly int _leftPad;
    private readonly int _delay;
    private readonly int _rightPad;
    private readonly int _positionAxes;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private CompiledModel? _encoder;
    private InferenceSession? _session;
    private bool _disposed;

    /// <summary>Creates the recognizer; nothing is compiled until the first request.</summary>
    /// <param name="model">A causal audio language model (<see cref="AudioLanguageModelDescription"/>).</param>
    /// <param name="services">The container a voice activity detector is resolved from, or <see langword="null"/>.</param>
    public CausalAudioSpeechRecognizer(TaskModel model, IServiceProvider? services = null)
    {
        _model = model;
        _services = services;
        _features = AudioFeatureExtractors.Create(model);
        var encoder = model.Description.Component(ModelComponentRoles.Encoder);
        _maxFrames = encoder.ConfigValue("maxFrames", 6000);
        _framesPerToken = encoder.ConfigValue("framesPerToken", 8);
        _leftPad = encoder.ConfigValue("leftPadTokens", 0);
        _delay = encoder.ConfigValue("delayTokens", 0);
        _rightPad = encoder.ConfigValue("rightPadTokens", 0);
        _prompt = AudioPromptSettings.FromComponent(model.Description.Component(ModelComponentRoles.Decoder));
        _llm = LlmModel.FromTaskModel(model);
        _tokenizer = _llm.Tokenizer;
        _positionAxes = model.Definition.Property(WellKnown.PositionAxes, 1);
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    /// <inheritdoc/>
    public TaskClientMetadata Metadata { get; }

    /// <inheritdoc/>
    public int SampleRate => _features.SampleRate;

    /// <summary>Whether <paramref name="model"/> is a causal audio language model this recognizer runs.</summary>
    /// <param name="model">A loaded speech-recognition model.</param>
    public static bool Handles(TaskModel model)
    {
        var description = model.Description;
        return description.TryGetComponent(ModelComponentRoles.Encoder, out var encoder)
            && encoder.ConfigValue("tower", "") == AudioTowerParameters.Causal
            && description.TryGetComponent(ModelComponentRoles.Decoder, out var decoder) && AudioPromptSettings.IsPresent(decoder)
            && description.TryGetComponent(ModelComponentRoles.FeatureExtractor, out _);
    }

    /// <inheritdoc/>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
        {
            return null;
        }

        return serviceType.IsInstanceOfType(this) ? this : serviceType.IsInstanceOfType(_model) ? _model : serviceType.IsInstanceOfType(_llm) ? _llm : null;
    }

    /// <inheritdoc/>
    public async Task<Transcription> TranscribeAsync(AudioBuffer audio, SpeechRecognitionOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        var segments = new List<TranscriptionSegment>();
        await foreach (var segment in TranscribeStreamingAsync(audio, options, cancellationToken).ConfigureAwait(false))
        {
            segments.Add(segment);
        }

        string text = string.Join(" ", segments.Select(s => s.Text.Trim()).Where(t => t.Length > 0));
        return new Transcription(text, options?.Language, segments, audio.Duration);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<TranscriptionSegment> TranscribeStreamingAsync(AudioBuffer audio, SpeechRecognitionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ObjectDisposedException.ThrowIf(_disposed, this);
        options ??= SpeechRecognitionOptions.Default;
        if (options.Translate)
        {
            throw new NotSupportedException("This model transcribes only; it does not translate.");
        }

        if (options.UseVoiceActivityDetection)
        {
            await foreach (var segment in VoiceActivityPresegmentation.TranscribeAsync(this, _services, audio, options, cancellationToken).ConfigureAwait(false))
            {
                yield return segment;
            }

            yield break;
        }

        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            _encoder ??= await Task.Run(() => _model.CompileComponent(ModelComponentRoles.Encoder), cancellationToken).ConfigureAwait(false);
            _session ??= await Task.Run(() => _llm.CreateSession(WellKnown.DecodeMultimodalGraph), cancellationToken).ConfigureAwait(false);
            var samples = audio.ToModelInput(SampleRate).Samples;
            int step = _framesPerToken * _features.HopLength;
            int padding = (_leftPad + _rightPad) * step;
            int window = (_maxFrames * _features.HopLength) - padding;
            window -= window % step;
            for (int start = 0; start < samples.Length || start == 0; start += window)
            {
                int first = start;
                int length = Math.Min(window, samples.Length - first);
                var segment = await Task.Run(() => RecognizeWindow(samples.AsSpan(first, length).ToArray(), first, options, cancellationToken), cancellationToken).ConfigureAwait(false);
                if (segment is not null)
                {
                    yield return segment;
                }

                if (samples.Length == 0)
                {
                    break;
                }
            }
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>The audio embedding rows of a padded recording ([rows, hidden] row-major, one per decoder step; compiles the encoder on first use).</summary>
    /// <param name="samples">The recording (unpadded).</param>
    public float[] Encode(ReadOnlySpan<float> samples)
    {
        var padded = Pad(samples);
        var features = _features.Extract(padded);
        return EncodeFeatures(features);
    }

    /// <summary>The prompt: the start token, then the pad token for the silent prefix and the delay.</summary>
    public int[] PromptTokens()
    {
        var tokens = new List<int>(_tokenizer.Encode("<s>", addSpecial: false, parseSpecial: true));
        tokens.AddRange(Enumerable.Repeat(_prompt.AudioToken, _leftPad + _delay));
        return [.. tokens];
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _session?.Dispose();
        _encoder?.Dispose();
        _gate.Dispose();
    }

    // The recording as the model saw it in training: silence for the prefix steps, the audio rounded up to whole steps, silence for the delay and a buffer.
    private float[] Pad(ReadOnlySpan<float> samples)
    {
        int step = _framesPerToken * _features.HopLength;
        int audio = (samples.Length + step - 1) / step * step;
        var padded = new float[(_leftPad * step) + audio + (_rightPad * step)];
        samples.CopyTo(padded.AsSpan(_leftPad * step));
        return padded;
    }

    private float[] EncodeFeatures(AudioFeatures features)
    {
        var encoder = _encoder ??= _model.CompileComponent(ModelComponentRoles.Encoder);
        int frames = features.Frames;
        int channels = features.Channels;
        var map = new float[(long)frames * channels];
        var values = features.Values;
        for (int t = 0; t < frames; t++)
        {
            for (int m = 0; m < channels; m++)
            {
                map[((long)t * channels) + m] = values[((long)m * frames) + t];
            }
        }

        var graph = encoder.Graph;
        Value Input(string name) => graph.Inputs.First(v => v.Input!.Name == name);
        int rows = frames / 2;
        int window = _model.Description.Component(ModelComponentRoles.Encoder).ConfigValue("slidingWindow", 0);
        encoder.SetInput<float>(Input(ChunkedAudioTowerGraph.FeaturesInput), map);
        var positions = new int[rows];
        var spans = new int[rows * 2];
        for (int i = 0; i < rows; i++)
        {
            positions[i] = i;
            int from = window > 0 ? Math.Max(0, i - window + 1) : 0;
            spans[2 * i] = from;
            spans[(2 * i) + 1] = i + 1 - from;
        }

        encoder.SetInput<int>(Input(ChunkedAudioTowerGraph.PositionsInput), positions);
        encoder.SetInput<int>(Input(ChunkedAudioTowerGraph.SpansInput), spans);
        var run = new RunParameters();
        run.Symbols[WellKnown.AudioFrameSymbol] = frames;
        int outRows = frames / _framesPerToken;
        run.Symbols[WellKnown.EncoderSymbol] = outRows;
        encoder.Run(run);
        int width = (int)graph.Outputs[ChunkedAudioTowerGraph.EmbeddingsOutput].Shape[1].Fixed;
        var embeddings = new float[(long)outRows * width];
        encoder.GetOutput<float>(ChunkedAudioTowerGraph.EmbeddingsOutput, embeddings);
        return embeddings;
    }

    private TranscriptionSegment? RecognizeWindow(float[] samples, int firstSample, SpeechRecognitionOptions options, CancellationToken cancellationToken)
    {
        var embeddings = Encode(samples);
        int width = _model.Definition.Property(WellKnown.EmbeddingLength, 0);
        int rows = embeddings.Length / width;
        var prompt = PromptTokens();
        var session = _session!;
        session.Reset();
        var sampler = new Sampler(options.Sampling);
        foreach (int t in prompt)
        {
            sampler.Accept(t);
        }

        var generated = new List<int>();
        double logProbabilities = 0;
        int budget = Math.Min(rows, session.ContextLength) - prompt.Length;
        var logits = session.Process(Rows(prompt, 0, session.RopePosition), [embeddings.AsSpan(0, prompt.Length * width).ToArray()]);
        for (int i = 0; i < budget; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int id = sampler.Sample(logits);
            if (_tokenizer.IsEndOfGeneration(id))
            {
                break;
            }

            logProbabilities += LogProbabilities.Of(logits, id);
            sampler.Accept(id);
            generated.Add(id);
            if (i + 1 < budget)
            {
                int row = prompt.Length + i;
                logits = session.Process(Rows([id], 0, session.RopePosition), [embeddings.AsSpan(row * width, width).ToArray()]);
            }
        }

        string text = _tokenizer.Decode([.. generated], skipSpecial: true).Trim();
        if (text.Length == 0)
        {
            return null;
        }

        double stepSeconds = (double)_framesPerToken * _features.HopLength / SampleRate;
        double start = (double)firstSample / SampleRate;
        return new TranscriptionSegment
        {
            Start = TimeSpan.FromSeconds(start),
            End = TimeSpan.FromSeconds(start + (samples.Length / (double)SampleRate)),
            Text = text,
            Tokens = [.. generated],
            Language = options.Language,
            AverageLogProbability = generated.Count > 0 ? logProbabilities / generated.Count : null,
        };
    }

    // The rows of consecutive steps as the multimodal graph's prompt: every row is a token whose embedding gets the step's audio row added.
    private MultimodalPrompt Rows(int[] tokens, int firstRow, int startPosition)
    {
        var index = new int[tokens.Length];
        var row = new int[tokens.Length];
        var positions = new int[tokens.Length * _positionAxes];
        for (int i = 0; i < tokens.Length; i++)
        {
            row[i] = firstRow + i;
            for (int a = 0; a < _positionAxes; a++)
            {
                positions[(i * _positionAxes) + a] = startPosition + i;
            }
        }

        return new MultimodalPrompt
        {
            Tokens = tokens,
            ImageIndex = index,
            ImageRow = row,
            Positions = positions,
            PositionAxes = _positionAxes,
            NextPosition = startPosition + tokens.Length,
        };
    }
}
