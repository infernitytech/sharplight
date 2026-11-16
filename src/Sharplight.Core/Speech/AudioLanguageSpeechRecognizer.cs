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
/// The <see cref="ISpeechRecognizer"/> of audio language models (Qwen3-ASR): the audio tower turns each window of up to
/// <see cref="AudioLanguageModelDescription.MaxWindowSeconds"/> seconds of log-mel features into embedding rows, which replace the audio
/// placeholder tokens of the model's transcription prompt (<see cref="AudioPromptSettings"/>) in the language model's multimodal graph, the
/// same path image embeddings take; the language model then generates the transcript (greedy unless the request samples). Without a language
/// the model names the language before the transcript marker, with one the language is forced by the assistant prefix. Each window becomes one
/// segment with the window's times. The graphs compile on the first request; one request runs at a time.
/// </summary>
internal sealed class AudioLanguageSpeechRecognizer : ISpeechRecognizer
{
    /// <summary>The most tokens generated per window.</summary>
    private const int MaxNewTokens = 1024;

    private readonly TaskModel _model;
    private readonly IAudioFeatureExtractor _features;
    private readonly ChunkedAudioLayout _layout;
    private readonly AudioPromptSettings _prompt;
    private readonly LlmModel _llm;
    private readonly ITokenizer _tokenizer;
    private readonly IServiceProvider? _services;
    private readonly int _maxFrames;
    private readonly int _minFrames;
    private readonly int _positionAxes;
    private readonly bool _wholeWindow;
    private readonly int _downsample;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private CompiledModel? _encoder;
    private InferenceSession? _session;
    private bool _disposed;

    /// <summary>Creates the recognizer; nothing is compiled until the first request.</summary>
    /// <param name="model">An audio language model (<see cref="AudioLanguageModelDescription"/>).</param>
    /// <param name="services">The container a voice activity detector is resolved from, or <see langword="null"/>.</param>
    public AudioLanguageSpeechRecognizer(TaskModel model, IServiceProvider? services = null)
    {
        _model = model;
        _services = services;
        _features = AudioFeatureExtractors.Create(model);
        var encoder = model.Description.Component(ModelComponentRoles.Encoder);
        _layout = new ChunkedAudioLayout(encoder.ConfigValue("chunkFrames", 100), encoder.ConfigValue("windowFrames", 800), encoder.ConfigValue("convCount", 3),
            encoder.ConfigValue("melBins", 128));
        _wholeWindow = encoder.ConfigValue("tower", "") == AudioTowerParameters.Whisper;
        _downsample = encoder.ConfigValue("downsample", 1);
        _maxFrames = encoder.ConfigValue("maxFrames", 3000) / _layout.ChunkFrames * _layout.ChunkFrames;
        _minFrames = encoder.ConfigValue("minFrames", 0);
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

    /// <summary>Whether <paramref name="model"/> is an audio language model this recognizer runs.</summary>
    /// <param name="model">A loaded speech-recognition model.</param>
    public static bool Handles(TaskModel model)
    {
        var description = model.Description;
        return description.TryGetComponent(ModelComponentRoles.Encoder, out var encoder)
            && encoder.ConfigValue("tower", "") is AudioTowerParameters.ChunkedConv2d or AudioTowerParameters.Whisper
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
        string? language = options?.Language ?? segments.Select(s => s.Language).FirstOrDefault(l => l is not null);
        return new Transcription(text, language, segments, audio.Duration);
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
            var features = await Task.Run(() => _features.Extract(samples), cancellationToken).ConfigureAwait(false);
            int content = Math.Max((samples.Length + _features.HopLength - 1) / _features.HopLength, Math.Max(_minFrames, 1));
            for (int seek = 0; seek < content; seek += _maxFrames)
            {
                int start = seek;
                int frames = Math.Min(_maxFrames, content - start);
                var segment = await Task.Run(() => RecognizeWindow(features, start, frames, options, cancellationToken), cancellationToken).ConfigureAwait(false);
                if (segment is not null)
                {
                    yield return segment;
                }
            }
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>The audio embedding rows of a window, [rows, hidden] row-major (compiles the encoder on first use).</summary>
    /// <param name="features">The extracted features.</param>
    /// <param name="first">The window's first frame.</param>
    /// <param name="frames">The window's valid frames.</param>
    public float[] Encode(AudioFeatures features, int first, int frames)
    {
        var encoder = _encoder ??= _model.CompileComponent(ModelComponentRoles.Encoder);
        if (_wholeWindow)
        {
            return EncodeWholeWindow(encoder, features, first);
        }

        int chunks = _layout.Chunks(frames);
        int rows = _layout.OutputRows(frames);
        var graph = encoder.Graph;
        Value Input(string name) => graph.Inputs.First(v => v.Input!.Name == name);
        encoder.SetInput<float>(Input(ChunkedAudioTowerGraph.FeaturesInput), _layout.FeatureMap(features, first, frames));
        for (int level = 1; level < _layout.ConvCount; level++)
        {
            encoder.SetInput<float>(Input(ChunkedAudioTowerGraph.MaskInput(level)), _layout.RowMask(level, chunks));
        }

        encoder.SetInput<int>(Input(ChunkedAudioTowerGraph.PositionsInput), _layout.Positions(chunks));
        encoder.SetInput<int>(Input(ChunkedAudioTowerGraph.RowsInput), _layout.ValidRowIndices(frames));
        encoder.SetInput<int>(Input(ChunkedAudioTowerGraph.SpansInput), _layout.WindowSpans(frames));
        var run = new RunParameters();
        run.Symbols[WellKnown.AudioFrameSymbol] = (long)chunks * _layout.PaddedFrames;
        run.Symbols[WellKnown.EncoderSymbol] = rows;
        encoder.Run(run);
        int width = (int)graph.Outputs[ChunkedAudioTowerGraph.EmbeddingsOutput].Shape[1].Fixed;
        var embeddings = new float[(long)rows * width];
        encoder.GetOutput<float>(ChunkedAudioTowerGraph.EmbeddingsOutput, embeddings);
        return embeddings;
    }

    /// <summary>The number of audio embedding rows of a window of <paramref name="frames"/> valid frames.</summary>
    /// <param name="frames">The window's valid frames.</param>
    public int AudioRows(int frames) => _wholeWindow ? _maxFrames / 2 / _downsample : _layout.OutputRows(frames);

    // A Whisper-style tower: always the whole fixed window (the frames after the recording are the extractor's trailing silence).
    private float[] EncodeWholeWindow(CompiledModel encoder, AudioFeatures features, int first)
    {
        var window = new float[(long)_maxFrames * _layout.MelBins];
        var values = features.Values;
        for (int t = 0; t < _maxFrames && first + t < features.Frames; t++)
        {
            for (int m = 0; m < _layout.MelBins; m++)
            {
                window[((long)t * _layout.MelBins) + m] = values[((long)m * features.Frames) + first + t];
            }
        }

        var graph = encoder.Graph;
        encoder.SetInput<float>(graph.Inputs.First(v => v.Input!.Name == ChunkedAudioTowerGraph.FeaturesInput), window);
        encoder.Run(new RunParameters());
        var output = graph.Outputs[ChunkedAudioTowerGraph.EmbeddingsOutput];
        var embeddings = new float[output.Shape[0].Fixed * output.Shape[1].Fixed];
        encoder.GetOutput<float>(ChunkedAudioTowerGraph.EmbeddingsOutput, embeddings);
        return embeddings;
    }

    /// <summary>The prompt token ids of a window with <paramref name="audioRows"/> audio rows (placeholders where the rows go).</summary>
    /// <param name="audioRows">The number of audio embedding rows.</param>
    /// <param name="options">The request (prompt text, language).</param>
    public int[] PromptTokens(int audioRows, SpeechRecognitionOptions options)
    {
        string template = _prompt.Template;
        if (!template.Contains("{audio}", StringComparison.Ordinal))
        {
            throw new InvalidDataException("The audio prompt template needs the {audio} placeholder.");
        }

        if (!template.Contains("{language}", StringComparison.Ordinal))
        {
            template += "{language}";
        }

        var tokens = new List<int>();
        int at = 0;
        while (at < template.Length)
        {
            int open = template.IndexOf('{', at);
            int close = open < 0 ? -1 : template.IndexOf('}', open);
            string? placeholder = close > open ? template[(open + 1)..close] : null;
            if (placeholder is not ("context" or "audio" or "language"))
            {
                int end = open < 0 ? template.Length : open + 1;
                int next = template.IndexOf('{', end);
                end = next < 0 ? template.Length : next;
                tokens.AddRange(Encode(template[at..end], special: true));
                at = end;
                continue;
            }

            if (open > at)
            {
                tokens.AddRange(Encode(template[at..open], special: true));
            }

            switch (placeholder)
            {
                case "context" when !string.IsNullOrEmpty(options.Prompt):
                    tokens.AddRange(Encode(options.Prompt, special: false));
                    break;
                case "audio":
                    tokens.AddRange(Enumerable.Repeat(_prompt.AudioToken, audioRows));
                    break;
                case "language" when options.Language is { Length: > 0 } language && _prompt.LanguagePrefix.Length > 0:
                    string prefix = _prompt.LanguagePrefix
                        .Replace("{language}", SpeechLanguages.EnglishName(language), StringComparison.Ordinal)
                        .Replace("{code}", SpeechLanguages.Code(language), StringComparison.Ordinal);
                    tokens.AddRange(Encode(prefix, special: true));
                    break;
                case "language":
                    tokens.AddRange(Encode(_prompt.AutoLanguage, special: true));
                    break;
            }

            at = close + 1;
        }

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

    private int[] Encode(string text, bool special) => _tokenizer.Encode(text, addSpecial: false, parseSpecial: special);

    private TranscriptionSegment? RecognizeWindow(AudioFeatures features, int first, int frames, SpeechRecognitionOptions options, CancellationToken cancellationToken)
    {
        var embeddings = Encode(features, first, frames);
        int rows = AudioRows(frames);
        var tokens = PromptTokens(rows, options);
        var session = _session!;
        session.Reset();
        var prompt = Assemble(tokens, session.RopePosition);
        var sampler = new Sampler(options.Sampling);
        foreach (int t in tokens)
        {
            sampler.Accept(t);
        }

        int budget = Math.Max(1, Math.Min(MaxNewTokens, session.ContextLength - tokens.Length));
        var generated = new List<int>();
        double logProbabilities = 0;
        var logits = session.Process(prompt, [embeddings]);
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
                logits = session.Process([id]);
            }
        }

        var (language, transcript) = Split(generated, options.Language);
        string text = _tokenizer.Decode([.. transcript], skipSpecial: true).Trim();
        if (text.Length == 0)
        {
            return null;
        }

        double frameSeconds = (double)_features.HopLength / SampleRate;
        return new TranscriptionSegment
        {
            Start = TimeSpan.FromSeconds(first * frameSeconds),
            End = TimeSpan.FromSeconds((first + frames) * frameSeconds),
            Text = text,
            Tokens = transcript,
            Language = language,
            AverageLogProbability = generated.Count > 0 ? logProbabilities / generated.Count : null,
        };
    }

    // The prompt as the multimodal graph's rows: text tokens, and the audio rows (one per embedding row) at the placeholders.
    private MultimodalPrompt Assemble(int[] tokens, int startPosition)
    {
        var imageIndex = new int[tokens.Length];
        var imageRow = new int[tokens.Length];
        var positions = new int[tokens.Length * _positionAxes];
        int row = 0;
        for (int i = 0; i < tokens.Length; i++)
        {
            bool audio = tokens[i] == _prompt.AudioToken;
            imageIndex[i] = audio ? 0 : -1;
            imageRow[i] = audio ? row++ : 0;
            for (int a = 0; a < _positionAxes; a++)
            {
                positions[(i * _positionAxes) + a] = startPosition + i;
            }
        }

        return new MultimodalPrompt
        {
            Tokens = tokens,
            ImageIndex = imageIndex,
            ImageRow = imageRow,
            Positions = positions,
            PositionAxes = _positionAxes,
            NextPosition = startPosition + tokens.Length,
        };
    }

    // Without a forced language the output is "<language preamble><marker><transcript>": the preamble names the language.
    private (string? Language, int[] Transcript) Split(List<int> generated, string? forced)
    {
        if (forced is { Length: > 0 } || _prompt.TranscriptMarker.Length == 0)
        {
            return (forced, [.. generated]);
        }

        int at = generated.FindIndex(id => _tokenizer.TokenString(id) == _prompt.TranscriptMarker);
        if (at < 0)
        {
            return (null, [.. generated]);
        }

        string preamble = _tokenizer.Decode([.. generated.Take(at)], skipSpecial: true).Trim();
        int space = preamble.LastIndexOf(' ');
        string name = space >= 0 ? preamble[(space + 1)..] : preamble;
        string? language = name.Length == 0 || name.Equals("None", StringComparison.OrdinalIgnoreCase) ? null : SpeechLanguages.Code(name);
        return (language, [.. generated.Skip(at + 1)]);
    }
}
