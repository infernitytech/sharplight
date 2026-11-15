using System.Runtime.CompilerServices;
using Sharplight.Audio;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Tasks;
using Sharplight.Tokenizers;

namespace Sharplight.Speech;

/// <summary>
/// The <see cref="ISpeechRecognizer"/> of encoder-decoder speech models whose decoder is prompted with control tokens (Whisper and its
/// derivatives): host-side features over the whole recording, then window by window (the encoder's fixed window, e.g. 30 s) the encoder,
/// language detection when no language is given, and the decoding loop with the timestamp rules. Long recordings advance by the last
/// complete timestamp segment of each window, as the reference implementations do. The graphs compile on the first request. One
/// request runs at a time; concurrent requests wait.
/// </summary>
internal sealed class EncoderDecoderSpeechRecognizer : ISpeechRecognizer
{
    /// <summary>A window counts as silence when its no-speech probability is above this and its text is unlikely (see below).</summary>
    private const double NoSpeechThreshold = 0.6;

    /// <summary>The mean log probability below which the text of a probable no-speech window is dropped.</summary>
    private const double LogProbabilityThreshold = -1.0;

    private readonly TaskModel _model;
    private readonly IAudioFeatureExtractor _features;
    private readonly SpeechTokenProtocol _protocol;
    private readonly ITokenizer _tokenizer;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private SpeechEncoderDecoder? _network;
    private bool _disposed;

    /// <summary>Creates the recognizer; nothing is compiled until the first request.</summary>
    /// <param name="model">A speech-recognition model with encoder, decoder, tokenizer and feature-extractor components.</param>
    public EncoderDecoderSpeechRecognizer(TaskModel model)
    {
        _model = model;
        _features = AudioFeatureExtractors.Create(model);
        _protocol = SpeechTokenProtocol.From(model.Metadata(ModelComponentRoles.Decoder));
        _tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    /// <inheritdoc/>
    public TaskClientMetadata Metadata { get; }

    /// <inheritdoc/>
    public int SampleRate => _features.SampleRate;

    /// <inheritdoc/>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
        {
            return null;
        }

        return serviceType.IsInstanceOfType(_model) ? _model : serviceType.IsInstanceOfType(this) ? this : null;
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

        string text = string.Concat(segments.Select(s => s.Text)).Trim();
        string? language = options?.Language ?? segments.FirstOrDefault()?.Language;
        return new Transcription(text, language, segments, audio.Duration);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<TranscriptionSegment> TranscribeStreamingAsync(AudioBuffer audio, SpeechRecognitionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ObjectDisposedException.ThrowIf(_disposed, this);
        options ??= SpeechRecognitionOptions.Default;
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var network = _network ??= await Task.Run(() => new SpeechEncoderDecoder(_model), cancellationToken).ConfigureAwait(false);
            if (network.FeatureChannels != _features.Channels)
            {
                throw new InvalidDataException($"The encoder takes {network.FeatureChannels} feature channels, the feature extractor makes {_features.Channels}.");
            }

            var samples = audio.ToModelInput(SampleRate).Samples;
            var features = await Task.Run(() => _features.Extract(samples), cancellationToken).ConfigureAwait(false);
            var state = new RecognitionState(options, samples.Length / _features.HopLength, PromptHistory(options));
            while (state.Seek < state.ContentFrames)
            {
                var segments = await Task.Run(() => RecognizeWindow(network, features, state, cancellationToken), cancellationToken).ConfigureAwait(false);
                foreach (var segment in segments)
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

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _network?.Dispose();
        _gate.Dispose();
    }

    private List<int> PromptHistory(SpeechRecognitionOptions options)
        => string.IsNullOrWhiteSpace(options.Prompt) ? [] : [.. _tokenizer.Encode(" " + options.Prompt.Trim(), addSpecial: false, parseSpecial: false)];

    /// <summary>Decodes the window at <see cref="RecognitionState.Seek"/>, advances the seek position and returns the window's segments.</summary>
    private List<TranscriptionSegment> RecognizeWindow(SpeechEncoderDecoder network, AudioFeatures features, RecognitionState state, CancellationToken cancellationToken)
    {
        int windowFrames = network.FeatureFrames;
        int segmentFrames = Math.Min(windowFrames, state.ContentFrames - state.Seek);
        var window = new float[network.FeatureChannels * windowFrames];
        features.CopyWindow(state.Seek, windowFrames, window);
        network.Encode(window);
        state.Language ??= DetectLanguage(network, state.Options);

        bool timestamps = state.Options.Timestamps != TimestampGranularity.None;
        var decoding = DecodeWindow(network, state, timestamps, cancellationToken);
        double frameSeconds = (double)_features.HopLength / SampleRate;
        double offset = state.Seek * frameSeconds;
        string? language = state.Language is int languageToken ? _protocol.LanguageCode(languageToken) : null;
        if (decoding.NoSpeechProbability > NoSpeechThreshold && decoding.AverageLogProbability < LogProbabilityThreshold)
        {
            state.Seek += segmentFrames;
            return [];
        }

        var segments = new List<TranscriptionSegment>();
        var tokens = decoding.Tokens;
        int framesPerRow = windowFrames / network.SourceRows;
        int begin = _protocol.TimestampBeginId;
        bool singleTimestampEnding = tokens.Count >= 2 && tokens[^2] < begin && tokens[^1] >= begin;
        var boundaries = new List<int>();
        for (int i = 1; i < tokens.Count; i++)
        {
            if (tokens[i] >= begin && tokens[i - 1] >= begin)
            {
                boundaries.Add(i);
            }
        }

        if (timestamps && boundaries.Count > 0)
        {
            if (singleTimestampEnding)
            {
                boundaries.Add(tokens.Count);
            }

            int last = 0;
            foreach (int boundary in boundaries)
            {
                var slice = tokens.Skip(last).Take(boundary - last).ToArray();
                AddSegment(segments, slice, offset + ((slice[0] - begin) * _protocol.TimestampStep), offset + ((slice[^1] - begin) * _protocol.TimestampStep),
                    decoding, language);
                last = boundary;
            }

            // Continue after the last complete segment; a window that ended in silence (a lone timestamp) is consumed whole.
            int advance = singleTimestampEnding ? segmentFrames : (tokens[last - 1] - begin) * framesPerRow;
            state.Seek += advance > 0 ? advance : segmentFrames;
        }
        else
        {
            double duration = segmentFrames * frameSeconds;
            int lastTimestamp = tokens.LastOrDefault(t => t >= begin, -1);
            if (timestamps && lastTimestamp > begin)
            {
                duration = (lastTimestamp - begin) * _protocol.TimestampStep;
            }

            AddSegment(segments, tokens, offset, offset + duration, decoding, language);
            state.Seek += segmentFrames;
        }

        if (state.Options.ConditionOnPreviousText)
        {
            state.History.AddRange(tokens);
        }

        return segments;
    }

    private void AddSegment(List<TranscriptionSegment> segments, IReadOnlyList<int> tokens, double start, double end, WindowDecoding decoding, string? language)
    {
        int[] text = [.. tokens.Where(t => t < _protocol.EndId)];
        // Timestamp and other control tokens are above the end token; only the text is decoded.
        string decoded = _tokenizer.Decode(text, skipSpecial: true);
        if (string.IsNullOrWhiteSpace(decoded) || end <= start)
        {
            return;
        }

        segments.Add(new TranscriptionSegment
        {
            Start = TimeSpan.FromSeconds(start),
            End = TimeSpan.FromSeconds(end),
            Text = decoded,
            Tokens = tokens,
            Language = language,
            AverageLogProbability = decoding.AverageLogProbability,
            NoSpeechProbability = decoding.NoSpeechProbability,
        });
    }

    /// <summary>The requested language's token, the detected one on a multilingual model, or null for single-language models.</summary>
    private int? DetectLanguage(SpeechEncoderDecoder network, SpeechRecognitionOptions options)
    {
        if (!_protocol.IsMultilingual)
        {
            return null;
        }

        if (options.Language is { } code)
        {
            return _protocol.LanguageToken(code);
        }

        var logits = network.Decode([_protocol.StartId]);
        network.Rewind(0);
        int best = _protocol.LanguageIds[0];
        foreach (int id in _protocol.LanguageIds)
        {
            if (logits[id] > logits[best])
            {
                best = id;
            }
        }

        return best;
    }

    /// <summary>Runs the decoder over the window's prompt and generates until the end token or the window's text limit.</summary>
    private WindowDecoding DecodeWindow(SpeechEncoderDecoder network, RecognitionState state, bool timestamps, CancellationToken cancellationToken)
    {
        var start = _protocol.StartSequence(state.Language, state.Options.Translate, timestamps);
        var prompt = new List<int>();
        if (state.History.Count > 0 && _protocol.PreviousId is int previous)
        {
            int keep = Math.Min(state.History.Count, (network.ContextLength / 2) - 1);
            prompt.Add(previous);
            prompt.AddRange(state.History.Skip(state.History.Count - keep));
        }

        // The no-speech probability is read at the start token, so the prompt is fed up to it first.
        prompt.Add(start[0]);
        var logits = network.Decode(prompt.ToArray());
        double? noSpeech = _protocol.NoSpeechId is int noSpeechId ? Softmax(logits, noSpeechId) : null;
        if (start.Count > 1)
        {
            logits = network.Decode(start.Skip(1).ToArray());
        }

        var filter = new SpeechLogitFilter(_protocol, timestamps);
        var sampler = new Sampler(state.Options.Sampling);
        var generated = new List<int>();
        double sumLogProbability = 0;
        int limit = Math.Min(_protocol.MaxTextTokens, network.ContextLength - network.Position);
        for (int step = 0; step < limit; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            filter.Apply(logits, generated);
            int token = sampler.Sample(logits);
            sumLogProbability += LogSoftmax(logits, token);
            if (token == _protocol.EndId)
            {
                break;
            }

            generated.Add(token);
            if (step + 1 < limit)
            {
                logits = network.Decode([token]);
            }
        }

        return new WindowDecoding(generated, sumLogProbability / (generated.Count + 1), noSpeech);
    }

    private static double Softmax(ReadOnlySpan<float> logits, int index) => Math.Exp(LogSoftmax(logits, index));

    private static double LogSoftmax(ReadOnlySpan<float> logits, int index)
    {
        float max = float.NegativeInfinity;
        foreach (float value in logits)
        {
            max = MathF.Max(max, value);
        }

        double sum = 0;
        foreach (float value in logits)
        {
            sum += Math.Exp(value - max);
        }

        return logits[index] - max - Math.Log(sum);
    }
}
