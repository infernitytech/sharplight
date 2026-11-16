using System.Runtime.CompilerServices;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Tasks;
using Sharplight.Tokenizers;

namespace Sharplight.Speech;

/// <summary>
/// The <see cref="ISpeechRecognizer"/> of speech models with a frame-aligned decoder (CTC heads and transducers such as Parakeet): host-side
/// features over the whole recording, then window by window (up to the encoder's bound, e.g. 30 s, without padding) the encoder and the
/// decoding on the host — CTC greedy or prefix beam search (<see cref="SpeechRecognitionOptions.BeamSize"/>), greedy RNN-T or TDT. Each
/// window becomes one segment whose times are those of its first and last token (the window bounds without timestamps); with
/// <see cref="TimestampGranularity.Word"/> the words carry the times of their tokens. The graphs compile on the first request. One request
/// runs at a time; concurrent requests wait.
/// </summary>
internal sealed class FrameSpeechRecognizer : ISpeechRecognizer
{
    /// <summary>The word-start marker of SentencePiece vocabularies.</summary>
    private const char WordStart = '▁';

    private readonly TaskModel _model;
    private readonly IAudioFeatureExtractor _features;
    private readonly FrameDecodingSettings _settings;
    private readonly ITokenizer _tokenizer;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private SpeechEncoder? _encoder;
    private TransducerNetwork? _transducer;
    private bool _disposed;

    /// <summary>Creates the recognizer; nothing is compiled until the first request.</summary>
    /// <param name="model">A speech-recognition model with encoder, decoder, tokenizer and feature-extractor components.</param>
    public FrameSpeechRecognizer(TaskModel model)
    {
        _model = model;
        _features = AudioFeatureExtractors.Create(model);
        _settings = FrameDecodingSettings.From(model.Description.Component(ModelComponentRoles.Decoder));
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
        if (options.UseVoiceActivityDetection)
        {
            await foreach (var segment in VoiceActivityPresegmentation.TranscribeAsync(this, audio, options, cancellationToken).ConfigureAwait(false))
            {
                yield return segment;
            }

            yield break;
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(options.BeamSize, 1, nameof(options));
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var encoder = _encoder ??= await Task.Run(() => new SpeechEncoder(_model, _settings.IsCtc ? WellKnown.LogitsOutput : WellKnown.EncoderOutput), cancellationToken)
                .ConfigureAwait(false);
            if (!_settings.IsCtc)
            {
                _transducer ??= await Task.Run(() => new TransducerNetwork(_model), cancellationToken).ConfigureAwait(false);
            }

            if (encoder.Channels != _features.Channels)
            {
                throw new InvalidDataException($"The encoder takes {encoder.Channels} feature channels, the feature extractor makes {_features.Channels}.");
            }

            var samples = audio.ToModelInput(SampleRate).Samples;
            var features = await Task.Run(() => _features.Extract(samples), cancellationToken).ConfigureAwait(false);
            int content = Math.Max(Math.Min(features.Frames, samples.Length / _features.HopLength), 1);
            for (int seek = 0; seek < content; seek += encoder.MaxFrames)
            {
                int start = seek;
                var segment = await Task.Run(() => RecognizeWindow(encoder, features, start, Math.Min(encoder.MaxFrames, content - start), options, cancellationToken),
                    cancellationToken).ConfigureAwait(false);
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

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _transducer?.Dispose();
        _encoder?.Dispose();
        _gate.Dispose();
    }

    private TranscriptionSegment? RecognizeWindow(SpeechEncoder encoder, AudioFeatures features, int seek, int frames, SpeechRecognitionOptions options,
        CancellationToken cancellationToken)
    {
        int encoded = Math.Max(frames, encoder.MinFrames);
        var window = new float[encoder.Channels * encoded];
        features.CopyWindow(seek, encoded, window);
        encoder.Encode(window, encoded);
        var tokens = _settings.IsCtc
            ? options.BeamSize > 1
                ? CtcDecoding.BeamSearch(encoder.Output, encoder.Width, _settings.Blank, options.BeamSize)
                : CtcDecoding.Greedy(encoder.Output, encoder.Width, _settings.Blank)
            : TransducerDecoding.Greedy(_transducer!, _settings, encoder.Output, cancellationToken);
        int[] ids = [.. tokens.Select(t => t.Token)];
        string text = _tokenizer.Decode(ids, skipSpecial: true);
        if (string.IsNullOrWhiteSpace(text))
        {
            return null;
        }

        double frameSeconds = (double)_features.HopLength / SampleRate;
        double rowSeconds = frameSeconds * _settings.RowFrames;
        double offset = seek * frameSeconds;
        bool timestamps = options.Timestamps != TimestampGranularity.None;
        double start = timestamps ? offset + (tokens[0].Row * rowSeconds) : offset;
        double end = timestamps ? offset + ((tokens[^1].Row + tokens[^1].Rows) * rowSeconds) : offset + (frames * frameSeconds);
        return new TranscriptionSegment
        {
            Start = TimeSpan.FromSeconds(start),
            End = TimeSpan.FromSeconds(Math.Min(end, offset + (frames * frameSeconds))),
            Text = text,
            Tokens = ids,
            Language = options.Language,
            Words = options.Timestamps == TimestampGranularity.Word ? Words(tokens, offset, rowSeconds) : [],
            AverageLogProbability = tokens.Average(t => (double)t.LogProbability),
        };
    }

    // Words start at tokens whose piece begins with the SentencePiece marker; a word spans its tokens' rows.
    private List<WordTiming> Words(List<AlignedToken> tokens, double offset, double rowSeconds)
    {
        var words = new List<WordTiming>();
        int first = 0;
        for (int i = 1; i <= tokens.Count; i++)
        {
            if (i < tokens.Count && !(_tokenizer.TokenString(tokens[i].Token) is { Length: > 0 } piece && piece[0] == WordStart))
            {
                continue;
            }

            var word = tokens.GetRange(first, i - first);
            string text = _tokenizer.Decode(word.Select(t => t.Token).ToArray(), skipSpecial: true);
            if (!string.IsNullOrWhiteSpace(text))
            {
                double probability = Math.Exp(word.Sum(t => (double)t.LogProbability));
                words.Add(new WordTiming(first == 0 ? text : " " + text.TrimStart(), TimeSpan.FromSeconds(offset + (word[0].Row * rowSeconds)),
                    TimeSpan.FromSeconds(offset + ((word[^1].Row + word[^1].Rows) * rowSeconds)), (float)probability));
            }

            first = i;
        }

        return words;
    }
}
