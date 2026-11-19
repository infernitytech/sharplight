using System.Globalization;
using Sharplight.Audio;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Architectures;

/// <summary>
/// The <see cref="ISpeechSynthesizer"/> and <see cref="IVoiceCloner"/> of a Dia model (<see cref="DiaAdapter"/>), as the reference's <c>generate</c>: the text's bytes go through the
/// encoder; the decoder runs over the delayed sequence of the start row (and of an audio prompt's codes) and predicts all nine codebooks of the next row at once. Classifier-free
/// guidance runs a second stream on a text of zeros. The tokens are chosen by <see cref="DiaTokenSampler"/>; the reverted frames go through the DAC decoder to 44.1 kHz audio.
/// A voice is an audio prompt (its codes) with its transcript, which goes in front of the text.
/// </summary>
internal sealed class DiaSynthesizer : ISpeechSynthesizer, IVoiceCloner
{
    /// <summary>The architecture family of the voices this model creates.</summary>
    public const string VoiceFamily = DiaAdapter.Family;

    private const int StreamFrames = 86;
    private const int StreamContext = 16;
    private const int StreamLookahead = 16;

    private readonly TaskModel _model;
    private readonly int _channels;
    private readonly int _vocabulary;
    private readonly int _bos;
    private readonly int _eos;
    private readonly int _pad;
    private readonly int _maxText;
    private readonly int _width;
    private readonly int[] _delays;
    private readonly int _frameSamples;
    private readonly int _codecSize;
    private int _maxLength;
    private readonly float _guidance;
    private readonly SamplingOptions _defaults;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private CompiledModel? _text;
    private DiaStream? _conditional;
    private DiaStream? _unconditional;
    private CompiledModel? _encoder;
    private CompiledModel? _decoder;
    private bool _disposed;

    /// <summary>Creates the synthesizer; nothing is compiled until the first request.</summary>
    /// <param name="model">A loaded Dia model.</param>
    public DiaSynthesizer(TaskModel model)
    {
        _model = model;
        var decoder = model.Description.Component(DiaGraphs.DecoderRole);
        _channels = decoder.ConfigValue("channels", 9);
        _vocabulary = decoder.ConfigValue("vocabulary", 1028);
        _bos = decoder.ConfigValue("bos", 1026);
        _eos = decoder.ConfigValue("eos", 1024);
        _pad = decoder.ConfigValue("pad", 1025);
        _maxText = decoder.ConfigValue("maxText", 1024);
        _width = model.Description.Component(DiaGraphs.TextRole).ConfigValue("width", 1024);
        _delays = [.. decoder.ConfigValue("delays", "0,8,9,10,11,12,13,14,15").Split(',').Select(d => int.Parse(d, CultureInfo.InvariantCulture))];
        var codec = model.Description.Component(DiaGraphs.CodecDecoderRole);
        SampleRate = codec.ConfigValue(AudioCodecGraphs.SampleRateKey, 44100);
        _frameSamples = codec.ConfigValue(AudioCodecGraphs.SamplesPerFrameKey, 512);
        _codecSize = codec.ConfigValue(AudioCodecGraphs.CodebookSizeKey, 1024);
        _maxLength = (int)decoder.SymbolBounds.GetValueOrDefault(WellKnown.ContextSymbol, 3088) - 1;
        _guidance = (float)decoder.ConfigValue("sampling.guidance_scale", 3.0);
        _defaults = new SamplingOptions
        {
            Temperature = (float)decoder.ConfigValue("sampling.temperature", 1.3), TopK = (int)decoder.ConfigValue("sampling.top_k", 35.0),
            TopP = (float)decoder.ConfigValue("sampling.top_p", 0.95), MinP = 0f, RepetitionPenalty = 1f,
        };
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    /// <inheritdoc/>
    public TaskClientMetadata Metadata { get; }

    /// <summary>The most rows of a request's decoder sequence (the model's context by default); tests lower it to bound a run.</summary>
    internal int MaxLength
    {
        get => _maxLength;
        set => _maxLength = Math.Clamp(value, 32, _maxLength);
    }

    /// <inheritdoc/>
    public int SampleRate { get; }

    /// <inheritdoc/>
    public IReadOnlyList<string> Voices => [];

    /// <inheritdoc/>
    public string VoiceArchitecture => VoiceFamily;

    /// <summary>Whether <paramref name="model"/> is a Dia model (<see cref="DiaAdapter"/>).</summary>
    /// <param name="model">A loaded speech-synthesis model.</param>
    public static bool Handles(TaskModel model)
        => model.Description.TryGetComponent(DiaGraphs.DecoderRole, out var decoder) && decoder.ConfigValue("family", "") == DiaAdapter.Family;

    /// <inheritdoc/>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
        {
            return null;
        }

        return serviceType.IsInstanceOfType(this) ? this : serviceType.IsInstanceOfType(_model) ? _model : null;
    }

    /// <inheritdoc/>
    public async Task<SpeechSynthesisResult> SynthesizeAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        ObjectDisposedException.ThrowIf(_disposed, this);
        var audio = await BackgroundCompute.RunAsync(_gate, () =>
        {
            var (prompt, full) = Prepare(text, options, cancellationToken);
            var result = Run(full, Settings(options), prompt, null, null, null, cancellationToken);
            return Decode(DiaDelay.Revert(result.Rows, _delays, result.Start, result.End), cancellationToken);
        }, cancellationToken).ConfigureAwait(false);
        return new SpeechSynthesisResult(audio, options?.Voice ?? "");
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Returns at once: generation and decoding run on a background thread, one request at a time per client. A frame is complete once the last codebook's delay has passed;
    /// every <see cref="StreamFrames"/> complete frames are decoded with context before and lookahead after them (the decoder's convolutions are not causal).
    /// </remarks>
    public IAsyncEnumerable<AudioBuffer> SynthesizeStreamingAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        ObjectDisposedException.ThrowIf(_disposed, this);
        return BackgroundCompute.StreamBlocking<AudioBuffer>((emit, token) =>
        {
            _gate.Wait(token);
            try
            {
                var (prompt, full) = Prepare(text, options, token);
                int emitted = 0;
                var result = Run(full, Settings(options), prompt, null, null, (rows, start, complete) =>
                {
                    while (complete - emitted >= StreamFrames + StreamLookahead)
                    {
                        emit(DecodeSlice(DiaDelay.Revert(rows, _delays, start, start + emitted + StreamFrames + StreamLookahead), emitted, emitted + StreamFrames, StreamContext));
                        emitted += StreamFrames;
                    }
                }, token);
                int total = result.End - result.Start;
                if (total > emitted)
                {
                    emit(DecodeSlice(DiaDelay.Revert(result.Rows, _delays, result.Start, result.End), emitted, total, StreamContext));
                }
            }
            finally
            {
                _gate.Release();
            }
        }, cancellationToken);
    }

    /// <inheritdoc/>
    /// <remarks>The voice is the DAC codes of the recording (at 44.1 kHz) with its transcript, which must tell the speakers as the model's text does (<c>[S1]</c>, <c>[S2]</c>).</remarks>
    public async Task<VoiceProfile> CreateVoiceAsync(string name, AudioBuffer recording, string? referenceText = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(recording);
        ArgumentException.ThrowIfNullOrWhiteSpace(referenceText, nameof(referenceText));
        ObjectDisposedException.ThrowIf(_disposed, this);
        var frames = await BackgroundCompute.RunAsync(_gate, () => EncodePrompt(recording, cancellationToken), cancellationToken).ConfigureAwait(false);
        return new VoiceProfile
        {
            Name = name,
            Architecture = VoiceArchitecture,
            ReferenceText = DiaText.EnsureSpeaker(referenceText),
            SampleRate = SampleRate,
            Tensors = new Dictionary<string, VoiceTensor>
            {
                ["codes"] = VoiceTensor.From([.. frames.SelectMany(f => f)], [frames.Count, _channels]),
            },
        };
    }

    /// <summary>The text tokens of a request's text (the bytes, the speaker tags as 1 and 2).</summary>
    /// <param name="text">The text.</param>
    public int[] TokensFor(string text) => DiaText.Encode(text);

    /// <summary>
    /// Runs the model over a text: the encoder, then the decoder step by step until every codebook ended. With <paramref name="forced"/> the decoder is fed these rows
    /// (teacher forcing) instead of the chosen ones, and <paramref name="maxLength"/> bounds the sequence.
    /// </summary>
    /// <param name="text">The full text (a voice's transcript included).</param>
    /// <param name="settings">The generation settings.</param>
    /// <param name="prompt">The audio prompt's frames, or <see langword="null"/>.</param>
    /// <param name="forced">The sequence to feed, or <see langword="null"/>.</param>
    /// <param name="observe">Receives the conditional and unconditional logits of every step, or <see langword="null"/>.</param>
    /// <param name="cancellationToken">Stops generation between steps.</param>
    /// <param name="maxLength">Overrides the settings' maximum sequence length.</param>
    internal DiaResult Generate(string text, DiaGenerationSettings settings, IReadOnlyList<int[]>? prompt, IReadOnlyList<int[]>? forced, ObserveLogits? observe,
        CancellationToken cancellationToken, int? maxLength = null)
        => Run(text, maxLength is { } max ? settings with { MaxLength = max } : settings, prompt, forced, observe, null, cancellationToken);

    /// <summary>The generation settings of a request.</summary>
    /// <param name="options">The request's options.</param>
    internal DiaGenerationSettings Settings(SpeechSynthesisOptions? options)
    {
        var sampling = options.SamplingOr(_defaults);
        return new DiaGenerationSettings(options?.GuidanceScale ?? _guidance, sampling.Temperature, sampling.TopK, sampling.TopP, sampling.TopK, sampling.Seed, _maxLength);
    }

    /// <summary>Decodes frames of codes into audio (chunks with context before them).</summary>
    /// <param name="frames">The frames.</param>
    /// <param name="cancellationToken">Stops between chunks.</param>
    internal AudioBuffer Decode(IReadOnlyList<int[]> frames, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(frames);
        int chunk = DiaAdapter.CodecDecodeFrames - DiaAdapter.CodecContextFrames;
        var samples = new List<float>(frames.Count * _frameSamples);
        for (int start = 0; start < frames.Count; start += chunk)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int end = Math.Min(frames.Count, start + chunk);
            samples.AddRange(DecodeSlice(frames, start, end, Math.Min(DiaAdapter.CodecContextFrames, start)).Samples);
        }

        return new AudioBuffer([.. samples], SampleRate);
    }

    /// <summary>Encodes a recording into frames of codes (the codec's encoder at 44.1 kHz; the recording is padded to whole frames).</summary>
    /// <param name="recording">The recording.</param>
    /// <param name="cancellationToken">Stops before the encoder runs.</param>
    internal List<int[]> EncodePrompt(AudioBuffer recording, CancellationToken cancellationToken = default)
    {
        var mono = recording.ToModelInput(SampleRate).Samples;
        int frames = (mono.Length + _frameSamples - 1) / _frameSamples;
        if (frames < 1)
        {
            throw new ArgumentException("The recording is empty.", nameof(recording));
        }

        if (frames > DiaAdapter.PromptFrames)
        {
            throw new ArgumentException($"The recording is {frames * _frameSamples / (double)SampleRate:0.#} s long; an audio prompt can be {DiaAdapter.PromptFrames * _frameSamples / (double)SampleRate:0.#} s at most.", nameof(recording));
        }

        cancellationToken.ThrowIfCancellationRequested();
        var audio = new float[frames * _frameSamples];
        mono.CopyTo(audio, 0);
        var encoder = _encoder ??= _model.CompileComponent(DiaGraphs.CodecEncoderRole, AudioCodecGraphs.EncodeGraph);
        encoder.SetInput<float>(encoder.Graph.Inputs.First(v => v.Input!.Name == AudioCodecGraphs.AudioInput), audio);
        var parameters = new RunParameters();
        parameters.Symbols[WellKnown.TokenSymbol] = audio.Length;
        encoder.Run(parameters);
        var result = Enumerable.Range(0, frames).Select(_ => new int[_channels]).ToList();
        var part = new int[frames];
        for (int c = 0; c < _channels; c++)
        {
            encoder.GetOutput<int>(AudioCodecGraphs.CodesName(c), part);
            for (int f = 0; f < frames; f++)
            {
                result[f][c] = part[f];
            }
        }

        return result;
    }

    /// <summary>The receiver of one step's logits.</summary>
    /// <param name="conditional">The conditional logits [channels · vocabulary].</param>
    /// <param name="unconditional">The unconditional logits (empty without guidance).</param>
    internal delegate void ObserveLogits(ReadOnlySpan<float> conditional, ReadOnlySpan<float> unconditional);

    private delegate void FramesReady(IReadOnlyList<int[]> rows, int start, int complete);

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _text?.Dispose();
        _conditional?.Dispose();
        _unconditional?.Dispose();
        _encoder?.Dispose();
        _decoder?.Dispose();
        _gate.Dispose();
    }

    // The audio prompt's frames and the text to speak (the voice's transcript in front).
    private (List<int[]>? Prompt, string Text) Prepare(string text, SpeechSynthesisOptions? options, CancellationToken cancellationToken)
    {
        var profile = options?.VoiceProfile;
        if (profile is not null)
        {
            profile.RequireArchitecture(VoiceArchitecture);
            var codes = profile.Tensor("codes").ToInts();
            var frames = Enumerable.Range(0, codes.Length / _channels).Select(f => codes.AsSpan(f * _channels, _channels).ToArray()).ToList();
            return (frames, Join(profile.ReferenceText ?? "", text));
        }

        if (options?.ReferenceAudio is { } audio)
        {
            string transcript = options.ReferenceText ?? throw new ArgumentException("Dia needs the transcript of the reference recording (SpeechSynthesisOptions.ReferenceText).", nameof(options));
            return (EncodePrompt(audio, cancellationToken), Join(DiaText.EnsureSpeaker(transcript), text));
        }

        return (null, DiaText.EnsureSpeaker(text));
    }

    private static string Join(string transcript, string text)
        => transcript.Length == 0 ? DiaText.EnsureSpeaker(text) : transcript + (char.IsWhiteSpace(transcript[^1]) ? "" : " ") + text;

    private DiaResult Run(string text, DiaGenerationSettings settings, IReadOnlyList<int[]>? prompt, IReadOnlyList<int[]>? forced, ObserveLogits? observe, FramesReady? onFrames,
        CancellationToken cancellationToken)
    {
        var tokens = DiaText.Encode(text);
        if (tokens.Length == 0)
        {
            throw new ArgumentException("There is no text to speak.", nameof(text));
        }

        if (tokens.Length > _maxText)
        {
            throw new ArgumentException($"Dia takes at most {_maxText} text tokens (bytes), the text has {tokens.Length}; split it into shorter requests.", nameof(text));
        }

        int promptRows = (prompt?.Count ?? 0) + 1;
        if (promptRows + _delays.Max() + 1 >= settings.MaxLength)
        {
            throw new ArgumentException($"The audio prompt ({promptRows - 1} frames) leaves no room for speech in a sequence of {settings.MaxLength} rows.", nameof(prompt));
        }

        bool guide = settings.Guidance > 1f;
        var conditional = _conditional ??= new DiaStream(_model, _channels, _vocabulary, _width);
        conditional.Begin(Encode(tokens));
        DiaStream? unconditional = null;
        if (guide)
        {
            unconditional = _unconditional ??= new DiaStream(_model, _channels, _vocabulary, _width);
            unconditional.Begin(Encode(new int[tokens.Length]));
        }

        var table = DiaDelay.Table(prompt, _delays, _bos, _pad);
        int prefill = (prompt?.Count ?? 0) + 1;
        var rows = new List<int[]>(table.Length + 64);
        for (int t = 0; t < prefill; t++)
        {
            rows.Add(table[t]);
        }

        var sampler = new DiaTokenSampler(_channels, _vocabulary, _delays, _eos, _pad, settings);
        int start = prefill;
        int maxDelay = _delays.Max();
        int firstEnd = -1;
        var logits = conditional.Process(rows, 0, prefill);
        var other = unconditional is null ? default : unconditional.Process(rows, 0, prefill);
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            observe?.Invoke(logits, other);
            int[] next;
            int length = rows.Count;
            if (forced is not null)
            {
                if (length >= forced.Count)
                {
                    break;
                }

                next = forced[length];
            }
            else
            {
                if (sampler.Done)
                {
                    break;
                }

                next = sampler.Next(logits, other, length);
                if (firstEnd < 0 && next[0] == _eos)
                {
                    firstEnd = length;
                }

                // The rows of the prompt's delay table take the place of the generated tokens (a codebook that has not started yet sees the start token).
                if (length < table.Length)
                {
                    for (int c = 0; c < _channels; c++)
                    {
                        if (table[length][c] != _pad)
                        {
                            next[c] = table[length][c];
                        }
                    }
                }
            }

            rows.Add(next);
            if (forced is not null && firstEnd < 0 && next[0] == _eos)
            {
                firstEnd = length;
            }

            if (onFrames is not null)
            {
                int complete = (firstEnd >= 0 ? Math.Min(firstEnd, rows.Count - maxDelay) : rows.Count - maxDelay) - start;
                if (complete > 0)
                {
                    onFrames(rows, start, complete);
                }
            }

            if (forced is not null && rows.Count >= forced.Count)
            {
                break;
            }

            logits = conditional.Process(rows, rows.Count - 1, 1);
            other = unconditional is null ? default : unconditional.Process(rows, rows.Count - 1, 1);
        }

        if (firstEnd < 0)
        {
            firstEnd = rows.Count - maxDelay;
        }

        return new DiaResult(rows, start, Math.Max(start, firstEnd));
    }

    // The conditioning rows of a text: the encoder over its tokens.
    private float[] Encode(int[] tokens)
    {
        var text = _text ??= _model.CompileComponent(DiaGraphs.TextRole, DiaGraphs.TextGraph);
        text.SetInput<int>(text.Graph.Inputs.First(v => v.Input!.Name == DiaGraphs.TokensInput), tokens);
        text.SetInput<int>(text.Graph.Inputs.First(v => v.Input!.Name == DiaGraphs.PositionsInput), Enumerable.Range(0, tokens.Length).ToArray());
        var parameters = new RunParameters();
        parameters.Symbols[WellKnown.TokenSymbol] = tokens.Length;
        text.Run(parameters);
        var rows = new float[tokens.Length * _width];
        text.GetOutput<float>(DiaGraphs.ConditionOutput, rows);
        return rows;
    }

    /// <summary>Decodes frames <c>[first, end)</c> of <paramref name="frames"/> with up to <paramref name="context"/> frames before them whose audio is dropped.</summary>
    private AudioBuffer DecodeSlice(IReadOnlyList<int[]> frames, int first, int end, int context)
    {
        int begin = Math.Max(0, first - context);
        int count = Math.Min(frames.Count, end) - begin;
        if (count <= 0)
        {
            return new AudioBuffer([], SampleRate);
        }

        var codec = _decoder ??= _model.CompileComponent(DiaGraphs.CodecDecoderRole, AudioCodecGraphs.DecodeGraph);
        var mask = new float[_channels];
        mask.AsSpan().Fill(1f);
        codec.SetInput<float>(codec.Graph.Inputs.First(v => v.Input!.Name == AudioCodecGraphs.MaskInput), mask);
        for (int c = 0; c < _channels; c++)
        {
            var codes = new int[count];
            for (int f = 0; f < count; f++)
            {
                codes[f] = Math.Clamp(frames[begin + f][c], 0, _codecSize - 1);
            }

            codec.SetInput<int>(codec.Graph.Inputs.First(v => v.Input!.Name == AudioCodecGraphs.CodesName(c)), codes);
        }

        var parameters = new RunParameters();
        parameters.Symbols[WellKnown.TokenSymbol] = count;
        codec.Run(parameters);
        var samples = new float[count * _frameSamples];
        codec.GetOutput<float>(AudioCodecGraphs.AudioOutput, samples);
        int skip = (first - begin) * _frameSamples;
        int take = Math.Min(end, begin + count) * _frameSamples - (begin * _frameSamples) - skip;
        return new AudioBuffer(samples.AsSpan(skip, take).ToArray(), SampleRate);
    }
}
