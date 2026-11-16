using System.Runtime.InteropServices;
using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Speech;
using Sharplight.Tasks;
using Sharplight.Tokenizers;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// The <see cref="ISpeechSynthesizer"/> of XTTS v2 (<see cref="XttsAdapter"/>), following Coqui's <c>Xtts.inference</c>: the text cleaned (English:
/// <see cref="XttsTextNormalizer"/>; other languages lowercased), prefixed with the language token, spaces as <c>[SPACE]</c>, between the start and stop
/// text tokens; the GPT runs on the speaker's 32 conditioning latents, the text rows (embedding plus learned position) and the start-of-audio row, then
/// code by code (code embedding plus learned audio position), sampling with the repetition penalty over the generated codes (and the two placeholder
/// ids the reference's prompt carries); the hidden states at the start row and at every code are the latents; they are linearly
/// interpolated to the decoder's frame rate (×4, then ×24000/22050) and decoded by the HiFi-GAN with the speaker embedding. Long text is spoken sentence
/// by sentence (<see cref="XttsTextSplitter"/>); the sampling defaults are the model's own. One request runs at a time.
/// </summary>
internal sealed class XttsSynthesizer : ISpeechSynthesizer
{
    private const string SpaceToken = "[SPACE]";

    // The reference's text_split_length for English: longer text is spoken sentence by sentence.
    private const int MaxSentenceLength = 250;

    private readonly TaskModel _model;
    private readonly LlmModel _llm;
    private readonly ITokenizer _tokenizer;
    private readonly int _startText;
    private readonly int _stopText;
    private readonly int _startAudio;
    private readonly int _stopAudio;
    private readonly int _maxAudioTokens;
    private readonly double _upsample;
    private readonly double _resample;
    private readonly int _samplesPerFrame;
    private readonly int _width;
    private readonly SamplingOptions _defaultSampling;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly Lazy<Tables> _tables;
    private InferenceSession? _session;
    private CompiledModel? _vocoder;
    private XttsVoiceCloning? _cloning;
    private (AudioBuffer Recording, XttsVoice Voice)? _lastClone;
    private bool _disposed;

    /// <summary>Creates the synthesizer; nothing is compiled until the first request.</summary>
    /// <param name="model">A loaded XTTS model.</param>
    public XttsSynthesizer(TaskModel model)
    {
        _model = model;
        _llm = LlmModel.FromTaskModel(model);
        _tokenizer = _llm.Tokenizer;
        var text = model.Description.Component(XttsAdapter.TextModelRole);
        _startText = text.ConfigValue("startText", 261);
        _stopText = text.ConfigValue("stopText", 0);
        _startAudio = text.ConfigValue("startAudio", 1024);
        _stopAudio = text.ConfigValue("stopAudio", 1025);
        _maxAudioTokens = text.ConfigValue("maxAudioTokens", 605);
        _defaultSampling = new SamplingOptions
        {
            Temperature = text.ConfigValue("temperature", 0.75f),
            TopK = text.ConfigValue("topK", 50),
            TopP = text.ConfigValue("topP", 0.85f),
            RepetitionPenalty = text.ConfigValue("repetitionPenalty", 5f),
        };
        _upsample = text.ConfigValue("codeStride", 1024) / (double)text.ConfigValue("outputHop", 256);
        SampleRate = model.Description.ConfigValue("sampleRate", 24000);
        _resample = SampleRate / (double)text.ConfigValue("inputSampleRate", 22050);
        _samplesPerFrame = model.Description.Component(XttsAdapter.VocoderRole).ConfigValue("samplesPerFrame", 256);
        _width = model.Definition.Property(WellKnown.EmbeddingLength, 1024);
        Voices = [.. text.Attachments.Keys
            .Where(k => k.EndsWith("/speaker.f32", StringComparison.Ordinal))
            .Select(k => k[XttsAdapter.VoicePrefix.Length..^"/speaker.f32".Length])
            .Order(StringComparer.Ordinal)];
        _tables = new Lazy<Tables>(() => new Tables(model.Definition));
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    /// <inheritdoc/>
    public TaskClientMetadata Metadata { get; }

    /// <inheritdoc/>
    public int SampleRate { get; }

    /// <inheritdoc/>
    public IReadOnlyList<string> Voices { get; }

    /// <summary>Whether <paramref name="model"/> is an XTTS model.</summary>
    /// <param name="model">A loaded speech-synthesis model.</param>
    public static bool Handles(TaskModel model)
        => model.Description.TryGetComponent(XttsAdapter.TextModelRole, out var text) && text.ConfigValue("family", "") == XttsAdapter.Family
           && model.Description.TryGetComponent(XttsAdapter.VocoderRole, out _);

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
    public async Task<SpeechSynthesisResult> SynthesizeAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        ObjectDisposedException.ThrowIf(_disposed, this);
        string name = options?.ReferenceAudio is not null ? "reference" : options?.Voice ?? DefaultVoice;
        var audio = await BackgroundCompute.RunAsync(_gate, () => Synthesize(text, options, cancellationToken), cancellationToken).ConfigureAwait(false);
        return new SpeechSynthesisResult(audio, name);
    }

    /// <inheritdoc/>
    /// <remarks>The whole utterance as one buffer.</remarks>
    public async IAsyncEnumerable<AudioBuffer> SynthesizeStreamingAsync(string text, SpeechSynthesisOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var result = await SynthesizeAsync(text, options, cancellationToken).ConfigureAwait(false);
        yield return result.Audio;
    }

    /// <summary>The text tokens of a request: the cleaned text after the language token, spaces as <c>[SPACE]</c>, between the start and stop tokens.</summary>
    /// <param name="text">The text.</param>
    /// <param name="language">The language code (default <c>en</c>).</param>
    public int[] TextTokens(string text, string? language = null)
    {
        string lang = (language ?? "en").Split('-')[0].ToLowerInvariant();
        string cleaned = lang == "en" ? XttsTextNormalizer.English(text) : string.Join(' ', text.ToLowerInvariant().Replace("\"", "", StringComparison.Ordinal)
            .Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
        string tagged = $"[{(lang == "zh" ? "zh-cn" : lang)}]{cleaned}".Replace(" ", SpaceToken, StringComparison.Ordinal);
        return [_startText, .. _tokenizer.Encode(tagged, addSpecial: false, parseSpecial: true), _stopText];
    }

    /// <summary>
    /// The audio codes (without the stop code) and the latents [codes + 1, width] (the start row and every code) of the text tokens for a voice; a temperature of zero is greedy. The
    /// repetition penalty of <paramref name="sampling"/> applies to the generated codes and the reference's placeholder ids.
    /// </summary>
    /// <param name="textTokens">The text tokens with start and stop.</param>
    /// <param name="voice">The voice name.</param>
    /// <param name="sampling">The sampling settings.</param>
    /// <param name="cancellationToken">Stops between codes.</param>
    public (int[] Codes, float[] Latents) Generate(int[] textTokens, XttsVoice voice, SamplingOptions sampling, CancellationToken cancellationToken)
    {
        var tables = _tables.Value;
        var session = _session ??= _llm.CreateSession(WellKnown.DecodeMultimodalGraph);
        session.CaptureHidden = true;
        session.Reset();
        ArgumentNullException.ThrowIfNull(voice);
        var latents = voice.Latents;
        int conditionRows = latents.Length / _width;
        int rows = conditionRows + textTokens.Length + 1;
        var prompt = new float[rows * _width];
        latents.CopyTo(prompt, 0);
        for (int i = 0; i < textTokens.Length; i++)
        {
            var target = prompt.AsSpan((conditionRows + i) * _width, _width);
            tables.TextEmbedding(textTokens[i]).CopyTo(target);
            Add(target, tables.TextPosition(i));
        }

        AudioRow(tables, _startAudio, 0).CopyTo(prompt, (rows - 1) * _width);
        var logits = Process(session, prompt, rows);
        var hidden = new List<float[]> { session.LastHidden[^_width..].ToArray() };
        var sampler = new Sampler(sampling with { RepetitionPenalty = 1f });
        var history = new HashSet<int> { 1, _startAudio };
        var codes = new List<int>();
        for (int i = 0; i < _maxAudioTokens; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ApplyRepetitionPenalty(logits, history, sampling.RepetitionPenalty);
            int code = sampling.Temperature <= 0f ? Sampler.ArgMax(logits) : sampler.Sample(logits);
            history.Add(code);
            if (code == _stopAudio)
            {
                break;
            }

            codes.Add(code);
            logits = Process(session, AudioRow(tables, code, i + 1), 1);
            hidden.Add(session.LastHidden[^_width..].ToArray());
        }

        var result = new float[hidden.Count * _width];
        for (int r = 0; r < hidden.Count; r++)
        {
            hidden[r].CopyTo(result, r * _width);
        }

        return ([.. codes], result);
    }

    /// <summary>The waveform of latents [rows, width] for a voice: linear interpolation to the decoder's frame rate, then the HiFi-GAN.</summary>
    /// <param name="latents">The latents.</param>
    /// <param name="voice">The voice name.</param>
    public float[] Decode(float[] latents, XttsVoice voice)
    {
        var frames = Interpolate(Interpolate(latents, _width, _upsample), _width, _resample);
        int count = frames.Length / _width;
        if (count == 0)
        {
            return [];
        }

        var vocoder = _vocoder ??= _model.CompileComponent(XttsAdapter.VocoderRole);
        vocoder.SetInput<float>(Input(vocoder, XttsVocoderGraph.LatentsInput), frames);
        vocoder.SetInput<float>(Input(vocoder, XttsVocoderGraph.SpeakerInput), voice.Speaker);
        var run = new RunParameters();
        run.Symbols[WellKnown.TokenSymbol] = count;
        vocoder.Run(run);
        var samples = new float[count * _samplesPerFrame];
        vocoder.GetOutput<float>(XttsVocoderGraph.AudioOutput, samples);
        return samples;
    }

    /// <summary>
    /// torch's <c>F.interpolate(mode="linear", align_corners=False)</c> along the rows of [rows, width] by <paramref name="scale"/>: <c>floor(rows · scale)</c>
    /// rows, source coordinate <c>(row + 0.5) / scale − 0.5</c> clamped at zero.
    /// </summary>
    /// <param name="values">The rows.</param>
    /// <param name="width">Values per row.</param>
    /// <param name="scale">The scale factor.</param>
    public static float[] Interpolate(float[] values, int width, double scale)
    {
        int rows = values.Length / width;
        int count = (int)Math.Floor(rows * scale);
        var result = new float[count * width];
        float inverse = (float)(1.0 / scale);
        for (int r = 0; r < count; r++)
        {
            float source = Math.Max(((r + 0.5f) * inverse) - 0.5f, 0f);
            int i0 = Math.Min((int)source, rows - 1);
            int i1 = Math.Min(i0 + 1, rows - 1);
            float lambda = source - i0;
            for (int c = 0; c < width; c++)
            {
                result[(r * width) + c] = ((1f - lambda) * values[(i0 * width) + c]) + (lambda * values[(i1 * width) + c]);
            }
        }

        return result;
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
        _vocoder?.Dispose();
        _cloning?.Dispose();
        _gate.Dispose();
    }

    /// <summary>Whether the model can speak with the voice of a recording (<see cref="SpeechSynthesisOptions.ReferenceAudio"/>).</summary>
    public bool SupportsCloning => XttsVoiceCloning.Supports(_model);

    /// <summary>A built-in speaker.</summary>
    /// <param name="name">The speaker's name.</param>
    public XttsVoice LoadVoice(string name) => new(Floats(name, "latents.f32"), Floats(name, "speaker.f32"));

    /// <summary>The voice of a recording (compiles the cloning encoders on first use).</summary>
    /// <param name="recording">The recording.</param>
    /// <param name="cancellationToken">Stops between stages.</param>
    /// <exception cref="NotSupportedException">The model was imported without the cloning encoders.</exception>
    public XttsVoice CloneVoice(AudioBuffer recording, CancellationToken cancellationToken = default)
    {
        if (!SupportsCloning)
        {
            throw new NotSupportedException("This XTTS model has no style and speaker encoders, so it cannot clone a voice.");
        }

        var cloning = _cloning ??= new XttsVoiceCloning(_model);
        return cloning.Clone(recording, cancellationToken);
    }

    // The recording's voice when the request has one (kept for the next request with the same recording), else the named speaker.
    private XttsVoice VoiceFor(SpeechSynthesisOptions? options, CancellationToken cancellationToken)
    {
        if (options?.ReferenceAudio is not { } recording)
        {
            return LoadVoice(options?.Voice ?? DefaultVoice);
        }

        if (_lastClone is { } last && ReferenceEquals(last.Recording, recording))
        {
            return last.Voice;
        }

        var voice = CloneVoice(recording, cancellationToken);
        _lastClone = (recording, voice);
        return voice;
    }

    private string DefaultVoice => Voices.Count > 0 ? Voices[0] : throw new InvalidOperationException("The model has no voices.");

    private AudioBuffer Synthesize(string text, SpeechSynthesisOptions? options, CancellationToken cancellationToken)
    {
        var voice = VoiceFor(options, cancellationToken);
        var sampling = options.SamplingOr(_defaultSampling);
        var samples = new List<float>();
        foreach (string sentence in XttsTextSplitter.Split(text, MaxSentenceLength))
        {
            var (_, latents) = Generate(TextTokens(sentence, options?.Language), voice, sampling, cancellationToken);
            samples.AddRange(Decode(latents, voice));
        }

        return new AudioBuffer([.. samples], SampleRate);
    }

    private float[] Process(InferenceSession session, float[] rows, int count)
    {
        var prompt = new MultimodalPrompt
        {
            Tokens = new int[count],
            ImageIndex = new int[count],
            ImageRow = [.. Enumerable.Range(0, count)],
            Positions = [.. Enumerable.Range(session.RopePosition, count)],
            PositionAxes = 1,
            NextPosition = session.RopePosition + count,
        };
        return session.Process(prompt, [rows]);
    }

    private float[] AudioRow(Tables tables, int code, int position)
    {
        var row = new float[_width];
        tables.AudioEmbedding(code).CopyTo(row);
        Add(row, tables.AudioPosition(position));
        return row;
    }

    private static void Add(Span<float> target, ReadOnlySpan<float> values)
    {
        for (int i = 0; i < target.Length; i++)
        {
            target[i] += values[i];
        }
    }

    private static void ApplyRepetitionPenalty(float[] logits, HashSet<int> history, float penalty)
    {
        if (penalty == 1f)
        {
            return;
        }

        foreach (int token in history)
        {
            if (token < logits.Length)
            {
                float value = logits[token];
                logits[token] = value < 0 ? value * penalty : value / penalty;
            }
        }
    }

    private static Value Input(CompiledModel compiled, string name) => compiled.Graph.Inputs.First(v => v.Input!.Name == name);

    private float[] Floats(string voice, string part)
        => MemoryMarshal.Cast<byte, float>(_model.ReadAttachment(XttsAdapter.TextModelRole, $"{XttsAdapter.VoicePrefix}{voice}/{part}")).ToArray();

    /// <summary>The GPT's input tables, read once on the host.</summary>
    private sealed class Tables(ModelDefinition definition)
    {
        private readonly float[] _text = WeightConversions.ReadFloats(definition.Weights["gpt.text_embedding.weight"]);
        private readonly float[] _textPositions = WeightConversions.ReadFloats(definition.Weights["gpt.text_pos_embedding.emb.weight"]);
        private readonly float[] _audio = WeightConversions.ReadFloats(definition.Weights["token_embedding.weight"]);
        private readonly float[] _audioPositions = WeightConversions.ReadFloats(definition.Weights["gpt.mel_pos_embedding.emb.weight"]);
        private readonly int _width = (int)definition.Weights["token_embedding.weight"].Shape[1].Fixed;

        public ReadOnlySpan<float> TextEmbedding(int token) => _text.AsSpan(token * _width, _width);

        public ReadOnlySpan<float> TextPosition(int position) => _textPositions.AsSpan(position * _width, _width);

        public ReadOnlySpan<float> AudioEmbedding(int code) => _audio.AsSpan(code * _width, _width);

        public ReadOnlySpan<float> AudioPosition(int position) => _audioPositions.AsSpan(position * _width, _width);
    }
}
