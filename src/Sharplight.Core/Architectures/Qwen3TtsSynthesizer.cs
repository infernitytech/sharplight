using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Speech;
using Sharplight.Tasks;
using Sharplight.Tensors;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// The <see cref="ISpeechSynthesizer"/> of a Qwen3-TTS model (<see cref="Qwen3TtsAdapter"/>), as the official <c>qwen_tts</c> package's <c>generate_custom_voice</c>
/// (built-in speakers), <c>generate_voice_design</c> (a voice described by <see cref="SpeechSynthesisOptions.Instructions"/>) and <c>generate_voice_clone</c> (a reference
/// recording, <see cref="IVoiceCloner"/>): the talker's prompt is the projected instruction (<c>&lt;|im_start|&gt;user\n…&lt;|im_end|&gt;\n</c>, when the model reads
/// one), the projected text of <c>&lt;|im_start|&gt;assistant\n</c>, the codec control rows (think or no-think with the language, the speaker's codec row or the
/// reference's speaker embedding, pad and bos) under text pad / bos rows, then the text. Non-streaming text (the custom-voice and voice-design default) prefills the whole
/// text and its end over codec pad rows; interleaved text (the voice-cloning default) prefills its first token over codec bos and adds one more text row to each of the
/// first generation steps. With a reference transcript the prompt continues the reference instead (in-context): the reference's and the request's text over the
/// reference's frames. Every step the talker predicts a frame's first codebook, the code predictor the other fifteen from the talker's hidden state, and the sum of the
/// frame's sixteen codebook embeddings (plus the next text row or the text pad row) is the talker's next input. The codec decoder turns the frames (after the reference's
/// frames, whose audio is dropped) into 24 kHz audio in chunks of <see cref="Qwen3TtsAdapter.CodecChunkFrames"/> frames with
/// <see cref="Qwen3TtsAdapter.CodecContextFrames"/> frames of context, like the reference.
/// </summary>
internal sealed class Qwen3TtsSynthesizer : ISpeechSynthesizer, IVoiceCloner
{
    /// <summary>Frames decoded per streamed buffer.</summary>
    public const int StreamFrames = 25;

    private const int SuppressedTail = 1024;

    private readonly TaskModel _model;
    private readonly LlmModel _llm;
    private readonly Dictionary<string, (int Id, string Dialect)> _speakers;
    private readonly Dictionary<string, int> _languages;
    private readonly int _textBos;
    private readonly int _textEos;
    private readonly int _textPad;
    private readonly int _groups;
    private readonly int _codecBos;
    private readonly int _codecEos;
    private readonly int _codecPad;
    private readonly int _think;
    private readonly int _nothink;
    private readonly int _thinkBos;
    private readonly int _thinkEos;
    private readonly int _frameSamples;
    private readonly SamplingOptions _defaults;
    private readonly SamplingOptions _predictorDefaults;
    private readonly bool _instructions;
    private readonly bool _interleaved;
    private readonly bool _cloning;
    private readonly Lazy<IAudioFeatureExtractor>? _speakerFeatures;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly Lazy<Tables> _tables;
    private InferenceSession? _talker;
    private InferenceSession? _predictor;
    private CompiledModel? _codec;
    private CompiledModel? _codecEncoder;
    private CompiledModel? _speakerEncoder;
    private bool _disposed;

    /// <summary>Creates the synthesizer; nothing is compiled until the first request.</summary>
    /// <param name="model">A loaded Qwen3-TTS model.</param>
    public Qwen3TtsSynthesizer(TaskModel model)
    {
        _model = model;
        _llm = LlmModel.FromTaskModel(model);
        var text = model.Description.Component(Qwen3TtsAdapter.TextModelRole);
        _textBos = text.ConfigValue("textBos", 0);
        _textEos = text.ConfigValue("textEos", 0);
        _textPad = text.ConfigValue("textPad", 0);
        _groups = text.ConfigValue("codeGroups", 16);
        _codecBos = text.ConfigValue("codecBos", 0);
        _codecEos = text.ConfigValue("codecEos", 0);
        _codecPad = text.ConfigValue("codecPad", 0);
        _think = text.ConfigValue("codecThink", 0);
        _nothink = text.ConfigValue("codecNothink", 0);
        _thinkBos = text.ConfigValue("codecThinkBos", 0);
        _thinkEos = text.ConfigValue("codecThinkEos", 0);
        _instructions = text.ConfigValue("instructions", false);
        _interleaved = text.ConfigValue("interleavedText", false);
        _cloning = model.Description.TryGetComponent(Qwen3TtsAdapter.SpeakerEncoderRole, out _) && model.Description.TryGetComponent(Qwen3TtsAdapter.CodecEncoderRole, out _);
        if (_cloning)
        {
            _speakerFeatures = new Lazy<IAudioFeatureExtractor>(() => AudioFeatureExtractors.Create(model, Qwen3TtsAdapter.SpeakerFeaturesRole));
        }

        _speakers = text.ConfigValue("speakers", "").Split(',', StringSplitOptions.RemoveEmptyEntries).Select(e => e.Split('='))
            .ToDictionary(p => p[0], p => (int.Parse(p[1], System.Globalization.CultureInfo.InvariantCulture), p[2]), StringComparer.OrdinalIgnoreCase);
        _languages = text.ConfigValue("languages", "").Split(',', StringSplitOptions.RemoveEmptyEntries).Select(e => e.Split('='))
            .ToDictionary(p => p[0], p => int.Parse(p[1], System.Globalization.CultureInfo.InvariantCulture), StringComparer.OrdinalIgnoreCase);
        _defaults = new SamplingOptions
        {
            Temperature = (float)text.ConfigValue("sampling.temperature", 0.9), TopK = (int)text.ConfigValue("sampling.top_k", 50.0), TopP = (float)text.ConfigValue("sampling.top_p", 1.0),
            MinP = 0f, RepetitionPenalty = (float)text.ConfigValue("sampling.repetition_penalty", 1.05), RepetitionWindow = Qwen3TtsAdapter.MaxFrames,
        };
        _predictorDefaults = new SamplingOptions
        {
            Temperature = (float)text.ConfigValue("sampling.code_predictor.temperature", 0.9), TopK = (int)text.ConfigValue("sampling.code_predictor.top_k", 50.0),
            TopP = (float)text.ConfigValue("sampling.code_predictor.top_p", 1.0), MinP = 0f,
        };
        var codec = model.Description.Component(Qwen3TtsAdapter.CodecRole);
        SampleRate = codec.ConfigValue("sampleRate", 24000);
        _frameSamples = codec.ConfigValue("samplesPerFrame", 1920);
        Voices = [.. _speakers.Keys.Order(StringComparer.Ordinal)];
        Languages = [.. _languages.Keys.Where(l => !l.Contains("dialect", StringComparison.Ordinal)).Order(StringComparer.Ordinal)];
        _tables = new Lazy<Tables>(() => new Tables(model.Definition, _groups - 1));
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    /// <inheritdoc/>
    public TaskClientMetadata Metadata { get; }

    /// <inheritdoc/>
    public int SampleRate { get; }

    /// <inheritdoc/>
    /// <remarks>The built-in speakers of a custom-voice model; voice-cloning and voice-design models have none (they speak with a reference's or a described voice).</remarks>
    public IReadOnlyList<string> Voices { get; }

    /// <summary>Whether the model reads <see cref="SpeechSynthesisOptions.Instructions"/> (voice design, and the larger custom-voice models' speaking style).</summary>
    public bool AcceptsInstructions => _instructions;

    /// <inheritdoc/>
    public string VoiceArchitecture => Qwen3TtsAdapter.Family;

    /// <summary>The languages a request can name (<see cref="SpeechSynthesisOptions.Language"/>; none or <c>auto</c> lets the model choose).</summary>
    public IReadOnlyList<string> Languages { get; }

    /// <summary>Whether <paramref name="model"/> is a Qwen3-TTS model (<see cref="Qwen3TtsAdapter"/>).</summary>
    /// <param name="model">A loaded speech-synthesis model.</param>
    public static bool Handles(TaskModel model)
        => model.Description.TryGetComponent(Qwen3TtsAdapter.TextModelRole, out var text) && text.ConfigValue("family", "") == Qwen3TtsAdapter.Family
           && model.Description.TryGetComponent(Qwen3TtsAdapter.CodecRole, out _);

    /// <inheritdoc/>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
        {
            return null;
        }

        if (serviceType == typeof(IVoiceCloner) && !_cloning)
        {
            return null;
        }

        return serviceType.IsInstanceOfType(this) ? this : serviceType.IsInstanceOfType(_model) ? _model : serviceType.IsInstanceOfType(_llm) ? _llm : null;
    }

    /// <inheritdoc/>
    public async Task<SpeechSynthesisResult> SynthesizeAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
    {
        string normalized = options.NormalizedText(text);
        var audio = await BackgroundCompute.RunAsync(_gate, () =>
        {
            var prompt = BuildPrompt(normalized, options, cancellationToken);
            var frames = new List<int[]>(prompt.ReferenceFrames);
            frames.AddRange(GenerateFrames(prompt, options, cancellationToken));
            return DecodeAfter(frames, prompt.ReferenceFrames.Count, cancellationToken);
        }, cancellationToken).ConfigureAwait(false);
        return new SpeechSynthesisResult(audio, options?.VoiceProfile?.Name ?? options?.Voice ?? (Voices.Count > 0 ? Voices[0] : ""));
    }

    /// <inheritdoc/>
    /// <remarks>The voice is the recording's speaker embedding and, with <paramref name="referenceText"/>, its codec frames for the in-context prompt.</remarks>
    /// <exception cref="NotSupportedException">The model does not clone voices (it is not a voice-cloning model).</exception>
    public async Task<VoiceProfile> CreateVoiceAsync(string name, AudioBuffer recording, string? referenceText = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(recording);
        ObjectDisposedException.ThrowIf(_disposed, this);
        RequireCloning();
        var reference = await BackgroundCompute.RunAsync(_gate, () => EncodeReference(recording, referenceText, cancellationToken), cancellationToken).ConfigureAwait(false);
        var tensors = new Dictionary<string, VoiceTensor>(StringComparer.Ordinal) { ["speaker"] = VoiceTensor.From(reference.Speaker) };
        if (reference.Frames is { } frames)
        {
            tensors["codes"] = VoiceTensor.From([.. frames.SelectMany(f => f)], [frames.Count, _groups]);
        }

        return new VoiceProfile { Name = name, Architecture = VoiceArchitecture, ReferenceText = reference.Text, SampleRate = SampleRate, Tensors = tensors };
    }

    /// <summary>
    /// The voice of a recording: the speaker encoder's embedding and, when <paramref name="text"/> is given, the codec frames of the recording (in-context prompt).
    /// Recordings longer than <see cref="Qwen3TtsAdapter.MaxReferenceFrames"/> frames are cut.
    /// </summary>
    /// <param name="recording">The recording (any rate and channel count).</param>
    /// <param name="text">The words spoken in it, or <see langword="null"/> for the speaker embedding alone.</param>
    /// <param name="cancellationToken">Stops between the encoders.</param>
    public Qwen3TtsReference EncodeReference(AudioBuffer recording, string? text, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(recording);
        RequireCloning();
        var all = recording.ToModelInput(SampleRate).Samples;
        var samples = all.AsSpan(0, Math.Min(all.Length, Qwen3TtsAdapter.MaxReferenceFrames * _frameSamples));
        if (samples.Length < _frameSamples)
        {
            throw new ArgumentException("The reference recording is shorter than one codec frame.", nameof(recording));
        }

        var speaker = SpeakerEmbedding(samples);
        cancellationToken.ThrowIfCancellationRequested();
        return string.IsNullOrEmpty(text) ? new Qwen3TtsReference(speaker, null, null) : new Qwen3TtsReference(speaker, ReferenceFrames(samples, cancellationToken), text);
    }

    /// <summary>The speaker encoder's embedding of a recording at the model's rate.</summary>
    /// <param name="samples">Mono samples at <see cref="SampleRate"/>.</param>
    public float[] SpeakerEmbedding(ReadOnlySpan<float> samples)
    {
        RequireCloning();
        var features = _speakerFeatures!.Value.Extract(samples);
        var encoder = _speakerEncoder ??= _model.CompileComponent(Qwen3TtsAdapter.SpeakerEncoderRole, EcapaTdnnGraph.GraphName);
        encoder.SetInput<float>(encoder.Graph.Inputs.First(v => v.Input!.Name == EcapaTdnnGraph.FeaturesInput), S3SpeechEncoders.TimeMajor(features));
        var run = new RunParameters();
        run.Symbols[WellKnown.TokenSymbol] = features.Frames;
        encoder.Run(run);
        var embedding = new float[(int)encoder.Graph.Outputs[EcapaTdnnGraph.EmbeddingOutput].Shape[-1].Fixed];
        encoder.GetOutput<float>(EcapaTdnnGraph.EmbeddingOutput, embedding);
        return embedding;
    }

    /// <summary>The codec frames (16 codes each) of a recording at the model's rate, zero padded to whole frames and encoded in segments of the codec's limit.</summary>
    /// <param name="samples">Mono samples at <see cref="SampleRate"/>.</param>
    /// <param name="cancellationToken">Stops between segments.</param>
    public IReadOnlyList<int[]> ReferenceFrames(ReadOnlySpan<float> samples, CancellationToken cancellationToken = default)
    {
        RequireCloning();
        var encoder = _codecEncoder ??= _model.CompileComponent(Qwen3TtsAdapter.CodecEncoderRole, AudioCodecGraphs.EncodeGraph);
        var component = _model.Description.Component(Qwen3TtsAdapter.CodecEncoderRole);
        int rowsPerSample = component.ConfigValue(AudioCodecGraphs.PositionsDenominatorKey, 1);
        int frames = (samples.Length + _frameSamples - 1) / _frameSamples;
        var result = new List<int[]>(frames);
        for (int start = 0; start < frames; start += Qwen3TtsCodecAdapter.MaxFrames)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(Qwen3TtsCodecAdapter.MaxFrames, frames - start);
            var segment = new float[count * _frameSamples];
            int from = start * _frameSamples;
            samples.Slice(from, Math.Min(segment.Length, samples.Length - from)).CopyTo(segment);
            encoder.SetInput<float>(encoder.Graph.Inputs.First(v => v.Input!.Name == AudioCodecGraphs.AudioInput), segment);
            SetPositions(encoder, segment.Length / rowsPerSample);
            var run = new RunParameters();
            run.Symbols[WellKnown.TokenSymbol] = segment.Length;
            encoder.Run(run);
            var part = new int[count];
            var codes = new int[count][];
            for (int f = 0; f < count; f++)
            {
                codes[f] = new int[_groups];
            }

            for (int q = 0; q < _groups; q++)
            {
                encoder.GetOutput<int>(AudioCodecGraphs.CodesName(q), part);
                for (int f = 0; f < count; f++)
                {
                    codes[f][q] = part[f];
                }
            }

            result.AddRange(codes);
        }

        return result;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Returns at once: generation and decoding run on a background task (<see cref="BackgroundCompute"/>), one request at a time per client; every
    /// <see cref="StreamFrames"/> frames are decoded with up to <see cref="Qwen3TtsAdapter.CodecContextFrames"/> frames of context before them.
    /// </remarks>
    public IAsyncEnumerable<AudioBuffer> SynthesizeStreamingAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
        => BackgroundCompute.StreamAsync(token => SynthesizeCoreAsync(text, options, token), cancellationToken);

    private async IAsyncEnumerable<AudioBuffer> SynthesizeCoreAsync(string text, SpeechSynthesisOptions? options, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            string normalized = options.NormalizedText(text);
            var prompt = await Task.Run(() => BuildPrompt(normalized, options, cancellationToken), cancellationToken).ConfigureAwait(false);
            var frames = new List<int[]>(prompt.ReferenceFrames);
            int decoded = frames.Count;
            using var generation = GenerateFrames(prompt, options, cancellationToken).GetEnumerator();
            while (await Task.Run(generation.MoveNext, cancellationToken).ConfigureAwait(false))
            {
                frames.Add(generation.Current);
                if (frames.Count - decoded >= StreamFrames)
                {
                    int end = frames.Count;
                    yield return await Task.Run(() => DecodeSlice(frames, decoded, end, Math.Min(Qwen3TtsAdapter.CodecContextFrames, decoded)), cancellationToken).ConfigureAwait(false);
                    decoded = end;
                }
            }

            if (frames.Count > decoded)
            {
                yield return await Task.Run(() => DecodeSlice(frames, decoded, frames.Count, Math.Min(Qwen3TtsAdapter.CodecContextFrames, decoded)), cancellationToken).ConfigureAwait(false);
            }
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>The token ids of the request's text: <c>&lt;|im_start|&gt;assistant\n{text}&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n</c>.</summary>
    /// <param name="text">The text to speak.</param>
    public int[] TokensFor(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return _llm.Tokenizer.Encode($"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n", addSpecial: false, parseSpecial: true);
    }

    /// <summary>The talker's prompt rows [rows, hidden] for a request with a built-in or no speaker (no reference recording).</summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="voice">The speaker (one of <see cref="Voices"/>), or <see langword="null"/> for the first (none for a model without speakers).</param>
    /// <param name="language">The language (one of <see cref="Languages"/>), or <see langword="null"/> / <c>auto</c> to let the model choose.</param>
    /// <param name="instructions">The instruction (models that read one), or <see langword="null"/>.</param>
    public float[] PromptFor(string text, string? voice = null, string? language = null, string? instructions = null)
        => BuildPrompt(text, new SpeechSynthesisOptions { Voice = voice, Language = language, Instructions = instructions }).Rows;

    /// <summary>The talker's input for a request: its voice (built-in speaker, reference recording or saved voice), language, instruction and text.</summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Stops while a reference recording is encoded.</param>
    public Qwen3TtsPrompt BuildPrompt(string text, SpeechSynthesisOptions? options, CancellationToken cancellationToken = default)
    {
        Qwen3TtsReference? reference = null;
        if (options?.VoiceProfile is { } profile)
        {
            reference = ReferenceOf(profile);
        }
        else if (options?.ReferenceAudio is { } recording)
        {
            reference = EncodeReference(recording, options.ReferenceText, cancellationToken);
        }

        return BuildPrompt(text, options, reference);
    }

    /// <summary>The talker's input for a request with an already encoded reference (or none).</summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="options">Request settings (voice, language, instructions), or <see langword="null"/>; its reference recording and saved voice are not read.</param>
    /// <param name="reference">The reference voice (voice-cloning models), or <see langword="null"/>.</param>
    public Qwen3TtsPrompt BuildPrompt(string text, SpeechSynthesisOptions? options, Qwen3TtsReference? reference)
    {
        var tables = _tables.Value;
        int h = tables.Hidden;
        var ids = TokensFor(text);
        if (ids.Length < 9)
        {
            throw new ArgumentException("The text is empty.", nameof(text));
        }

        var (speakerRow, dialect) = SpeakerOf(options?.Voice, reference);
        int? languageId = LanguageOf(options?.Language);

        // A dialect speaker speaks its dialect when the language is Chinese or chosen by the model.
        if (dialect.Length > 0 && (languageId is null || languageId == _languages.GetValueOrDefault("chinese", -1)))
        {
            languageId = _languages[dialect];
        }

        var rows = new List<float[]>();
        if (_instructions && !string.IsNullOrWhiteSpace(options?.Instructions))
        {
            foreach (int id in _llm.Tokenizer.Encode($"<|im_start|>user\n{options.Instructions}<|im_end|>\n", addSpecial: false, parseSpecial: true))
            {
                rows.Add(tables.TextRow(id));
            }
        }

        for (int i = 0; i < 3; i++)
        {
            rows.Add(tables.TextRow(ids[i]));
        }

        var codec = new List<float[]>();
        int[] control = languageId is { } l ? [_think, _thinkBos, l, _thinkEos] : [_nothink, _thinkBos, _thinkEos];
        codec.AddRange(control.Select(c => tables.Codec(c).ToArray()));
        if (speakerRow is not null)
        {
            codec.Add(speakerRow);
        }

        codec.Add(tables.Codec(_codecPad).ToArray());
        var pad = tables.TextRow(_textPad);
        var bos = tables.TextRow(_textBos);
        var eos = tables.TextRow(_textEos);
        for (int i = 0; i < codec.Count; i++)
        {
            rows.Add(Sum(i < codec.Count - 1 ? pad : bos, codec[i]));
        }

        // The text between the role tokens and the five closing tokens, and its end row.
        var content = ids[3..^5].Select(tables.TextRow).ToList();
        var trailing = new List<float[]>();
        var referenceFrames = reference?.Frames ?? [];
        if (reference is { Frames: { } frames, Text: { } transcript })
        {
            var referenceIds = _llm.Tokenizer.Encode($"<|im_start|>assistant\n{transcript}<|im_end|>\n", addSpecial: false, parseSpecial: true);
            var textRows = referenceIds[3..^2].Select(tables.TextRow).Concat(content).Append(eos).ToList();
            var codecRows = frames.Select(FrameSum).Prepend(tables.Codec(_codecBos).ToArray()).ToList();
            if (!_interleaved)
            {
                rows.AddRange(textRows.Select(t => Sum(t, tables.Codec(_codecPad))));
                rows.AddRange(codecRows.Select(c => Sum(c, pad)));
            }
            else
            {
                for (int i = 0; i < codecRows.Count; i++)
                {
                    rows.Add(Sum(i < textRows.Count ? textRows[i] : pad, codecRows[i]));
                }

                trailing.AddRange(textRows.Skip(codecRows.Count));
            }
        }
        else if (!_interleaved)
        {
            rows.AddRange(content.Append(eos).Select(t => Sum(t, tables.Codec(_codecPad))));
            rows.Add(Sum(pad, tables.Codec(_codecBos)));
        }
        else
        {
            rows.Add(Sum(content[0], tables.Codec(_codecBos)));
            trailing.AddRange(content.Skip(1).Append(eos));
        }

        return new Qwen3TtsPrompt(Flatten(rows, h), Flatten(trailing, h), referenceFrames);
    }

    // The speaker's codec row (built-in speaker) or embedding (reference) and the speaker's dialect; no row for a model without speakers and no reference.
    private (float[]? Row, string Dialect) SpeakerOf(string? voice, Qwen3TtsReference? reference)
    {
        if (reference is not null)
        {
            if (reference.Speaker.Length != _tables.Value.Hidden)
            {
                throw new ArgumentException($"The voice's speaker embedding has {reference.Speaker.Length} values; this model takes {_tables.Value.Hidden} (a voice of another model size).");
            }

            return (reference.Speaker, "");
        }

        if (_cloning)
        {
            throw new ArgumentException("A Qwen3-TTS voice-cloning model speaks with the voice of a recording: set SpeechSynthesisOptions.ReferenceAudio (with ReferenceText for "
                + "the in-context prompt) or VoiceProfile.");
        }

        if (Voices.Count == 0)
        {
            return (null, "");
        }

        string speaker = voice ?? Voices[0];
        if (!_speakers.TryGetValue(speaker, out var spk))
        {
            throw new ArgumentException($"Unknown voice '{speaker}' (voices: {string.Join(", ", Voices)}).", nameof(voice));
        }

        return (_tables.Value.Codec(spk.Id).ToArray(), spk.Dialect);
    }

    private int? LanguageOf(string? language)
    {
        if (string.IsNullOrEmpty(language) || language.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        return _languages.TryGetValue(language, out int id)
            ? id
            : throw new ArgumentException($"Unknown language '{language}' (languages: {string.Join(", ", Languages)}).", nameof(language));
    }

    private Qwen3TtsReference ReferenceOf(VoiceProfile profile)
    {
        profile.RequireArchitecture(VoiceArchitecture);
        RequireCloning();
        var speaker = profile.Tensor("speaker").ToFloats();
        if (!profile.Tensors.TryGetValue("codes", out var codes) || string.IsNullOrEmpty(profile.ReferenceText))
        {
            return new Qwen3TtsReference(speaker, null, null);
        }

        var values = codes.ToInts();
        var frames = Enumerable.Range(0, values.Length / _groups).Select(f => values.AsSpan(f * _groups, _groups).ToArray()).ToList();
        return new Qwen3TtsReference(speaker, frames, profile.ReferenceText);
    }

    private void RequireCloning()
    {
        if (!_cloning)
        {
            throw new NotSupportedException("This Qwen3-TTS model does not clone voices (only the Base models have the speaker encoder).");
        }
    }

    // The sum of a frame's sixteen codebook embeddings (the talker's input for that frame, without the text row).
    private float[] FrameSum(int[] frame)
    {
        var tables = _tables.Value;
        var sum = tables.Codec(frame[0]).ToArray();
        for (int g = 1; g < _groups; g++)
        {
            Add(sum, tables.PredictorEmbedding(g - 1, frame[g]));
        }

        return sum;
    }

    private static float[] Sum(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var sum = a.ToArray();
        Add(sum, b);
        return sum;
    }

    private static float[] Flatten(List<float[]> rows, int width)
    {
        var result = new float[rows.Count * width];
        for (int i = 0; i < rows.Count; i++)
        {
            rows[i].CopyTo(result, i * width);
        }

        return result;
    }

    /// <summary>
    /// Generates the codec frames of a request (16 codes each), one by one; the talker's first-codebook logits of each step go to <paramref name="observe"/> when given.
    /// </summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="options">Request settings (voice, language, sampling), or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Stops generation between frames.</param>
    /// <param name="forced">Feeds these frames instead of the generated ones (teacher forcing), or <see langword="null"/>.</param>
    /// <param name="observe">Receives each step's talker logits and the code predictor's logits of the frame, or <see langword="null"/>.</param>
    public IEnumerable<int[]> GenerateFrames(string text, SpeechSynthesisOptions? options, CancellationToken cancellationToken, IReadOnlyList<int[]>? forced = null,
        Action<float[], IReadOnlyList<float[]>>? observe = null)
        => GenerateFrames(BuildPrompt(text, options, cancellationToken), options, cancellationToken, forced, observe);

    /// <summary>Generates the codec frames of a prepared prompt (<see cref="BuildPrompt(string, SpeechSynthesisOptions?, CancellationToken)"/>), one by one.</summary>
    /// <param name="prompt">The talker's input.</param>
    /// <param name="options">Request settings (sampling), or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Stops generation between frames.</param>
    /// <param name="forced">Feeds these frames instead of the generated ones (teacher forcing), or <see langword="null"/>.</param>
    /// <param name="observe">Receives each step's talker logits and the code predictor's logits of the frame, or <see langword="null"/>.</param>
    public IEnumerable<int[]> GenerateFrames(Qwen3TtsPrompt prompt, SpeechSynthesisOptions? options, CancellationToken cancellationToken, IReadOnlyList<int[]>? forced = null,
        Action<float[], IReadOnlyList<float[]>>? observe = null)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ObjectDisposedException.ThrowIf(_disposed, this);
        var tables = _tables.Value;
        int h = tables.Hidden;
        var sampling = options.SamplingOr(_defaults);
        var predictorSampling = sampling.Temperature <= 0f ? new SamplingOptions { Temperature = 0f, RepetitionPenalty = 1f } : _predictorDefaults with { Seed = sampling.Seed };
        var sampler = new Sampler(sampling with { RepetitionWindow = Math.Max(sampling.RepetitionWindow, Qwen3TtsAdapter.MaxFrames) });
        var predictorSampler = new Sampler(predictorSampling);
        var talker = _talker ??= CreateSession(WellKnown.DecodeMultimodalGraph);
        var predictor = _predictor ??= CreateSession(Qwen3TtsAdapter.CodePredictorGraph);
        talker.Reset();
        int promptRows = prompt.Rows.Length / h;
        var logits = Process(talker, prompt.Rows, promptRows);
        var hidden = talker.LastHidden[^h..].ToArray();
        var pad = tables.TextRow(_textPad);
        int trailing = prompt.Trailing.Length / h;
        int budget = Math.Min(Qwen3TtsAdapter.MaxFrames, talker.ContextLength - promptRows - 1);
        int vocabulary = logits.Length;
        for (int step = 0; step < budget; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int v = vocabulary - SuppressedTail; v < vocabulary; v++)
            {
                if (v != _codecEos || step < 2)
                {
                    logits[v] = float.NegativeInfinity;
                }
            }

            var predictorLogits = observe is null ? null : new List<float[]>();
            int first = forced is not null ? (step < forced.Count ? forced[step][0] : _codecEos) : sampler.Sample(logits);
            if (first == _codecEos)
            {
                observe?.Invoke(logits, []);
                yield break;
            }

            sampler.Accept(first);
            var frame = new int[_groups];
            frame[0] = first;
            predictor.Reset();
            var rows = new float[2 * tables.PredictorHidden];
            tables.Project(hidden).CopyTo(rows, 0);
            tables.Project(tables.Codec(first)).CopyTo(rows, tables.PredictorHidden);
            Process(predictor, rows, 2);
            for (int g = 1; g < _groups; g++)
            {
                var head = tables.Head(g - 1, predictor.LastHidden[^tables.PredictorHidden..]);
                predictorLogits?.Add(head);
                frame[g] = forced is not null ? forced[step][g] : predictorSampler.Sample(head);
                if (g < _groups - 1)
                {
                    Process(predictor, tables.Project(tables.PredictorEmbedding(g - 1, frame[g])), 1);
                }
            }

            observe?.Invoke(logits, predictorLogits!);
            yield return frame;
            var next = FrameSum(frame);
            Add(next, step < trailing ? prompt.Trailing.AsSpan(step * h, h) : pad);

            logits = Process(talker, next, 1);
            hidden = talker.LastHidden[^h..].ToArray();
        }
    }

    /// <summary>Decodes frames of 16 codes into audio (chunks of <see cref="Qwen3TtsAdapter.CodecChunkFrames"/> with <see cref="Qwen3TtsAdapter.CodecContextFrames"/> frames of context).</summary>
    /// <param name="frames">The frames.</param>
    /// <param name="cancellationToken">Stops between chunks.</param>
    public AudioBuffer Decode(IReadOnlyList<int[]> frames, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(frames);
        var samples = new List<float>(frames.Count * _frameSamples);
        for (int start = 0; start < frames.Count; start += Qwen3TtsAdapter.CodecChunkFrames)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int end = Math.Min(frames.Count, start + Qwen3TtsAdapter.CodecChunkFrames);
            samples.AddRange(DecodeSlice(frames, start, end, Math.Min(Qwen3TtsAdapter.CodecContextFrames, start)).Samples.ToArray());
        }

        return new AudioBuffer([.. samples], SampleRate);
    }

    /// <summary>
    /// Decodes frames whose first <paramref name="reference"/> are a reference recording's and drops the reference's share of the audio, as the reference cuts
    /// <c>reference / frames</c> of the decoded samples.
    /// </summary>
    /// <param name="frames">The reference's frames followed by the generated ones.</param>
    /// <param name="reference">The reference's frames at the start.</param>
    /// <param name="cancellationToken">Stops between chunks.</param>
    public AudioBuffer DecodeAfter(IReadOnlyList<int[]> frames, int reference, CancellationToken cancellationToken = default)
    {
        var audio = Decode(frames, cancellationToken);
        if (reference == 0)
        {
            return audio;
        }

        int cut = (int)((double)reference / Math.Max(frames.Count, 1) * audio.Samples.Length);
        return new AudioBuffer(audio.Samples[cut..], SampleRate);
    }

    /// <summary>Decodes frames <c>[first, end)</c> with <paramref name="context"/> frames before them whose audio is dropped.</summary>
    private AudioBuffer DecodeSlice(IReadOnlyList<int[]> frames, int first, int end, int context)
    {
        int start = first - context;
        int count = end - start;
        if (count <= 0)
        {
            return new AudioBuffer([], SampleRate);
        }

        _codec ??= _model.CompileComponent(Qwen3TtsAdapter.CodecRole, AudioCodecGraphs.DecodeGraph);
        var mask = new float[_groups];
        mask.AsSpan().Fill(1f);
        _codec.SetInput<float>(Input(AudioCodecGraphs.MaskInput), mask);
        for (int q = 0; q < _groups; q++)
        {
            var codes = new int[count];
            for (int f = 0; f < count; f++)
            {
                codes[f] = Math.Max(0, frames[start + f][q]);
            }

            _codec.SetInput<int>(Input(AudioCodecGraphs.CodesName(q)), codes);
        }

        var positions = new int[count];
        var spans = new int[2 * count];
        for (int i = 0; i < count; i++)
        {
            positions[i] = i;
            spans[(2 * i) + 1] = count;
        }

        _codec.SetInput<int>(Input(AudioCodecGraphs.PositionsInput), positions);
        _codec.SetInput<int>(Input(AudioCodecGraphs.SpansInput), spans);
        var parameters = new RunParameters();
        parameters.Symbols[WellKnown.TokenSymbol] = count;
        _codec.Run(parameters);
        var samples = new float[count * _frameSamples];
        _codec.GetOutput<float>(AudioCodecGraphs.AudioOutput, samples);
        return new AudioBuffer(samples[(context * _frameSamples)..], SampleRate);
    }

    private Value Input(string name) => _codec!.Graph.Inputs.First(i => i.Input!.Name == name);

    // The row positions and the single-sequence attention spans of the codec encoder's transformer.
    private static void SetPositions(CompiledModel compiled, int rows)
    {
        if (compiled.Graph.Inputs.FirstOrDefault(v => v.Input!.Name == AudioCodecGraphs.PositionsInput) is { } positions)
        {
            compiled.SetInput<int>(positions, [.. Enumerable.Range(0, rows)]);
        }

        if (compiled.Graph.Inputs.FirstOrDefault(v => v.Input!.Name == AudioCodecGraphs.SpansInput) is { } spans)
        {
            var values = new int[2 * rows];
            for (int i = 0; i < rows; i++)
            {
                values[(2 * i) + 1] = rows;
            }

            compiled.SetInput<int>(spans, values);
        }
    }

    private InferenceSession CreateSession(string graph)
    {
        var session = _llm.CreateSession(graph);
        session.CaptureHidden = true;
        return session;
    }

    private static float[] Process(InferenceSession session, float[] rows, int count)
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

    private static void Add(Span<float> target, ReadOnlySpan<float> values)
    {
        for (int i = 0; i < target.Length; i++)
        {
            target[i] += values[i];
        }
    }

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var sum = Vector<float>.Zero;
        int i = 0;
        for (; i <= a.Length - Vector<float>.Count; i += Vector<float>.Count)
        {
            sum += new Vector<float>(a[i..]) * new Vector<float>(b[i..]);
        }

        float total = Vector.Sum(sum);
        for (; i < a.Length; i++)
        {
            total += a[i] * b[i];
        }

        return total;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _talker?.Dispose();
        _predictor?.Dispose();
        _codec?.Dispose();
        _codecEncoder?.Dispose();
        _speakerEncoder?.Dispose();
        _gate.Dispose();
    }

    /// <summary>The host-side tables: the talker's codec embedding, the text embedding (rows read on demand) and projection, the code predictor's embeddings, heads and projection.</summary>
    private sealed class Tables
    {
        private const string Prefix = Qwen3TtsAdapter.Architecture + ".";
        private readonly float[] _codec;
        private readonly WeightInfo _text;
        private readonly int _textWidth;
        private readonly float[] _fc1;
        private readonly float[] _fc1Bias;
        private readonly float[] _fc2;
        private readonly float[] _fc2Bias;
        private readonly float[][] _embeddings;
        private readonly float[][] _heads;
        private readonly float[]? _projection;
        private readonly float[]? _projectionBias;

        public Tables(ModelDefinition definition, int predictorGroups)
        {
            var w = definition.Weights;
            _codec = WeightConversions.ReadFloats(w["token_embedding.weight"]);
            Hidden = (int)w["token_embedding.weight"].Shape[1].Fixed;
            _text = w[Prefix + "text_embedding.weight"];
            _textWidth = (int)_text.Shape[1].Fixed;
            _fc1 = WeightConversions.ReadFloats(w[Prefix + "text_projection.linear_fc1.weight"]);
            _fc1Bias = WeightConversions.ReadFloats(w[Prefix + "text_projection.linear_fc1.bias"]);
            _fc2 = WeightConversions.ReadFloats(w[Prefix + "text_projection.linear_fc2.weight"]);
            _fc2Bias = WeightConversions.ReadFloats(w[Prefix + "text_projection.linear_fc2.bias"]);
            _embeddings = [.. Enumerable.Range(0, predictorGroups).Select(i => WeightConversions.ReadFloats(w[$"{Prefix}code_predictor.embedding.{i}.weight"]))];
            _heads = [.. Enumerable.Range(0, predictorGroups).Select(i => WeightConversions.ReadFloats(w[$"{Prefix}code_predictor.head.{i}.weight"]))];
            PredictorHidden = (int)w[$"{Prefix}code_predictor.head.0.weight"].Shape[1].Fixed;
            if (w.TryGetValue(Prefix + "code_predictor.projection.weight", out var projection))
            {
                _projection = WeightConversions.ReadFloats(projection);
                _projectionBias = WeightConversions.ReadFloats(w[Prefix + "code_predictor.projection.bias"]);
            }
        }

        public int Hidden { get; }

        public int PredictorHidden { get; }

        public ReadOnlySpan<float> Codec(int token) => _codec.AsSpan(token * Hidden, Hidden);

        public ReadOnlySpan<float> PredictorEmbedding(int group, int token) => _embeddings[group].AsSpan(token * Hidden, Hidden);

        // The text projection (SiLU MLP) of a text token's embedding row.
        public float[] TextRow(int token)
        {
            var embedding = new float[_textWidth];
            if (_text.Type == DataType.F32)
            {
                _text.Source.Read((long)token * _textWidth * sizeof(float), MemoryMarshal.AsBytes(embedding.AsSpan()));
            }
            else if (_text.Type == DataType.BF16)
            {
                var raw = new ushort[_textWidth];
                _text.Source.Read((long)token * _textWidth * sizeof(ushort), MemoryMarshal.AsBytes(raw.AsSpan()));
                for (int i = 0; i < raw.Length; i++)
                {
                    embedding[i] = BitConverter.Int32BitsToSingle(raw[i] << 16);
                }
            }
            else
            {
                throw new NotSupportedException($"Qwen3-TTS: text embedding of type {_text.Type} is not supported.");
            }

            var inner = Linear(_fc1, _fc1Bias, embedding, _textWidth);
            for (int i = 0; i < inner.Length; i++)
            {
                inner[i] /= 1f + MathF.Exp(-inner[i]);
            }

            return Linear(_fc2, _fc2Bias, inner, _textWidth);
        }

        public float[] Project(ReadOnlySpan<float> row)
            => _projection is null ? row.ToArray() : Linear(_projection, _projectionBias!, row, Hidden);

        public float[] Head(int group, ReadOnlySpan<float> hidden) => Linear(_heads[group], null, hidden, PredictorHidden);

        private static float[] Linear(float[] weight, float[]? bias, ReadOnlySpan<float> x, int inputs)
        {
            int outputs = weight.Length / inputs;
            var y = new float[outputs];
            for (int o = 0; o < outputs; o++)
            {
                y[o] = Dot(weight.AsSpan(o * inputs, inputs), x) + (bias is null ? 0f : bias[o]);
            }

            return y;
        }
    }
}
