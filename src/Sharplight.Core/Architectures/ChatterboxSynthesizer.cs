using System.Runtime.CompilerServices;
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
/// The <see cref="ISpeechSynthesizer"/> of Chatterbox (<see cref="ChatterboxAdapter"/>). Per request: the text is normalized like the reference
/// (<c>punc_norm</c>) and tokenized (spaces as <c>[SPACE]</c>, start and stop tokens); T3 runs twice side by side for classifier-free guidance, on the
/// conditioning rows, the text rows (zero text embeddings for the unconditional run) and the start-of-speech row, then token by token (speech embedding
/// plus its learned position); the guided logits <c>cond + w · (cond − uncond)</c> go through the repetition penalty and the sampler; the speech
/// tokens after the voice's flow prompt are encoded to <c>mu</c>, ten Euler steps of the guided flow-matching field turn noise into the mel (the
/// prompt's mel as condition), and HiFT turns the mel into 24 kHz audio with a 20 ms fade-in. One request runs at a time.
/// </summary>
internal sealed class ChatterboxSynthesizer : ISpeechSynthesizer
{
    /// <summary>The emotion exaggeration of the reference's default.</summary>
    public const float DefaultExaggeration = 0.5f;

    /// <summary>T3's classifier-free guidance weight.</summary>
    public const float GuidanceWeight = 0.5f;

    private const string SpaceToken = "[SPACE]";

    private readonly TaskModel _model;
    private readonly LlmModel _llm;
    private readonly ITokenizer _tokenizer;
    private readonly int _startText;
    private readonly int _stopText;
    private readonly int _startSpeech;
    private readonly int _stopSpeech;
    private readonly int _speechVocabulary;
    private readonly int _hidden;
    private readonly int _flowSteps;
    private readonly float _flowGuidance;
    private readonly int _samplesPerFrame;
    private readonly int _fftSize;
    private readonly int _hop;
    private readonly int _harmonics;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly Lazy<Tables> _tables;
    private InferenceSession? _conditional;
    private InferenceSession? _unconditional;
    private CompiledModel? _condition;
    private CompiledModel? _flowEncoder;
    private CompiledModel? _estimator;
    private CompiledModel? _vocoder;
    private bool _disposed;

    /// <summary>Creates the synthesizer; nothing is compiled until the first request.</summary>
    /// <param name="model">A loaded Chatterbox model.</param>
    public ChatterboxSynthesizer(TaskModel model)
    {
        _model = model;
        _llm = LlmModel.FromTaskModel(model);
        _tokenizer = _llm.Tokenizer;
        var text = model.Description.Component(ChatterboxAdapter.TextModelRole);
        _startText = text.ConfigValue("startText", 255);
        _stopText = text.ConfigValue("stopText", 0);
        _startSpeech = text.ConfigValue("startSpeech", 6561);
        _stopSpeech = text.ConfigValue("stopSpeech", 6562);
        _speechVocabulary = text.ConfigValue("speechVocabulary", 6561);
        _hidden = model.Definition.Property(WellKnown.EmbeddingLength, 1024);
        var estimator = model.Description.Component(ChatterboxAdapter.EstimatorRole);
        _flowSteps = estimator.ConfigValue("steps", 10);
        _flowGuidance = (float)estimator.ConfigValue("guidance", 0.7);
        var vocoder = model.Description.Component(ChatterboxAdapter.VocoderRole);
        _samplesPerFrame = vocoder.ConfigValue("samplesPerFrame", 480);
        _fftSize = vocoder.ConfigValue("fftSize", 16);
        _hop = vocoder.ConfigValue("hop", 4);
        _harmonics = vocoder.ConfigValue("harmonics", 9);
        SampleRate = model.Description.ConfigValue("sampleRate", 24000);
        var condition = model.Description.Component(ChatterboxAdapter.ConditionRole);
        Voices = [.. condition.Attachments.Keys
            .Where(k => k.EndsWith("/speaker.f32", StringComparison.Ordinal))
            .Select(k => k[ChatterboxAdapter.VoicePrefix.Length..^"/speaker.f32".Length])
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

    /// <summary>Replaces the source of standard-normal noise (the flow's starting noise and HiFT's source noise; <c>count</c> values, <c>stream</c> 0 or 1).</summary>
    public Func<int, int, float[]>? NoiseSource { get; set; }

    /// <summary>Replaces the start phases of HiFT's harmonics (<c>count</c> values in [−π, π), the first zero).</summary>
    public Func<int, float[]>? PhaseSource { get; set; }

    /// <summary>Whether <paramref name="model"/> is a Chatterbox model.</summary>
    /// <param name="model">A loaded speech-synthesis model.</param>
    public static bool Handles(TaskModel model)
        => model.Description.TryGetComponent(ChatterboxAdapter.TextModelRole, out var text) && text.ConfigValue("family", "") == ChatterboxAdapter.Family
           && model.Description.TryGetComponent(ChatterboxAdapter.VocoderRole, out _);

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
        var audio = await BackgroundCompute.RunAsync(_gate, () => Synthesize(text, options, cancellationToken), cancellationToken).ConfigureAwait(false);
        return new SpeechSynthesisResult(audio, options?.Voice ?? DefaultVoice);
    }

    /// <inheritdoc/>
    /// <remarks>One buffer per sentence; generation runs on a background task (<see cref="BackgroundCompute"/>).</remarks>
    public IAsyncEnumerable<AudioBuffer> SynthesizeStreamingAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        return BackgroundCompute.StreamAsync(token => StreamCoreAsync(text, options, token), cancellationToken);
    }

    /// <summary>The text tokens of a request: the normalized text with spaces as <c>[SPACE]</c>, between the start and stop tokens.</summary>
    /// <param name="text">The text.</param>
    public int[] TextTokens(string text)
    {
        string normalized = NormalizePunctuation(text).Replace(" ", SpaceToken, StringComparison.Ordinal);
        return [_startText, .. _tokenizer.Encode(normalized, addSpecial: false, parseSpecial: true), _stopText];
    }

    /// <summary>The conditioning rows of a voice [rows, hidden] (compiles the conditioning graph on first use).</summary>
    /// <param name="voice">The voice name.</param>
    /// <param name="exaggeration">The emotion exaggeration (0.5 by default).</param>
    public float[] ConditionRows(string voice, float exaggeration)
    {
        var compiled = _condition ??= _model.CompileComponent(ChatterboxAdapter.ConditionRole);
        compiled.SetInput<float>(Input(compiled, ChatterboxConditionGraph.SpeakerInput), Floats(voice, "speaker.f32"));
        compiled.SetInput<int>(Input(compiled, ChatterboxConditionGraph.PromptInput), Ints(voice, "prompt_tokens.i32"));
        compiled.SetInput<float>(Input(compiled, ChatterboxConditionGraph.EmotionInput), [exaggeration]);
        compiled.Run(new RunParameters());
        var output = compiled.Graph.Outputs[ChatterboxConditionGraph.RowsOutput];
        var rows = new float[output.Shape[0].Fixed * output.Shape[1].Fixed];
        compiled.GetOutput<float>(ChatterboxConditionGraph.RowsOutput, rows);
        return rows;
    }

    /// <summary>
    /// T3's speech tokens for the text tokens (up to and excluding the stop token), sampled with <paramref name="sampling"/> after the guidance and
    /// the repetition penalty; a temperature of zero is greedy.
    /// </summary>
    /// <remarks><paramref name="observe"/>, when given, receives every step's best guided token and its margin over the second best.</remarks>
    /// <param name="textTokens">The text tokens with start and stop.</param>
    /// <param name="condition">The conditioning rows.</param>
    /// <param name="sampling">The sampling settings (repetition penalty included).</param>
    /// <param name="cancellationToken">Stops generation between tokens.</param>
    /// <param name="observe">Receives each step's best token and margin, or <see langword="null"/>.</param>
    /// <param name="forced">Feeds these tokens instead of the chosen ones (teacher forcing), or <see langword="null"/>.</param>
    public int[] SpeechTokens(int[] textTokens, float[] condition, SamplingOptions sampling, CancellationToken cancellationToken, Action<int, float>? observe = null,
        IReadOnlyList<int>? forced = null)
    {
        var tables = _tables.Value;
        var conditional = _conditional ??= _llm.CreateSession(WellKnown.DecodeMultimodalGraph);
        var unconditional = _unconditional ??= _llm.CreateSession(WellKnown.DecodeMultimodalGraph);
        conditional.Reset();
        unconditional.Reset();
        int conditionRows = condition.Length / _hidden;
        int rows = conditionRows + textTokens.Length + 2;
        var withText = new float[rows * _hidden];
        var withoutText = new float[rows * _hidden];
        condition.CopyTo(withText, 0);
        condition.CopyTo(withoutText, 0);
        for (int i = 0; i < textTokens.Length; i++)
        {
            var target = withText.AsSpan((conditionRows + i) * _hidden, _hidden);
            var bare = withoutText.AsSpan((conditionRows + i) * _hidden, _hidden);
            tables.TextPosition(i).CopyTo(target);
            tables.TextPosition(i).CopyTo(bare);
            Add(target, tables.TextEmbedding(textTokens[i]));
        }

        var start = SpeechRow(tables, _startSpeech, 0);
        for (int r = rows - 2; r < rows; r++)
        {
            start.CopyTo(withText, r * _hidden);
            start.CopyTo(withoutText, r * _hidden);
        }

        var cond = Process(conditional, withText, rows);
        var uncond = Process(unconditional, withoutText, rows);
        var sampler = new Sampler(sampling with { RepetitionPenalty = 1f });
        var history = new List<int> { _startSpeech };
        var tokens = new List<int>();
        int budget = Math.Min(ChatterboxAdapter.MaxSpeechTokens, conditional.ContextLength - rows - 1);
        var guided = new float[cond.Length];
        for (int i = 0; i < budget; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int v = 0; v < guided.Length; v++)
            {
                guided[v] = cond[v] + (GuidanceWeight * (cond[v] - uncond[v]));
            }

            ApplyRepetitionPenalty(guided, history, sampling.RepetitionPenalty);
            observe?.Invoke(Sampler.ArgMax(guided), Margin(guided));
            int token = sampling.Temperature <= 0f ? Sampler.ArgMax(guided) : sampler.Sample(guided);
            if (forced is not null)
            {
                if (i >= forced.Count)
                {
                    break;
                }

                token = forced[i];
            }

            history.Add(token);
            if (token == _stopSpeech)
            {
                break;
            }

            tokens.Add(token);
            var row = SpeechRow(tables, token, i + 1);
            cond = Process(conditional, row, 1);
            uncond = Process(unconditional, row, 1);
        }

        return [.. tokens];
    }

    /// <summary>The flow encoder's <c>mu</c> [2 · tokens, 80] for the voice's flow prompt followed by the speech tokens, and the speaker condition [80].</summary>
    /// <param name="speech">The speech tokens (tokens outside the codebook are dropped).</param>
    /// <param name="voice">The voice name.</param>
    public (float[] Mu, float[] Speaker) EncodeFlow(int[] speech, string voice)
    {
        int[] tokens = [.. Ints(voice, "flow_tokens.i32"), .. speech.Where(t => t < _speechVocabulary)];
        if (tokens.Length > ChatterboxAdapter.MaxFlowTokens)
        {
            throw new ArgumentException($"At most {ChatterboxAdapter.MaxFlowTokens} flow tokens are decoded at once, got {tokens.Length}.", nameof(speech));
        }

        var encoder = _flowEncoder ??= _model.CompileComponent(ChatterboxAdapter.FlowEncoderRole);
        encoder.SetInput<int>(Input(encoder, ChatterboxFlowEncoderGraph.TokensInput), tokens);
        encoder.SetInput<int>(Input(encoder, ChatterboxFlowEncoderGraph.DistancesInput), Distances(tokens.Length));
        encoder.SetInput<int>(Input(encoder, ChatterboxFlowEncoderGraph.UpDistancesInput), Distances(2 * tokens.Length));
        encoder.SetInput<float>(Input(encoder, ChatterboxFlowEncoderGraph.SpeakerInput), Floats(voice, "flow_speaker.f32"));
        var run = new RunParameters();
        run.Symbols[WellKnown.TokenSymbol] = tokens.Length;
        encoder.Run(run);
        var mu = new float[2 * tokens.Length * 80];
        var speaker = new float[80];
        encoder.GetOutput<float>(ChatterboxFlowEncoderGraph.MuOutput, mu);
        encoder.GetOutput<float>(ChatterboxFlowEncoderGraph.SpeakerOutput, speaker);
        return (mu, speaker);
    }

    /// <summary>The voice's prompt mel [frames, 80] (the flow's condition before the generated frames).</summary>
    /// <param name="voice">The voice name.</param>
    public float[] PromptMel(string voice) => Floats(voice, "flow_mel.f32");

    /// <summary>One evaluation of the flow-matching field [frames, 80] (all inputs time-major).</summary>
    /// <param name="x">The current sample.</param>
    /// <param name="mu">The encoder's mean (zero for the unconditional evaluation).</param>
    /// <param name="speaker">The speaker condition (zero for the unconditional evaluation).</param>
    /// <param name="cond">The prompt mel, zero after the prompt (zero for the unconditional evaluation).</param>
    /// <param name="t">The time step.</param>
    public float[] Velocity(float[] x, float[] mu, float[] speaker, float[] cond, float t)
    {
        var estimator = _estimator ??= _model.CompileComponent(ChatterboxAdapter.EstimatorRole);
        var output = new float[x.Length];
        Estimate(estimator, x, mu, speaker, cond, t, x.Length / 80, output);
        return output;
    }

    /// <summary>The mel spectrogram [frames, 80] of speech tokens, starting from the given noise (or the noise source's) [prompt + frames, 80].</summary>
    /// <param name="speech">The speech tokens (tokens outside the codebook are dropped).</param>
    /// <param name="voice">The voice name.</param>
    /// <param name="noise">The starting noise, time-major, or <see langword="null"/> for the noise source.</param>
    public float[] Mel(int[] speech, string voice, float[]? noise = null)
    {
        var (mu, speaker) = EncodeFlow(speech, voice);
        var promptMel = Floats(voice, "flow_mel.f32");
        int frames = mu.Length / 80;
        int promptFrames = promptMel.Length / 80;
        var cond = new float[frames * 80];
        promptMel.AsSpan(0, Math.Min(promptMel.Length, cond.Length)).CopyTo(cond);
        var x = noise ?? (NoiseSource ?? ((n, s) => SpeechNoise.Gaussian(n, s)))(frames * 80, 0);
        if (x.Length != frames * 80)
        {
            throw new ArgumentException($"The flow noise has {x.Length} values, {frames * 80} are needed.", nameof(noise));
        }

        x = (float[])x.Clone();
        var zeros = new float[frames * 80];
        var noSpeaker = new float[80];
        var steps = new double[_flowSteps + 1];
        for (int i = 0; i <= _flowSteps; i++)
        {
            steps[i] = 1 - Math.Cos((double)i / _flowSteps * 0.5 * Math.PI);
        }

        for (int step = 0; step < _flowSteps; step++)
        {
            float t = (float)steps[step];
            float dt = (float)(steps[step + 1] - steps[step]);
            var conditioned = Velocity(x, mu, speaker, cond, t);
            var unconditioned = Velocity(x, zeros, noSpeaker, zeros, t);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] += dt * (((1f + _flowGuidance) * conditioned[i]) - (_flowGuidance * unconditioned[i]));
            }
        }

        return x[(promptFrames * 80)..];
    }

    /// <summary>
    /// HiFT over a mel spectrogram [frames, 80]: the waveform (with the 20 ms fade-in) and the predicted F0 per frame. The harmonic start phases and the
    /// source noise [samples, harmonics] come from the arguments or the sources.
    /// </summary>
    /// <param name="mel">The mel frames.</param>
    /// <param name="phase">The start phases, or <see langword="null"/>.</param>
    /// <param name="noise">The source noise, or <see langword="null"/>.</param>
    public (float[] Samples, float[] F0) Vocode(float[] mel, float[]? phase = null, float[]? noise = null)
    {
        int frames = mel.Length / 80;
        int samples = frames * _samplesPerFrame;
        var vocoder = _vocoder ??= _model.CompileComponent(ChatterboxAdapter.VocoderRole);
        phase ??= (PhaseSource ?? DefaultPhases)(_harmonics);
        noise ??= (NoiseSource ?? ((n, s) => SpeechNoise.Gaussian(n, s)))(samples * _harmonics, 1);
        vocoder.SetInput<float>(Input(vocoder, ChatterboxVocoderGraph.MelInput), mel);
        vocoder.SetInput<float>(Input(vocoder, ChatterboxVocoderGraph.PhaseInput), phase);
        vocoder.SetInput<float>(Input(vocoder, ChatterboxVocoderGraph.NoiseInput), noise);
        var run = new RunParameters();
        run.Symbols[WellKnown.TokenSymbol] = frames;
        vocoder.Run(run);
        int stftFrames = (samples / _hop) + 1;
        var spectrum = new float[stftFrames * _fftSize];
        vocoder.GetOutput<float>(ChatterboxVocoderGraph.FramesOutput, spectrum);
        var f0 = new float[frames];
        vocoder.GetOutput<float>(ChatterboxVocoderGraph.F0Output, f0);
        var audio = OverlapAdd.Synthesize(spectrum, stftFrames, _fftSize, _hop, Window.Hann(_fftSize), _fftSize / 2);
        if (audio.Length > samples)
        {
            audio = audio[..samples];
        }

        int fade = SampleRate / 50;
        for (int i = 0; i < Math.Min(2 * fade, audio.Length); i++)
        {
            audio[i] *= i < fade ? 0f : (float)((Math.Cos(Math.PI - (Math.PI * (i - fade) / (fade - 1))) + 1) / 2);
        }

        return (audio, f0);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _conditional?.Dispose();
        _unconditional?.Dispose();
        _condition?.Dispose();
        _flowEncoder?.Dispose();
        _estimator?.Dispose();
        _vocoder?.Dispose();
        _gate.Dispose();
    }

    /// <summary>The reference's <c>punc_norm</c>: capitalize, collapse spaces, replace rare punctuation, end with a sentence ender.</summary>
    /// <param name="text">The text.</param>
    public static string NormalizePunctuation(string text)
    {
        if (text.Length == 0)
        {
            return "You need to add some text for me to talk.";
        }

        if (char.IsLower(text[0]))
        {
            text = char.ToUpperInvariant(text[0]) + text[1..];
        }

        text = string.Join(' ', text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
        (string From, string To)[] replacements =
        [
            ("...", ", "), ("…", ", "), (":", ","), (" - ", ", "), (";", ", "), ("—", "-"), ("–", "-"), (" ,", ","), ("“", "\""), ("”", "\""), ("‘", "'"), ("’", "'"),
        ];
        foreach (var (from, to) in replacements)
        {
            text = text.Replace(from, to, StringComparison.Ordinal);
        }

        text = text.TrimEnd(' ');
        if (!(text.EndsWith('.') || text.EndsWith('!') || text.EndsWith('?') || text.EndsWith('-') || text.EndsWith(',')))
        {
            text += ".";
        }

        return text;
    }

    private string DefaultVoice => Voices.Contains("default") ? "default" : Voices.Count > 0 ? Voices[0] : throw new InvalidOperationException("The model has no voices.");

    private AudioBuffer Synthesize(string text, SpeechSynthesisOptions? options, CancellationToken cancellationToken)
    {
        string voice = options?.Voice ?? DefaultVoice;
        var sampling = options?.Sampling is { } requested && !ReferenceEquals(options, SpeechSynthesisOptions.Default) && requested != new SamplingOptions()
            ? requested
            : new SamplingOptions { Temperature = 0.8f, TopK = 0, TopP = 1f, MinP = 0.05f, RepetitionPenalty = 1.2f };
        var condition = ConditionRows(voice, DefaultExaggeration);
        var speech = SpeechTokens(TextTokens(text), condition, sampling, cancellationToken);
        if (speech.Length == 0)
        {
            return new AudioBuffer([], SampleRate);
        }

        var mel = Mel(speech, voice);
        return new AudioBuffer(Vocode(mel).Samples, SampleRate);
    }

    private async IAsyncEnumerable<AudioBuffer> StreamCoreAsync(string text, SpeechSynthesisOptions? options, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        foreach (string sentence in Sentences(text))
        {
            await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                yield return await Task.Run(() => Synthesize(sentence, options, cancellationToken), cancellationToken).ConfigureAwait(false);
            }
            finally
            {
                _gate.Release();
            }
        }
    }

    private static IEnumerable<string> Sentences(string text)
    {
        int start = 0;
        for (int i = 0; i < text.Length; i++)
        {
            if (text[i] is '.' or '!' or '?' && (i + 1 == text.Length || char.IsWhiteSpace(text[i + 1])))
            {
                string sentence = text[start..(i + 1)].Trim();
                if (sentence.Length > 0)
                {
                    yield return sentence;
                }

                start = i + 1;
            }
        }

        if (start < text.Length && text[start..].Trim() is { Length: > 0 } rest)
        {
            yield return rest;
        }
    }

    private static float[] DefaultPhases(int count)
    {
        var random = new Random(0);
        var phases = new float[count];
        for (int i = 1; i < count; i++)
        {
            phases[i] = (float)((random.NextDouble() * 2 * Math.PI) - Math.PI);
        }

        return phases;
    }

    private void Estimate(CompiledModel estimator, float[] x, float[] mu, float[] speaker, float[] cond, float t, int frames, float[] output)
    {
        estimator.SetInput<float>(Input(estimator, ChatterboxEstimatorGraph.SampleInput), x);
        estimator.SetInput<float>(Input(estimator, ChatterboxEstimatorGraph.MuInput), mu);
        estimator.SetInput<float>(Input(estimator, ChatterboxEstimatorGraph.SpeakerInput), speaker);
        estimator.SetInput<float>(Input(estimator, ChatterboxEstimatorGraph.CondInput), cond);
        estimator.SetInput<float>(Input(estimator, ChatterboxEstimatorGraph.TimeInput), [t]);
        var run = new RunParameters();
        run.Symbols[ChatterboxEstimatorGraph.FrameSymbol] = frames;
        estimator.Run(run);
        estimator.GetOutput<float>(ChatterboxEstimatorGraph.VelocityOutput, output);
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

    private float[] SpeechRow(Tables tables, int token, int position)
    {
        var row = new float[_hidden];
        tables.SpeechEmbedding(token).CopyTo(row);
        Add(row, tables.SpeechPosition(position));
        return row;
    }

    private static void Add(Span<float> target, ReadOnlySpan<float> values)
    {
        for (int i = 0; i < target.Length; i++)
        {
            target[i] += values[i];
        }
    }

    private static void ApplyRepetitionPenalty(float[] logits, List<int> history, float penalty)
    {
        if (penalty == 1f)
        {
            return;
        }

        foreach (int token in history.Distinct())
        {
            float value = logits[token];
            logits[token] = value < 0 ? value * penalty : value / penalty;
        }
    }

    private static float Margin(float[] logits)
    {
        float best = float.NegativeInfinity;
        float second = float.NegativeInfinity;
        foreach (float value in logits)
        {
            if (value > best)
            {
                second = best;
                best = value;
            }
            else if (value > second)
            {
                second = value;
            }
        }

        return best - second;
    }

    private static int[] Distances(int rows) => [.. Enumerable.Range(0, (2 * rows) - 1).Select(i => rows - 1 - i)];

    private static Value Input(CompiledModel compiled, string name) => compiled.Graph.Inputs.First(v => v.Input!.Name == name);

    private float[] Floats(string voice, string part)
        => MemoryMarshal.Cast<byte, float>(_model.ReadAttachment(ChatterboxAdapter.ConditionRole, $"{ChatterboxAdapter.VoicePrefix}{voice}/{part}")).ToArray();

    private int[] Ints(string voice, string part)
        => MemoryMarshal.Cast<byte, int>(_model.ReadAttachment(ChatterboxAdapter.ConditionRole, $"{ChatterboxAdapter.VoicePrefix}{voice}/{part}")).ToArray();

    /// <summary>T3's embedding tables, read once on the host (the rows are looked up per token).</summary>
    private sealed class Tables(ModelDefinition definition)
    {
        private readonly float[] _text = WeightConversions.ReadFloats(definition.Weights["t3.text_emb.weight"]);
        private readonly float[] _textPositions = WeightConversions.ReadFloats(definition.Weights["t3.text_pos_emb.emb.weight"]);
        private readonly float[] _speech = WeightConversions.ReadFloats(definition.Weights["token_embedding.weight"]);
        private readonly float[] _speechPositions = WeightConversions.ReadFloats(definition.Weights["t3.speech_pos_emb.emb.weight"]);
        private readonly int _width = (int)definition.Weights["token_embedding.weight"].Shape[1].Fixed;

        public ReadOnlySpan<float> TextEmbedding(int token) => _text.AsSpan(token * _width, _width);

        public ReadOnlySpan<float> TextPosition(int position) => _textPositions.AsSpan(position * _width, _width);

        public ReadOnlySpan<float> SpeechEmbedding(int token) => _speech.AsSpan(token * _width, _width);

        public ReadOnlySpan<float> SpeechPosition(int position) => _speechPositions.AsSpan(position * _width, _width);
    }
}
