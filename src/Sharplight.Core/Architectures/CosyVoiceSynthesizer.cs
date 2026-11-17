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
/// The <see cref="ISpeechSynthesizer"/> of CosyVoice 3 (<see cref="CosyVoiceAdapter"/>), following <c>CosyVoice3.inference_zero_shot</c>,
/// <c>inference_cross_lingual</c> and <c>inference_instruct2</c>. The voice is taken from <see cref="SpeechSynthesisOptions.ReferenceAudio"/> (its speech
/// tokens, prompt mel and x-vector); with <see cref="SpeechSynthesisOptions.ReferenceText"/> the language model continues the recording (zero-shot:
/// the system prompt and the reference text before the text, the recording's speech tokens after the task row), without it the recording only sets
/// the flow's voice (cross-lingual), and <see cref="SpeechSynthesisOptions.Instructions"/> replaces the reference text by an instruction (instruct).
/// Per utterance of the text front end: the language model's speech tokens by repetition-aware sampling; the flow's mel after the recording's prompt
/// (ten Euler steps of the guided DiT from noise); the causal HiFT's waveform. One request runs at a time.
/// </summary>
internal sealed class CosyVoiceSynthesizer : ISpeechSynthesizer
{
    /// <summary>The system prompt that CosyVoice 3 expects before <c>&lt;|endofprompt|&gt;</c>.</summary>
    public const string SystemPrompt = "You are a helpful assistant.";

    /// <summary>The end-of-prompt marker.</summary>
    public const string EndOfPrompt = "<|endofprompt|>";

    private const float AudioLimit = 0.99f;

    private readonly TaskModel _model;
    private readonly LlmModel _llm;
    private readonly ITokenizer _tokenizer;
    private readonly int _speechTokens;
    private readonly int _sos;
    private readonly int _taskId;
    private readonly float _topP;
    private readonly int _topK;
    private readonly int _window;
    private readonly float _tauR;
    private readonly double _minRatio;
    private readonly double _maxRatio;
    private readonly int _hidden;
    private readonly int _tokenMelRatio;
    private readonly int _mels;
    private readonly int _flowSteps;
    private readonly float _flowGuidance;
    private readonly int _samplesPerFrame;
    private readonly int _fftSize;
    private readonly int _hop;
    private readonly int _harmonics;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly Lazy<Tables> _tables;
    private InferenceSession? _session;
    private CompiledModel? _flowEncoder;
    private CompiledModel? _estimator;
    private CompiledModel? _vocoder;
    private CompiledModel? _f0Predictor;
    private S3SpeechEncoders? _encoders;
    private (AudioBuffer Recording, CosyVoiceVoice Voice)? _lastClone;
    private bool _disposed;

    /// <summary>Creates the synthesizer; nothing is compiled until the first request.</summary>
    /// <param name="model">A loaded CosyVoice 3 model.</param>
    public CosyVoiceSynthesizer(TaskModel model)
    {
        _model = model;
        _llm = LlmModel.FromTaskModel(model);
        _tokenizer = _llm.Tokenizer;
        var text = model.Description.Component(CosyVoiceAdapter.TextModelRole);
        _speechTokens = text.ConfigValue("speechTokens", 6561);
        _sos = text.ConfigValue("sos", 6561);
        _taskId = text.ConfigValue("taskId", 6563);
        _topP = (float)text.ConfigValue("topP", 0.8);
        _topK = text.ConfigValue("topK", 25);
        _window = text.ConfigValue("window", 10);
        _tauR = (float)text.ConfigValue("tauR", 0.1);
        _minRatio = text.ConfigValue("minTokenTextRatio", 2.0);
        _maxRatio = text.ConfigValue("maxTokenTextRatio", 20.0);
        _hidden = model.Definition.Property(WellKnown.EmbeddingLength, 896);
        var encoder = model.Description.Component(CosyVoiceAdapter.FlowEncoderRole);
        _tokenMelRatio = encoder.ConfigValue("tokenMelRatio", 2);
        _mels = encoder.ConfigValue("melBins", 80);
        var estimator = model.Description.Component(CosyVoiceAdapter.EstimatorRole);
        _flowSteps = estimator.ConfigValue("steps", 10);
        _flowGuidance = (float)estimator.ConfigValue("guidance", 0.7);
        var vocoder = model.Description.Component(CosyVoiceAdapter.VocoderRole);
        _samplesPerFrame = vocoder.ConfigValue("samplesPerFrame", 480);
        _fftSize = vocoder.ConfigValue("fftSize", 16);
        _hop = vocoder.ConfigValue("hop", 4);
        _harmonics = vocoder.ConfigValue("harmonics", 9);
        SampleRate = model.Description.ConfigValue("sampleRate", 24000);
        _tables = new Lazy<Tables>(() => new Tables(model.Definition));
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    /// <inheritdoc/>
    public TaskClientMetadata Metadata { get; }

    /// <inheritdoc/>
    public int SampleRate { get; }

    /// <inheritdoc/>
    /// <remarks>The release has no built-in voices: every request takes its voice from <see cref="SpeechSynthesisOptions.ReferenceAudio"/>.</remarks>
    public IReadOnlyList<string> Voices => [];

    /// <summary>Replaces the flow's standard-normal starting noise (<c>count</c> values, time-major).</summary>
    public Func<int, float[]>? FlowNoiseSource { get; set; }

    /// <summary>Replaces the vocoder's uniform source noise (<c>count</c> values in [0, 1), [samples, harmonics]).</summary>
    public Func<int, float[]>? SourceNoiseSource { get; set; }

    /// <summary>Whether <paramref name="model"/> is a CosyVoice 3 model.</summary>
    /// <param name="model">A loaded speech-synthesis model.</param>
    public static bool Handles(TaskModel model)
        => model.Description.TryGetComponent(CosyVoiceAdapter.TextModelRole, out var text) && text.ConfigValue("family", "") == CosyVoiceAdapter.Family
           && model.Description.TryGetComponent(CosyVoiceAdapter.VocoderRole, out _);

    /// <summary>Whether the model can take a voice from a recording (it was imported with the S3 tokenizer and CAM++).</summary>
    public bool SupportsCloning => S3SpeechEncoders.Supports(_model);

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
        return new SpeechSynthesisResult(audio, "reference");
    }

    /// <inheritdoc/>
    /// <remarks>One buffer per utterance of the text front end; generation runs on a background task (<see cref="BackgroundCompute"/>).</remarks>
    public IAsyncEnumerable<AudioBuffer> SynthesizeStreamingAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        return BackgroundCompute.StreamAsync(token => StreamCoreAsync(text, options, token), cancellationToken);
    }

    /// <summary>The tokens of a piece of text (special tokens such as <c>&lt;|endofprompt|&gt;</c> and <c>[breath]</c> recognized).</summary>
    /// <param name="text">The text.</param>
    public int[] TextTokens(string text) => _tokenizer.Encode(text, addSpecial: false, parseSpecial: true);

    /// <summary>The voice of a recording (compiles the encoders on first use), following <c>frontend_zero_shot</c>.</summary>
    /// <param name="recording">The recording (at most 30 s; any rate, channels mixed down).</param>
    /// <param name="cancellationToken">Stops between stages.</param>
    /// <exception cref="NotSupportedException">The model was imported without the S3 tokenizer or CAM++.</exception>
    public CosyVoiceVoice CloneVoice(AudioBuffer recording, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(recording);
        if (!SupportsCloning)
        {
            throw new NotSupportedException("This CosyVoice model was imported without speech_tokenizer_v3.onnx and campplus.onnx, so it cannot take a voice from a recording.");
        }

        var encoders = _encoders ??= new S3SpeechEncoders(_model);
        var mono = recording.ToMono().Samples;
        var ref16 = AudioResampler.Resample(mono, 1, recording.SampleRate, 16000);
        if (ref16.Length > CosyVoiceAdapter.MaxCloneSeconds * 16000)
        {
            throw new ArgumentException($"The reference recording is longer than {CosyVoiceAdapter.MaxCloneSeconds} s.", nameof(recording));
        }

        var ref24 = AudioResampler.Resample(mono, 1, recording.SampleRate, SampleRate);
        cancellationToken.ThrowIfCancellationRequested();
        var mel = encoders.PromptMel(ref24, out int frames);
        cancellationToken.ThrowIfCancellationRequested();
        var tokens = encoders.SpeechTokens(ref16, maxTokens: null);
        cancellationToken.ThrowIfCancellationRequested();
        var xvector = encoders.XVector(ref16);
        return Voice(tokens, mel, frames, xvector);
    }

    /// <summary>A voice from its parts, cut so that the mel has exactly <see cref="CosyVoiceAdapter"/>'s token-to-mel ratio times the tokens.</summary>
    /// <param name="tokens">The speech tokens.</param>
    /// <param name="mel">The prompt mel [frames, 80].</param>
    /// <param name="frames">The prompt mel frames.</param>
    /// <param name="xvector">The x-vector.</param>
    public CosyVoiceVoice Voice(int[] tokens, float[] mel, int frames, float[] xvector)
    {
        int count = Math.Min(frames / _tokenMelRatio, tokens.Length);
        return new CosyVoiceVoice(tokens[..count], mel[..(count * _tokenMelRatio * _mels)], xvector);
    }

    /// <summary>
    /// The language model's speech tokens (up to and excluding the stop token). The rows are the start row, the prompt and text tokens' embeddings,
    /// the task row and the prompt speech tokens' embeddings; every generated token's speech embedding follows. A temperature of zero is greedy.
    /// </summary>
    /// <param name="textTokens">The text tokens (their count sets the minimum and maximum length).</param>
    /// <param name="promptTextTokens">The prompt text tokens (system prompt, reference text or instruction, end of prompt), or empty.</param>
    /// <param name="promptSpeech">The prompt speech tokens (zero-shot), or empty.</param>
    /// <param name="sampling">The sampling settings.</param>
    /// <param name="cancellationToken">Stops generation between tokens.</param>
    /// <param name="observe">Receives each step's logits (before sampling), or <see langword="null"/>.</param>
    /// <param name="forced">Feeds these tokens instead of the chosen ones (teacher forcing), or <see langword="null"/>.</param>
    public int[] SpeechTokens(int[] textTokens, int[] promptTextTokens, int[] promptSpeech, SamplingOptions sampling, CancellationToken cancellationToken,
        Action<float[]>? observe = null, IReadOnlyList<int>? forced = null)
    {
        var tables = _tables.Value;
        var session = _session ??= _llm.CreateSession(WellKnown.DecodeMultimodalGraph);
        session.Reset();
        int[] allText = [.. promptTextTokens, .. textTokens];
        int rows = allText.Length + 2 + promptSpeech.Length;
        var input = new float[rows * _hidden];
        tables.SpeechEmbedding(_sos).CopyTo(input);
        for (int i = 0; i < allText.Length; i++)
        {
            tables.TextEmbedding(allText[i], input.AsSpan((i + 1) * _hidden, _hidden));
        }

        tables.SpeechEmbedding(_taskId).CopyTo(input.AsSpan((allText.Length + 1) * _hidden));
        for (int i = 0; i < promptSpeech.Length; i++)
        {
            tables.SpeechEmbedding(promptSpeech[i]).CopyTo(input.AsSpan((allText.Length + 2 + i) * _hidden));
        }

        int minLength = (int)(textTokens.Length * _minRatio);
        int maxLength = Math.Min((int)(textTokens.Length * _maxRatio), Math.Min(CosyVoiceAdapter.MaxSpeechTokens, session.ContextLength - rows - 1));
        var sampler = new CosyVoiceRasSampler(sampling.TopP, sampling.TopK, _window, _tauR, sampling.Temperature, sampling.Seed);
        var logits = Process(session, input, rows);
        var tokens = new List<int>();
        for (int i = 0; i < maxLength; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (forced is not null && i >= forced.Count)
            {
                break;
            }

            observe?.Invoke(logits);
            if (i < minLength)
            {
                // As the reference (sampling_ids with ignore_eos): only the first control row is masked.
                logits[_speechTokens] = float.NegativeInfinity;
            }

            int token = sampling.Temperature <= 0f ? Sampler.ArgMax(logits) : sampler.Sample(logits, tokens);
            if (forced is not null)
            {
                token = forced[i];
            }

            if (token >= _speechTokens)
            {
                break;
            }

            tokens.Add(token);
            logits = Process(session, tables.SpeechEmbedding(token).ToArray(), 1);
        }

        return [.. tokens];
    }

    /// <summary>The flow encoder's <c>mu</c> [2 · tokens, 80] for the voice's prompt tokens followed by the speech tokens, and the speaker condition [80].</summary>
    /// <param name="speech">The speech tokens.</param>
    /// <param name="voice">The voice.</param>
    public (float[] Mu, float[] Speaker) EncodeFlow(int[] speech, CosyVoiceVoice voice)
    {
        ArgumentNullException.ThrowIfNull(voice);
        int[] tokens = [.. voice.SpeechTokens, .. speech];
        if (tokens.Length > CosyVoiceAdapter.MaxFlowTokens)
        {
            throw new ArgumentException($"At most {CosyVoiceAdapter.MaxFlowTokens} flow tokens are decoded at once, got {tokens.Length}.", nameof(speech));
        }

        var encoder = _flowEncoder ??= _model.CompileComponent(CosyVoiceAdapter.FlowEncoderRole);
        encoder.SetInput<int>(Input(encoder, CosyVoiceFlowGraph.TokensInput), tokens);
        encoder.SetInput<float>(Input(encoder, CosyVoiceFlowGraph.SpeakerInput), Normalized(voice.XVector));
        var run = new RunParameters();
        run.Symbols[WellKnown.TokenSymbol] = tokens.Length;
        encoder.Run(run);
        var mu = new float[_tokenMelRatio * tokens.Length * _mels];
        var speaker = new float[_mels];
        encoder.GetOutput<float>(CosyVoiceFlowGraph.MuOutput, mu);
        encoder.GetOutput<float>(CosyVoiceFlowGraph.SpeakerOutput, speaker);
        return (mu, speaker);
    }

    /// <summary>One evaluation of the DiT vector field [frames, 80] (all inputs time-major).</summary>
    /// <param name="x">The current sample.</param>
    /// <param name="mu">The mean (zero for the unconditional evaluation).</param>
    /// <param name="speaker">The speaker condition (zero for the unconditional evaluation).</param>
    /// <param name="cond">The prompt mel, zero after the prompt (zero for the unconditional evaluation).</param>
    /// <param name="t">The time step.</param>
    public float[] Velocity(float[] x, float[] mu, float[] speaker, float[] cond, float t)
    {
        var estimator = _estimator ??= _model.CompileComponent(CosyVoiceAdapter.EstimatorRole);
        int frames = x.Length / _mels;
        estimator.SetInput<float>(Input(estimator, CosyVoiceFlowGraph.SampleInput), x);
        estimator.SetInput<float>(Input(estimator, CosyVoiceFlowGraph.MuInput), mu);
        estimator.SetInput<float>(Input(estimator, CosyVoiceFlowGraph.CondInput), cond);
        estimator.SetInput<float>(Input(estimator, CosyVoiceFlowGraph.SpeakerConditionInput), speaker);
        estimator.SetInput<float>(Input(estimator, CosyVoiceFlowGraph.TimeInput), [t]);
        estimator.SetInput<int>(Input(estimator, CosyVoiceFlowGraph.PositionsInput), [.. Enumerable.Range(0, frames)]);
        var run = new RunParameters();
        run.Symbols[CosyVoiceFlowGraph.FrameSymbol] = frames;
        estimator.Run(run);
        var output = new float[x.Length];
        estimator.GetOutput<float>(CosyVoiceFlowGraph.VelocityOutput, output);
        return output;
    }

    /// <summary>
    /// The mel spectrogram [frames, 80] of speech tokens after the voice's prompt: ten Euler steps over the cosine schedule of the guided field
    /// (<c>(1 + w) · conditioned − w · unconditioned</c>), from the given noise (or the noise source's) [prompt + frames, 80].
    /// </summary>
    /// <param name="speech">The speech tokens.</param>
    /// <param name="voice">The voice.</param>
    /// <param name="noise">The starting noise, time-major, or <see langword="null"/> for the noise source.</param>
    public float[] Mel(int[] speech, CosyVoiceVoice voice, float[]? noise = null)
    {
        var (mu, speaker) = EncodeFlow(speech, voice);
        int frames = mu.Length / _mels;
        int promptFrames = voice.PromptMel.Length / _mels;
        var cond = new float[frames * _mels];
        voice.PromptMel.AsSpan(0, Math.Min(voice.PromptMel.Length, cond.Length)).CopyTo(cond);
        var x = noise ?? (FlowNoiseSource ?? (n => SpeechNoise.Gaussian(n, 0)))(frames * _mels);
        if (x.Length != frames * _mels)
        {
            throw new ArgumentException($"The flow noise has {x.Length} values, {frames * _mels} are needed.", nameof(noise));
        }

        x = (float[])x.Clone();
        var zeros = new float[frames * _mels];
        var noSpeaker = new float[_mels];
        var span = new float[_flowSteps + 1];
        for (int i = 0; i <= _flowSteps; i++)
        {
            span[i] = 1f - MathF.Cos(i / (float)_flowSteps * 0.5f * MathF.PI);
        }

        float t = span[0];
        float dt = span[1] - span[0];
        for (int step = 1; step <= _flowSteps; step++)
        {
            var conditioned = Velocity(x, mu, speaker, cond, t);
            var unconditioned = Velocity(x, zeros, noSpeaker, zeros, t);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] += dt * (((1f + _flowGuidance) * conditioned[i]) - (_flowGuidance * unconditioned[i]));
            }

            t += dt;
            if (step < _flowSteps)
            {
                dt = span[step + 1] - t;
            }
        }

        return x[(promptFrames * _mels)..];
    }

    /// <summary>The causal F0 predictor's F0 per frame of a mel spectrogram [frames, 80].</summary>
    /// <param name="mel">The mel frames.</param>
    public float[] F0(float[] mel)
    {
        int frames = mel.Length / _mels;
        var predictor = _f0Predictor ??= _model.CompileComponent(CosyVoiceAdapter.VocoderRole, CosyVoiceVocoderGraph.F0GraphName);
        predictor.SetInput<float>(Input(predictor, CosyVoiceVocoderGraph.MelInput), mel);
        var run = new RunParameters();
        run.Symbols[WellKnown.TokenSymbol] = frames;
        predictor.Run(run);
        var f0 = new float[frames];
        predictor.GetOutput<float>(CosyVoiceVocoderGraph.F0Output, f0);
        return f0;
    }

    /// <summary>
    /// The causal HiFT over a mel spectrogram [frames, 80]: the waveform (clamped to ±0.99) and the F0 per frame (predicted unless given). The source
    /// noise [samples, harmonics] (uniform in [0, 1)) comes from the argument or the noise source.
    /// </summary>
    /// <param name="mel">The mel frames.</param>
    /// <param name="noise">The source noise, or <see langword="null"/>.</param>
    /// <param name="f0">The F0 per frame, or <see langword="null"/> for the predictor's.</param>
    public (float[] Samples, float[] F0) Vocode(float[] mel, float[]? noise = null, float[]? f0 = null)
    {
        int frames = mel.Length / _mels;
        int samples = frames * _samplesPerFrame;
        f0 ??= F0(mel);
        var vocoder = _vocoder ??= _model.CompileComponent(CosyVoiceAdapter.VocoderRole, CosyVoiceVocoderGraph.GraphName);
        noise ??= (SourceNoiseSource ?? (n => Uniform(n, 1)))(samples * _harmonics);
        vocoder.SetInput<float>(Input(vocoder, CosyVoiceVocoderGraph.MelInput), mel);
        var (sines, voiced) = CosyVoiceVocoderGraph.Harmonics(f0, _harmonics, SampleRate, _samplesPerFrame);
        vocoder.SetInput<float>(Input(vocoder, CosyVoiceVocoderGraph.SinesInput), sines);
        vocoder.SetInput<float>(Input(vocoder, CosyVoiceVocoderGraph.VoicedInput), voiced);
        vocoder.SetInput<float>(Input(vocoder, CosyVoiceVocoderGraph.NoiseInput), noise);
        var run = new RunParameters();
        run.Symbols[WellKnown.TokenSymbol] = frames;
        vocoder.Run(run);
        int stftFrames = (samples / _hop) + 1;
        var spectrum = new float[stftFrames * _fftSize];
        vocoder.GetOutput<float>(CosyVoiceVocoderGraph.FramesOutput, spectrum);
        var audio = OverlapAdd.Synthesize(spectrum, stftFrames, _fftSize, _hop, Window.Hann(_fftSize), _fftSize / 2);
        if (audio.Length > samples)
        {
            audio = audio[..samples];
        }

        for (int i = 0; i < audio.Length; i++)
        {
            audio[i] = Math.Clamp(audio[i], -AudioLimit, AudioLimit);
        }

        return (audio, f0);
    }

    /// <summary>The mel frames resampled for a speed factor (torch's linear interpolation to <c>floor(frames / speed)</c> frames, half-pixel centers).</summary>
    /// <param name="mel">The mel [frames, bins].</param>
    /// <param name="bins">The bins.</param>
    /// <param name="speed">The speed factor.</param>
    public static float[] Stretch(float[] mel, int bins, float speed)
    {
        int frames = mel.Length / bins;
        int target = Math.Max(1, (int)(frames / speed));
        if (target == frames)
        {
            return mel;
        }

        var result = new float[target * bins];
        double scale = (double)frames / target;
        for (int f = 0; f < target; f++)
        {
            double source = Math.Max(0, ((f + 0.5) * scale) - 0.5);
            int i0 = Math.Min((int)source, frames - 1);
            int i1 = Math.Min(i0 + 1, frames - 1);
            float w1 = (float)(source - i0);
            for (int c = 0; c < bins; c++)
            {
                result[(f * bins) + c] = ((1 - w1) * mel[(i0 * bins) + c]) + (w1 * mel[(i1 * bins) + c]);
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
        _flowEncoder?.Dispose();
        _estimator?.Dispose();
        _vocoder?.Dispose();
        _f0Predictor?.Dispose();
        _encoders?.Dispose();
        _gate.Dispose();
    }

    private CosyVoiceVoice VoiceFor(SpeechSynthesisOptions? options, CancellationToken cancellationToken)
    {
        if (options?.ReferenceAudio is not { } recording)
        {
            throw new InvalidOperationException("CosyVoice 3 has no built-in voices: pass a recording of the voice in SpeechSynthesisOptions.ReferenceAudio.");
        }

        if (_lastClone is { } last && ReferenceEquals(last.Recording, recording))
        {
            return last.Voice;
        }

        var voice = CloneVoice(recording, cancellationToken);
        _lastClone = (recording, voice);
        return voice;
    }

    private AudioBuffer Synthesize(string text, SpeechSynthesisOptions? options, CancellationToken cancellationToken)
    {
        var parts = new List<float>();
        foreach (string utterance in Utterances(text))
        {
            parts.AddRange(SynthesizeUtterance(utterance, options, cancellationToken));
        }

        return new AudioBuffer([.. parts], SampleRate);
    }

    private float[] SynthesizeUtterance(string utterance, SpeechSynthesisOptions? options, CancellationToken cancellationToken)
    {
        var voice = VoiceFor(options, cancellationToken);
        var sampling = options.SamplingOr(new SamplingOptions { Temperature = 1f, TopK = _topK, TopP = _topP });
        int[] promptText;
        int[] promptSpeech;
        string spoken = utterance;
        if (options?.Instructions is { Length: > 0 } instructions)
        {
            promptText = TextTokens(WithEndOfPrompt($"{SystemPrompt} {instructions}"));
            promptSpeech = [];
        }
        else if (options?.ReferenceText is { Length: > 0 } referenceText)
        {
            promptText = TextTokens(WithEndOfPrompt(referenceText.Contains(EndOfPrompt, StringComparison.Ordinal) ? referenceText : SystemPrompt + EndOfPrompt + referenceText));
            promptSpeech = voice.SpeechTokens;
        }
        else
        {
            promptText = [];
            promptSpeech = [];
            spoken = utterance.Contains(EndOfPrompt, StringComparison.Ordinal) ? utterance : SystemPrompt + EndOfPrompt + utterance;
        }

        var speech = SpeechTokens(TextTokens(spoken), promptText, promptSpeech, sampling, cancellationToken);
        if (speech.Length == 0)
        {
            return [];
        }

        var mel = Mel(speech, voice);
        float speed = options?.Speed ?? 1f;
        if (speed > 0 && Math.Abs(speed - 1f) > 1e-6f)
        {
            mel = Stretch(mel, _mels, speed);
        }

        return Vocode(mel).Samples;
    }

    private static string WithEndOfPrompt(string text) => text.Contains(EndOfPrompt, StringComparison.Ordinal) ? text : text + EndOfPrompt;

    private IReadOnlyList<string> Utterances(string text) => CosyVoiceTextFrontEnd.Split(text, piece => TextTokens(piece).Length);

    private async IAsyncEnumerable<AudioBuffer> StreamCoreAsync(string text, SpeechSynthesisOptions? options, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        foreach (string utterance in Utterances(text))
        {
            await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                var samples = await Task.Run(() => SynthesizeUtterance(utterance, options, cancellationToken), cancellationToken).ConfigureAwait(false);
                yield return new AudioBuffer(samples, SampleRate);
            }
            finally
            {
                _gate.Release();
            }
        }
    }

    private static float[] Uniform(int count, int seed)
    {
        var random = new Random(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = random.NextSingle();
        }

        return values;
    }

    private static float[] Normalized(float[] values)
    {
        double norm = 0;
        foreach (float v in values)
        {
            norm += (double)v * v;
        }

        float scale = (float)(1.0 / Math.Max(Math.Sqrt(norm), 1e-12));
        return [.. values.Select(v => v * scale)];
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

    private static Value Input(CompiledModel compiled, string name) => compiled.Graph.Inputs.First(v => v.Input!.Name == name);

    /// <summary>The embedding tables on the host: the speech embedding read once, text rows read per token from the weight.</summary>
    private sealed class Tables(ModelDefinition definition)
    {
        private readonly float[] _speech = WeightConversions.ReadFloats(definition.Weights["token_embedding.weight"]);
        private readonly int _width = (int)definition.Weights["token_embedding.weight"].Shape[1].Fixed;
        private readonly WeightInfo _text = definition.Weights[CosyVoiceAdapter.TextEmbeddingWeight];
        private float[]? _textAll;

        public ReadOnlySpan<float> SpeechEmbedding(int token) => _speech.AsSpan(token * _width, _width);

        public void TextEmbedding(int token, Span<float> target)
        {
            if (_text.Type == Tensors.DataType.F32)
            {
                _text.Source.Read((long)token * _width * sizeof(float), MemoryMarshal.AsBytes(target));
                return;
            }

            _textAll ??= WeightConversions.ReadFloats(_text);
            _textAll.AsSpan(token * _width, _width).CopyTo(target);
        }
    }
}
