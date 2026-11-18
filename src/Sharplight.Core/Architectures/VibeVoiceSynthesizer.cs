using System.Runtime.CompilerServices;
using System.Text;
using System.Text.RegularExpressions;
using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Speech;
using Sharplight.Tasks;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// The <see cref="ISpeechSynthesizer"/> of a VibeVoice model (<see cref="VibeVoiceAdapter"/>), as transformers' <c>VibeVoiceForConditionalGeneration.generate</c>: the prompt is the
/// system text, the voice prompts (one per speaker with a voice: <c>Speaker N:</c> and the speaker's acoustic latents in the place of the audio placeholders), the script
/// (<c>Speaker N: text</c> per turn) and <c>Speech output:</c> followed by the audio start token. Every step the language model's next token is chosen among the audio start, audio end,
/// audio (diffusion) and end-of-text tokens; an audio token asks the diffusion head for the next 64-wide acoustic latent (DPM-Solver, classifier-free guidance against a second run of the
/// language model that sees only an audio start token and the frames so far), the acoustic tokenizer's decoder turns the latent into 3200 samples (24 kHz, 7.5 frames per second), and
/// the next input of the language model is the acoustic connector's image of the latent plus the semantic connector's image of the semantic tokenizer's encoding of those samples.
/// An audio start token mid-speech starts a new segment: the context-free run starts over.
/// </summary>
internal sealed class VibeVoiceSynthesizer : ISpeechSynthesizer, IVoiceCloner
{
    /// <summary>Frames decoded per streamed buffer.</summary>
    public const int StreamFrames = 4;

    private static readonly Regex s_turn = new(@"^\s*Speaker\s+([^\s:]+)\s*:\s*(.*)$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

    private readonly TaskModel _model;
    private readonly LlmModel _llm;
    private readonly int _audioBos;
    private readonly int _audioEos;
    private readonly int _audioToken;
    private readonly int _eos;
    private readonly int _hidden;
    private readonly int _latent;
    private readonly int _semanticLatent;
    private readonly int _hop;
    private readonly int _frequencies;
    private readonly float _maxPeriod;
    private readonly int _trainSteps;
    private readonly string _betaSchedule;
    private readonly string _predictionType;
    private readonly float _defaultGuidance;
    private readonly int _defaultSteps;
    private readonly int _maxFrames;
    private readonly float _normalizeDbfs;
    private readonly string _systemPrompt;
    private readonly string _voicePrompt;
    private readonly string _textPrompt;
    private readonly string _speechPrompt;
    private readonly float _vaeStd;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly Lazy<Tables> _tables;
    private InferenceSession? _positive;
    private InferenceSession? _negative;
    private TokenizerStream? _decoder;
    private TokenizerStream? _semantic;
    private TokenizerStream? _encoder;
    private CompiledModel? _diffusion;
    private CompiledModel? _speechConnector;
    private CompiledModel? _voiceConnector;
    private bool _disposed;

    /// <summary>Creates the synthesizer; nothing is compiled until the first request.</summary>
    /// <param name="model">A loaded VibeVoice model.</param>
    public VibeVoiceSynthesizer(TaskModel model)
    {
        _model = model;
        _llm = LlmModel.FromTaskModel(model);
        var text = model.Description.Component(VibeVoiceAdapter.TextModelRole);
        _audioBos = text.ConfigValue("audioBos", 151652);
        _audioEos = text.ConfigValue("audioEos", 151653);
        _audioToken = text.ConfigValue("audioToken", 151654);
        _eos = text.ConfigValue("eos", 151643);
        _hidden = text.ConfigValue("hidden", 1536);
        _latent = text.ConfigValue("latent", 64);
        _semanticLatent = text.ConfigValue("semanticLatent", 128);
        _hop = text.ConfigValue("hop", VibeVoiceAdapter.Hop);
        _frequencies = text.ConfigValue("diffusion.frequencies", 256);
        _maxPeriod = (float)text.ConfigValue("diffusion.maxPeriod", 10000.0);
        _trainSteps = text.ConfigValue("diffusion.trainSteps", 1000);
        _betaSchedule = text.ConfigValue("diffusion.betaSchedule", "squaredcos_cap_v2");
        _predictionType = text.ConfigValue("diffusion.predictionType", "v_prediction");
        _defaultGuidance = (float)text.ConfigValue("sampling.cfg_scale", 1.3);
        _defaultSteps = (int)text.ConfigValue("sampling.diffusion_steps", 10.0);
        _maxFrames = (int)text.ConfigValue("sampling.max_frames", 4096.0);
        _normalizeDbfs = (float)text.ConfigValue("normalizeDbfs", -25.0);
        _systemPrompt = text.ConfigValue("prompt.system", "");
        _voicePrompt = text.ConfigValue("prompt.voice", "");
        _textPrompt = text.ConfigValue("prompt.text", "");
        _speechPrompt = text.ConfigValue("prompt.speech", "");
        _vaeStd = (float)text.ConfigValue("vaeStd", 0.625);
        SampleRate = text.ConfigValue("sampleRate", 24000);
        _tables = new Lazy<Tables>(() => new Tables(model.Definition, _hidden));
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    /// <inheritdoc/>
    public TaskClientMetadata Metadata { get; }

    /// <inheritdoc/>
    public int SampleRate { get; }

    /// <summary>The model has no preset voices: speakers speak with the voice the model invents unless a reference recording or a saved voice is given.</summary>
    public IReadOnlyList<string> Voices => [];

    /// <inheritdoc/>
    public string VoiceArchitecture => VibeVoiceAdapter.Family;

    /// <summary>Whether <paramref name="model"/> is a VibeVoice model (<see cref="VibeVoiceAdapter"/>).</summary>
    /// <param name="model">A loaded speech-synthesis model.</param>
    public static bool Handles(TaskModel model)
        => model.Description.TryGetComponent(VibeVoiceAdapter.TextModelRole, out var text) && text.ConfigValue("family", "") == VibeVoiceAdapter.Family
           && model.Description.TryGetComponent(VibeVoiceAdapter.DiffusionRole, out _);

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
        var audio = await BackgroundCompute.RunAsync(_gate, () =>
        {
            var samples = new List<float>();
            foreach (var frame in GenerateFrames(text, options, cancellationToken))
            {
                samples.AddRange(frame.Audio);
            }

            return new AudioBuffer([.. samples], SampleRate);
        }, cancellationToken).ConfigureAwait(false);
        return new SpeechSynthesisResult(audio, options?.Voice ?? "");
    }

    /// <inheritdoc/>
    /// <remarks>Returns at once: generation runs on a background task (<see cref="BackgroundCompute"/>), one request at a time per client; a buffer holds <see cref="StreamFrames"/> frames (about half a second).</remarks>
    public IAsyncEnumerable<AudioBuffer> SynthesizeStreamingAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
        => BackgroundCompute.StreamAsync(token => SynthesizeCoreAsync(text, options, token), cancellationToken);

    private async IAsyncEnumerable<AudioBuffer> SynthesizeCoreAsync(string text, SpeechSynthesisOptions? options, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var pending = new List<float>();
            using var generation = GenerateFrames(text, options, cancellationToken).GetEnumerator();
            int frames = 0;
            while (await Task.Run(generation.MoveNext, cancellationToken).ConfigureAwait(false))
            {
                pending.AddRange(generation.Current.Audio);
                if (++frames % StreamFrames == 0)
                {
                    yield return new AudioBuffer([.. pending], SampleRate);
                    pending.Clear();
                }
            }

            if (pending.Count > 0)
            {
                yield return new AudioBuffer([.. pending], SampleRate);
            }
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <inheritdoc/>
    public Task<VoiceProfile> CreateVoiceAsync(string name, AudioBuffer recording, string? referenceText = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(recording);
        return BackgroundCompute.RunAsync(_gate, () =>
        {
            var latents = EncodeVoice(recording, new VibeVoiceNoise(0), cancellationToken);
            return new VoiceProfile
            {
                Name = name,
                Architecture = VibeVoiceAdapter.Family,
                ReferenceText = referenceText,
                SampleRate = SampleRate,
                Tensors = new Dictionary<string, VoiceTensor> { [VoiceKey] = VoiceTensor.From(latents, [latents.Length / _latent, _latent]) },
            };
        }, cancellationToken);
    }

    /// <summary>The name of the tensor of a voice profile that holds the scaled acoustic latents [frames, latent].</summary>
    public const string VoiceKey = "latents";

    /// <summary>The turns of a script: lines <c>Speaker N: text</c>; lines without a label continue the turn before (a script without labels is one turn of speaker 0).</summary>
    /// <param name="text">The script.</param>
    public static IReadOnlyList<VibeVoiceTurn> ParseScript(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var turns = new List<VibeVoiceTurn>();
        foreach (string raw in text.Split('\n'))
        {
            string line = raw.TrimEnd('\r');
            if (s_turn.Match(line) is { Success: true } match)
            {
                turns.Add(new VibeVoiceTurn(match.Groups[1].Value, match.Groups[2].Value.Trim()));
            }
            else if (line.Trim().Length > 0)
            {
                if (turns.Count == 0)
                {
                    turns.Add(new VibeVoiceTurn("0", line.Trim()));
                }
                else
                {
                    var last = turns[^1];
                    turns[^1] = last with { Text = (last.Text + " " + line.Trim()).Trim() };
                }
            }
        }

        return turns;
    }

    /// <summary>The prompt of a request: its token ids (audio placeholders where the voice prompts go) and the voice rows [frames, hidden] of each voiced speaker, in the order of the placeholders.</summary>
    /// <param name="turns">The script.</param>
    /// <param name="voices">The scaled acoustic latents [frames, latent] of the speakers that have a voice, by speaker.</param>
    public (int[] Tokens, List<int> VoiceFrames) PromptTokens(IReadOnlyList<VibeVoiceTurn> turns, IReadOnlyList<(string Speaker, int Frames)> voices)
    {
        var prompt = new StringBuilder(_systemPrompt);
        string bos = _llm.Tokenizer.TokenString(_audioBos);
        string eos = _llm.Tokenizer.TokenString(_audioEos);
        string pad = _llm.Tokenizer.TokenString(_audioToken);
        var frames = new List<int>();
        if (voices.Count > 0)
        {
            prompt.Append(_voicePrompt);
            foreach (var (speaker, count) in voices)
            {
                prompt.Append(" Speaker ").Append(speaker).Append(':').Append(bos);
                for (int i = 0; i < count; i++)
                {
                    prompt.Append(pad);
                }

                prompt.Append(eos).Append('\n');
                frames.Add(count);
            }
        }

        prompt.Append(_textPrompt);
        foreach (var turn in turns)
        {
            prompt.Append(" Speaker ").Append(turn.Speaker).Append(": ").Append(turn.Text).Append('\n');
        }

        prompt.Append(_speechPrompt).Append(bos);
        return (_llm.Tokenizer.Encode(prompt.ToString(), addSpecial: false, parseSpecial: true), frames);
    }

    /// <summary>
    /// Generates the frames of a request one by one: each has the acoustic latent the diffusion head produced and the 3200 samples the acoustic decoder made of it.
    /// </summary>
    /// <param name="text">The script.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Stops generation between frames.</param>
    /// <param name="forcing">Feeds the language model the given tokens, latents and noise instead of its own (teacher forcing for verification), or <see langword="null"/>.</param>
    /// <param name="observe">Receives each step's logits and hidden states, or <see langword="null"/>.</param>
    public IEnumerable<VibeVoiceFrame> GenerateFrames(string text, SpeechSynthesisOptions? options, CancellationToken cancellationToken, VibeVoiceForcing? forcing = null,
        Action<VibeVoiceStep>? observe = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        ObjectDisposedException.ThrowIf(_disposed, this);
        var tables = _tables.Value;
        int h = _hidden;
        var turns = ParseScript(text).Select(t => t with { Text = options.NormalizedText(t.Text) }).ToList();
        if (turns.Count == 0)
        {
            yield break;
        }

        var noise = new VibeVoiceNoise(options?.Sampling.Seed);
        var voices = ResolveVoices(turns, options, noise, cancellationToken);
        var (tokens, voiceFrames) = PromptTokens(turns, [.. voices.Select(v => (v.Speaker, v.Latents.Length / _latent))]);
        var positive = _positive ??= CreateSession();
        var negative = _negative ??= CreateSession();
        positive.Reset();
        negative.Reset();
        _decoder ??= new TokenizerStream(_model.CompileComponent(VibeVoiceAdapter.AcousticDecoderRole, VibeVoiceAdapter.AcousticDecodeGraph), VibeVoiceAdapter.AcousticPrefix);
        _semantic ??= new TokenizerStream(_model.CompileComponent(VibeVoiceAdapter.SemanticEncoderRole, VibeVoiceAdapter.SemanticEncodeGraph), VibeVoiceAdapter.SemanticPrefix);
        _decoder.Caches.Reset();
        _semantic.Caches.Reset();

        // The prompt rows: embedding rows of the tokens, the voice rows at the audio placeholders.
        var voiceRows = voices.Select(v => ConnectVoice(v.Latents)).ToList();
        var rows = new float[tokens.Length * h];
        int voiceIndex = 0;
        int inVoice = 0;
        for (int i = 0; i < tokens.Length; i++)
        {
            if (tokens[i] == _audioToken && voiceIndex < voiceRows.Count)
            {
                voiceRows[voiceIndex].AsSpan(inVoice * h, h).CopyTo(rows.AsSpan(i * h, h));
                if (++inVoice == voiceFrames[voiceIndex])
                {
                    voiceIndex++;
                    inVoice = 0;
                }
            }
            else
            {
                tables.TokenRow(tokens[i]).CopyTo(rows, i * h);
            }
        }

        var logits = Process(positive, rows, tokens.Length);
        var hidden = positive.LastHidden[^h..].ToArray();
        float[]? inputRow = null;
        bool negativeStarted = false;
        int budget = Math.Min(_maxFrames, positive.ContextLength - tokens.Length - 1);
        int steps = options?.DiffusionSteps ?? _defaultSteps;
        float guidance = options?.GuidanceScale ?? _defaultGuidance;
        var solver = new VibeVoiceDpmSolver(_trainSteps, _betaSchedule, _predictionType);
        int frame = 0;
        for (int step = 0; step < budget; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int token = forcing?.Tokens is { } forced ? (step < forced.Count ? forced[step] : _eos) : Constrained(logits);
            float[] next = tables.TokenRow(token);
            float[]? negativeHidden = null;
            float[]? latent = null;
            if (token == _audioBos)
            {
                // A new speech segment: the context-free run starts over (the reference masks every earlier position, so its next input, the audio start token's embedding, is all it sees).
                if (negativeStarted)
                {
                    negative.Reset();
                    negativeStarted = false;
                }
            }
            else if (token == _audioToken)
            {
                Process(negative, inputRow ?? tables.TokenRow(_audioBos), 1);
                negativeStarted = true;
                negativeHidden = negative.LastHidden[^h..].ToArray();
                latent = Diffuse(solver, hidden, negativeHidden, steps, guidance, forcing?.Noise is { } given && frame < given.Count ? given[frame] : noise.Next(_latent));
            }

            observe?.Invoke(new VibeVoiceStep(step, token, logits, hidden, negativeHidden, latent));
            if (token == _eos)
            {
                yield break;
            }

            if (token == _audioToken)
            {
                var used = forcing?.Latents is { } fed && frame < fed.Count ? fed[frame] : latent!;
                var audio = DecodeFrame(used, tables);
                var feature = EncodeSemantic(audio);
                next = Connect(used, feature);
                yield return new VibeVoiceFrame(used, audio);
                frame++;
            }

            logits = Process(positive, next, 1);
            hidden = positive.LastHidden[^h..].ToArray();
            inputRow = next;
        }
    }

    /// <summary>The scaled acoustic latents [frames · latent] of a reference recording: normalized to the model's level, encoded in blocks, a sample of the encoder's distribution, then <c>(latent + bias) · scale</c>.</summary>
    /// <param name="recording">The recording (any rate and channel count).</param>
    /// <param name="noise">The noise of the sample.</param>
    /// <param name="cancellationToken">Stops between blocks.</param>
    /// <param name="sample">Whether to add the encoder's sampling noise (false: its mean).</param>
    public float[] EncodeVoice(AudioBuffer recording, VibeVoiceNoise noise, CancellationToken cancellationToken, bool sample = true)
    {
        var tables = _tables.Value;
        var samples = Normalize(recording.ToModelInput(SampleRate).Samples.ToArray());
        int frames = (samples.Length + _hop - 1) / _hop;
        int block = VibeVoiceAdapter.EncodeFrames;
        var encoder = _encoder ??= new TokenizerStream(_model.CompileComponent(VibeVoiceAdapter.AcousticEncoderRole, VibeVoiceAdapter.AcousticEncodeGraph), VibeVoiceAdapter.AcousticPrefix);
        encoder.Caches.Reset();
        var latents = new float[frames * _latent];
        for (int first = 0; first < frames; first += block)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var input = new float[block * _hop];
            int count = Math.Min(input.Length, Math.Max(0, samples.Length - (first * _hop)));
            samples.AsSpan(first * _hop, count).CopyTo(input);
            var output = encoder.Run("audio", input, "latents", block * _latent);
            output.AsSpan(0, Math.Min(block, frames - first) * _latent).CopyTo(latents.AsSpan(first * _latent));
        }

        if (sample)
        {
            float scale = _vaeStd * (float)noise.Next();
            for (int i = 0; i < latents.Length; i++)
            {
                latents[i] += scale * (float)noise.Next();
            }
        }

        for (int i = 0; i < latents.Length; i++)
        {
            latents[i] = (latents[i] + tables.LatentBias) * tables.LatentScale;
        }

        return latents;
    }

    /// <summary>The semantic tokenizer's encodings [frames · semantic latent] and the acoustic tokenizer's latents [frames · latent] of audio (tokenizer verification), frame by frame as generation runs them.</summary>
    /// <param name="audio">24 kHz mono samples, a multiple of the hop.</param>
    public (float[] Acoustic, float[] Semantic) EncodeBoth(float[] audio)
    {
        ArgumentNullException.ThrowIfNull(audio);
        int frames = audio.Length / _hop;
        int block = VibeVoiceAdapter.EncodeFrames;
        var encoder = _encoder ??= new TokenizerStream(_model.CompileComponent(VibeVoiceAdapter.AcousticEncoderRole, VibeVoiceAdapter.AcousticEncodeGraph), VibeVoiceAdapter.AcousticPrefix);
        encoder.Caches.Reset();
        var acoustic = new float[frames * _latent];
        for (int first = 0; first < frames; first += block)
        {
            var input = new float[block * _hop];
            audio.AsSpan(first * _hop, Math.Min(input.Length, audio.Length - (first * _hop))).CopyTo(input);
            encoder.Run("audio", input, "latents", block * _latent).AsSpan(0, Math.Min(block, frames - first) * _latent).CopyTo(acoustic.AsSpan(first * _latent));
        }

        _semantic ??= new TokenizerStream(_model.CompileComponent(VibeVoiceAdapter.SemanticEncoderRole, VibeVoiceAdapter.SemanticEncodeGraph), VibeVoiceAdapter.SemanticPrefix);
        _semantic.Caches.Reset();
        var semantic = new float[frames * _semanticLatent];
        for (int f = 0; f < frames; f++)
        {
            EncodeSemantic(audio.AsSpan(f * _hop, _hop).ToArray()).CopyTo(semantic, f * _semanticLatent);
        }

        return (acoustic, semantic);
    }

    /// <summary>Decodes acoustic latents (as the model produces them, scaled) to audio, frame by frame (decoder verification).</summary>
    /// <param name="latents">The latents [frames · latent].</param>
    /// <param name="scaled">Whether the latents are in the model's scaled domain (as the diffusion head produces them) or the tokenizer's own.</param>
    public float[] DecodeLatents(float[] latents, bool scaled = true)
    {
        ArgumentNullException.ThrowIfNull(latents);
        var tables = _tables.Value;
        _decoder ??= new TokenizerStream(_model.CompileComponent(VibeVoiceAdapter.AcousticDecoderRole, VibeVoiceAdapter.AcousticDecodeGraph), VibeVoiceAdapter.AcousticPrefix);
        _decoder.Caches.Reset();
        int frames = latents.Length / _latent;
        var audio = new float[frames * _hop];
        for (int f = 0; f < frames; f++)
        {
            var frame = latents.AsSpan(f * _latent, _latent).ToArray();
            (scaled ? DecodeFrame(frame, tables) : _decoder.Run("latents", frame, AudioCodecGraphs.AudioOutput, _hop)).CopyTo(audio, f * _hop);
        }

        return audio;
    }

    /// <summary>
    /// Runs the diffusion head's sampling loop for one frame: <paramref name="initial"/> noise denoised over <paramref name="steps"/> steps of the DPM solver with classifier-free guidance
    /// between the conditional and the unconditional hidden state.
    /// </summary>
    /// <param name="solver">The sampler.</param>
    /// <param name="condition">The conditional hidden state.</param>
    /// <param name="unconditional">The unconditional hidden state.</param>
    /// <param name="steps">The steps.</param>
    /// <param name="guidance">The guidance scale.</param>
    /// <param name="initial">The starting noise [latent].</param>
    public float[] Diffuse(VibeVoiceDpmSolver solver, float[] condition, float[] unconditional, int steps, float guidance, float[] initial)
    {
        var head = _diffusion ??= _model.CompileComponent(VibeVoiceAdapter.DiffusionRole, VibeVoiceDiffusionGraph.Graph);
        var graph = head.Graph;
        Value Input(string name) => graph.Inputs.First(i => i.Input!.Name == name);
        var rows = new float[2 * _hidden];
        condition.CopyTo(rows, 0);
        unconditional.CopyTo(rows, _hidden);
        head.SetInput<float>(Input(VibeVoiceDiffusionGraph.ConditionInput), rows);
        var sample = (float[])initial.Clone();
        var noisy = new float[2 * _latent];
        var velocity = new float[2 * _latent];
        solver.Begin(steps);
        for (int i = 0; i < steps; i++)
        {
            sample.CopyTo(noisy, 0);
            sample.CopyTo(noisy, _latent);
            head.SetInput<float>(Input(VibeVoiceDiffusionGraph.NoisyInput), noisy);
            head.SetInput<float>(Input(VibeVoiceDiffusionGraph.TimestepInput), TimestepEmbedding(solver.Timestep(i)));
            head.Run(new RunParameters());
            head.GetOutput<float>(VibeVoiceDiffusionGraph.Output, velocity);
            var guided = new float[_latent];
            for (int k = 0; k < _latent; k++)
            {
                float conditional = velocity[k];
                float unconditionalValue = velocity[_latent + k];
                guided[k] = unconditionalValue + (guidance * (conditional - unconditionalValue));
            }

            sample = solver.Step(guided, sample);
        }

        return sample;
    }

    /// <summary>The sinusoidal embedding of a timestep: cosines then sines of <c>t · exp(-ln(period) · k / half)</c>.</summary>
    /// <param name="timestep">The timestep.</param>
    public float[] TimestepEmbedding(int timestep)
    {
        int half = _frequencies / 2;
        var embedding = new float[_frequencies];
        for (int k = 0; k < half; k++)
        {
            float frequency = MathF.Exp(-MathF.Log(_maxPeriod) * k / half);
            float angle = timestep * frequency;
            embedding[k] = MathF.Cos(angle);
            embedding[half + k] = MathF.Sin(angle);
        }

        return embedding;
    }

    // Language model rows for generated frames: acoustic connector of the latent plus semantic connector of its semantic feature.
    private float[] Connect(float[] latent, float[] semantic)
    {
        var connector = _speechConnector ??= _model.CompileComponent(VibeVoiceAdapter.ConnectorRole, VibeVoiceAdapter.SpeechConnectorGraph);
        var graph = connector.Graph;
        connector.SetInput<float>(graph.Inputs.First(i => i.Input!.Name == "acoustic"), latent);
        connector.SetInput<float>(graph.Inputs.First(i => i.Input!.Name == "semantic"), semantic);
        var parameters = new RunParameters();
        parameters.Symbols[WellKnown.TokenSymbol] = 1;
        connector.Run(parameters);
        var rows = new float[_hidden];
        connector.GetOutput<float>("rows", rows);
        return rows;
    }

    // Language model rows of a voice prompt: the acoustic connector of each scaled latent.
    private float[] ConnectVoice(float[] latents)
    {
        var connector = _voiceConnector ??= _model.CompileComponent(VibeVoiceAdapter.ConnectorRole, VibeVoiceAdapter.VoiceConnectorGraph);
        int frames = latents.Length / _latent;
        var rows = new float[frames * _hidden];
        for (int first = 0; first < frames; first += VibeVoiceAdapter.MaxVoiceFrames)
        {
            int count = Math.Min(VibeVoiceAdapter.MaxVoiceFrames, frames - first);
            connector.SetInput<float>(connector.Graph.Inputs.First(i => i.Input!.Name == "acoustic"), latents.AsSpan(first * _latent, count * _latent));
            var parameters = new RunParameters();
            parameters.Symbols[WellKnown.TokenSymbol] = count;
            connector.Run(parameters);
            connector.GetOutput<float>("rows", rows.AsSpan(first * _hidden, count * _hidden));
        }

        return rows;
    }

    // One frame of audio from one latent (the diffusion head's output is in the scaled domain).
    private float[] DecodeFrame(float[] latent, Tables tables)
    {
        var scaled = new float[_latent];
        for (int i = 0; i < scaled.Length; i++)
        {
            scaled[i] = (latent[i] / tables.LatentScale) - tables.LatentBias;
        }

        return _decoder!.Run("latents", scaled, AudioCodecGraphs.AudioOutput, _hop);
    }

    private float[] EncodeSemantic(float[] audio) => _semantic!.Run(AudioCodecGraphs.AudioInput, audio, "latents", _semanticLatent);

    // The speakers' voices: from the request's saved voices and reference recording, in the order the speakers first speak.
    private List<(string Speaker, float[] Latents)> ResolveVoices(IReadOnlyList<VibeVoiceTurn> turns, SpeechSynthesisOptions? options, VibeVoiceNoise noise, CancellationToken cancellationToken)
    {
        var voices = new List<(string, float[])>();
        var speakers = turns.Select(t => t.Speaker).Distinct(StringComparer.Ordinal).ToList();
        for (int i = 0; i < speakers.Count; i++)
        {
            var profile = options?.SpeakerVoices is { } map && map.TryGetValue(speakers[i], out var chosen) ? chosen : i == 0 ? options?.VoiceProfile : null;
            if (profile is not null)
            {
                if (profile.Architecture != VibeVoiceAdapter.Family)
                {
                    throw new ArgumentException($"The voice '{profile.Name}' belongs to '{profile.Architecture}', not to VibeVoice.");
                }

                voices.Add((speakers[i], profile.Tensors[VoiceKey].ToFloats()));
            }
            else if (i == 0 && options?.ReferenceAudio is { } recording)
            {
                voices.Add((speakers[i], EncodeVoice(recording, noise, cancellationToken)));
            }
        }

        return voices;
    }

    // The next token among the audio start, audio end, audio and end-of-text tokens.
    private int Constrained(float[] logits)
    {
        int best = _audioToken;
        foreach (int id in new[] { _audioBos, _audioEos, _audioToken, _eos })
        {
            if (logits[id] > logits[best])
            {
                best = id;
            }
        }

        return best;
    }

    private float[] Normalize(float[] samples)
    {
        double energy = 0;
        foreach (float s in samples)
        {
            energy += (double)s * s;
        }

        double rms = Math.Sqrt(energy / Math.Max(1, samples.Length));
        float gain = (float)(Math.Pow(10, _normalizeDbfs / 20) / (rms + 1e-6));
        float peak = 0;
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] *= gain;
            peak = MathF.Max(peak, MathF.Abs(samples[i]));
        }

        if (peak > 1f)
        {
            for (int i = 0; i < samples.Length; i++)
            {
                samples[i] /= peak + 1e-6f;
            }
        }

        return samples;
    }

    private InferenceSession CreateSession()
    {
        var session = _llm.CreateSession(WellKnown.DecodeMultimodalGraph);
        session.CaptureHidden = true;
        return session;
    }

    private static float[] Process(InferenceSession session, float[] rows, int count)
    {
        int first = session.RopePosition;
        var prompt = new MultimodalPrompt
        {
            Tokens = new int[count],
            ImageIndex = new int[count],
            ImageRow = [.. Enumerable.Range(0, count)],
            Positions = [.. Enumerable.Range(first, count)],
            PositionAxes = 1,
            NextPosition = first + count,
        };
        return session.Process(prompt, [rows]);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _positive?.Dispose();
        _negative?.Dispose();
        _decoder?.Dispose();
        _semantic?.Dispose();
        _encoder?.Dispose();
        _diffusion?.Dispose();
        _speechConnector?.Dispose();
        _voiceConnector?.Dispose();
        _gate.Dispose();
    }

    /// <summary>A compiled tokenizer graph with the caches of its convolutions.</summary>
    private sealed class TokenizerStream : IDisposable
    {
        private readonly CompiledModel _compiled;
        private readonly Dictionary<string, Value> _inputs;

        public TokenizerStream(CompiledModel compiled, string prefix)
        {
            _compiled = compiled;
            _inputs = compiled.Graph.Inputs.ToDictionary(i => i.Input!.Name, i => i, StringComparer.Ordinal);
            Caches = new VibeVoiceConvCaches(compiled, prefix);
        }

        public VibeVoiceConvCaches Caches { get; }

        // One run: the input, the caches in, the output out, the caches refreshed.
        public float[] Run(string input, float[] values, string output, int length)
        {
            _compiled.SetInput<float>(_inputs[input], values);
            Caches.Bind();
            _compiled.Run(new RunParameters());
            var result = new float[length];
            _compiled.GetOutput<float>(output, result);
            Caches.Collect();
            return result;
        }

        public void Dispose() => _compiled.Dispose();
    }

    /// <summary>The host-side tables: the token embedding (rows read on demand) and the latent scale and bias.</summary>
    private sealed class Tables
    {
        private readonly WeightInfo _embedding;
        private readonly int _hidden;

        public Tables(ModelDefinition definition, int hidden)
        {
            _embedding = definition.Weights["token_embedding.weight"];
            _hidden = hidden;
            LatentScale = WeightConversions.ReadFloats(definition.Weights["vibevoice.latent_scale"])[0];
            LatentBias = WeightConversions.ReadFloats(definition.Weights["vibevoice.latent_bias"])[0];
        }

        public float LatentScale { get; }

        public float LatentBias { get; }

        public float[] TokenRow(int token) => WeightConversions.ReadFloats(_embedding, (long)token * _hidden, _hidden);
    }
}
