using System.Runtime.CompilerServices;
using Sharplight.Audio;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Speech;
using Sharplight.Tasks;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// The <see cref="ISpeechSynthesizer"/> of a VoxCPM2 model (<see cref="VoxCpm2Adapter"/>), as the official package's <c>VoxCPM2Model._generate_with_prompt_cache</c>: the prompt is
/// the text tokens with the audio start token, preceded by the isolated reference voice (<c>[reference start, latent patches, reference end]</c>) when one is given and
/// followed by the continuation prompt's patches (its transcript goes before the text) when the reference recording comes with its transcript. Every step the diffusion
/// transformer turns the two language models' states and the previous patch into the next 4-frame latent patch (Euler steps with classifier-free guidance, zero-star
/// rescaling, the first step skipped), the local encoder feeds the patch back to both language models, and the stop head ends the speech; the AudioVAE decodes each patch
/// to 160 ms of 48 kHz audio as it comes (its causal convolutions carry their context, so streaming equals a whole decode).
/// <see cref="SpeechSynthesisOptions.Instructions"/> become the parenthesized voice description the model reads before the text (voice design, or style control of a
/// cloned voice).
/// </summary>
internal sealed class VoxCpm2Synthesizer : ISpeechSynthesizer, IVoiceCloner
{
    /// <summary>The name of a voice profile's tensor with the reference patches ([patches · patch, feature], the recording padded at its end).</summary>
    public const string ReferenceKey = "reference";

    /// <summary>The name of a voice profile's tensor with the continuation patches ([patches · patch, feature], the recording padded at its start).</summary>
    public const string PromptKey = "prompt";

    /// <summary>Patches decoded per streamed buffer.</summary>
    public const int StreamPatches = 2;

    // The reference's defaults (VoxCPM.generate): at least this many patches before the stop head may end, at most target tokens · ratio + slack patches.
    private const int MinPatches = 2;
    private const int MaxPatches = 4096;
    private const float LengthRatio = 6f;
    private const int LengthSlack = 10;

    // Trailing prompt patches the decoder sees before the first generated one (the reference's streaming_prefix_len - 1).
    private const int ContextPatches = 3;

    private readonly TaskModel _model;
    private readonly LlmModel _llm;
    private readonly VoxCpm2Text _text;
    private readonly int _hidden;
    private readonly int _patch;
    private readonly int _feature;
    private readonly int _ditWidth;
    private readonly int _audioStart;
    private readonly int _referenceStart;
    private readonly int _referenceEnd;
    private readonly bool _meanMode;
    private readonly float _defaultGuidance;
    private readonly int _defaultSteps;
    private readonly int _inputRate;
    private readonly int _encoderHop;
    private readonly int _decoderHop;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private InferenceSession? _base;
    private InferenceSession? _residual;
    private CompiledModel? _encoder;
    private CompiledModel? _dit;
    private CompiledModel? _fuse;
    private CompiledModel? _condition;
    private CompiledModel? _vaeDecoder;
    private CompiledModel? _vaeEncoder;
    private VibeVoiceConvCaches? _decoderCaches;
    private VibeVoiceConvCaches? _encoderCaches;
    private bool _disposed;

    /// <summary>Creates the synthesizer; nothing is compiled until the first request.</summary>
    /// <param name="model">A loaded VoxCPM2 model.</param>
    public VoxCpm2Synthesizer(TaskModel model)
    {
        _model = model;
        _llm = LlmModel.FromTaskModel(model);
        _text = new VoxCpm2Text(_llm.Tokenizer);
        var text = model.Description.Component(VoxCpm2Adapter.TextModelRole);
        _hidden = text.ConfigValue("hidden", 2048);
        _patch = text.ConfigValue("patchSize", 4);
        _feature = text.ConfigValue("featureSize", 64);
        _ditWidth = text.ConfigValue("ditWidth", 1024);
        _audioStart = text.ConfigValue("audioStart", 101);
        _referenceStart = text.ConfigValue("referenceStart", 103);
        _referenceEnd = text.ConfigValue("referenceEnd", 104);
        _meanMode = text.ConfigValue("meanMode", false);
        _defaultGuidance = (float)text.ConfigValue("sampling.guidance", 2.0);
        _defaultSteps = text.ConfigValue("sampling.steps", 10);
        _inputRate = text.ConfigValue("inputSampleRate", 16000);
        _encoderHop = text.ConfigValue("encoderHop", 640);
        _decoderHop = text.ConfigValue("decoderHop", 1920);
        SampleRate = model.Description.ConfigValue("sampleRate", 48000);
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    /// <inheritdoc/>
    public TaskClientMetadata Metadata { get; }

    /// <inheritdoc/>
    public int SampleRate { get; }

    /// <summary>The model has no preset voices: it invents one (or one described by <see cref="SpeechSynthesisOptions.Instructions"/>) unless a reference recording or a saved voice is given.</summary>
    public IReadOnlyList<string> Voices => [];

    /// <inheritdoc/>
    public string VoiceArchitecture => VoxCpm2Adapter.Family;

    /// <summary>The samples of output audio per patch.</summary>
    public int SamplesPerPatch => _patch * _decoderHop;

    /// <summary>Whether <paramref name="model"/> is a VoxCPM2 model (<see cref="VoxCpm2Adapter"/>).</summary>
    /// <param name="model">A loaded speech-synthesis model.</param>
    public static bool Handles(TaskModel model)
        => model.Description.TryGetComponent(VoxCpm2Adapter.TextModelRole, out var text) && text.ConfigValue("family", "") == VoxCpm2Adapter.Family
           && model.Description.TryGetComponent(VoxCpm2Adapter.DitRole, out _);

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
            foreach (var patch in GeneratePatches(text, options, cancellationToken))
            {
                samples.AddRange(patch.Audio);
            }

            return new AudioBuffer([.. samples], SampleRate);
        }, cancellationToken).ConfigureAwait(false);
        return new SpeechSynthesisResult(audio, options?.Voice ?? "");
    }

    /// <inheritdoc/>
    /// <remarks>Returns at once: generation runs on a background task (<see cref="BackgroundCompute"/>), one request at a time per client; a buffer holds <see cref="StreamPatches"/> patches (320 ms).</remarks>
    public IAsyncEnumerable<AudioBuffer> SynthesizeStreamingAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
        => BackgroundCompute.StreamAsync(token => SynthesizeCoreAsync(text, options, token), cancellationToken);

    private async IAsyncEnumerable<AudioBuffer> SynthesizeCoreAsync(string text, SpeechSynthesisOptions? options, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var pending = new List<float>();
            using var generation = GeneratePatches(text, options, cancellationToken).GetEnumerator();
            int patches = 0;
            while (await Task.Run(generation.MoveNext, cancellationToken).ConfigureAwait(false))
            {
                pending.AddRange(generation.Current.Audio);
                if (++patches % StreamPatches == 0)
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
    /// <remarks>
    /// The profile keeps the recording's latent patches as the isolated reference (<see cref="ReferenceKey"/>) and, with a transcript, also as the continuation prompt
    /// (<see cref="PromptKey"/>, the reference's "ultimate cloning"); the recording is resampled to the model's 16 kHz input.
    /// </remarks>
    public Task<VoiceProfile> CreateVoiceAsync(string name, AudioBuffer recording, string? referenceText = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(recording);
        return BackgroundCompute.RunAsync(_gate, () =>
        {
            var samples = recording.ToModelInput(_inputRate).Samples.ToArray();
            var tensors = new Dictionary<string, VoiceTensor> { [ReferenceKey] = Patches(EncodeAudio(samples, padStart: false, cancellationToken)) };
            if (!string.IsNullOrWhiteSpace(referenceText))
            {
                tensors[PromptKey] = Patches(EncodeAudio(samples, padStart: true, cancellationToken));
            }

            return new VoiceProfile
            {
                Name = name,
                Architecture = VoxCpm2Adapter.Family,
                ReferenceText = string.IsNullOrWhiteSpace(referenceText) ? null : referenceText,
                SampleRate = _inputRate,
                Tensors = tensors,
            };
        }, cancellationToken);
    }

    private VoiceTensor Patches(float[] latents) => VoiceTensor.From(latents, [latents.Length / _feature, _feature]);

    /// <summary>The tokens of a text (no special tokens; multi-character CJK pieces split, see <see cref="VoxCpm2Text"/>).</summary>
    /// <param name="text">The text.</param>
    public int[] Tokenize(string text) => _text.Encode(text);

    /// <summary>
    /// The prompt of a request: the isolated reference (when <paramref name="reference"/> is given), the text tokens (the continuation's transcript, then the text to
    /// speak) with the audio start token, then the continuation's patches (when <paramref name="continuation"/> is given).
    /// </summary>
    /// <param name="target">The text to speak (folded and with any voice description already in place).</param>
    /// <param name="reference">The reference patches ([patches · patch, feature]), or <see langword="null"/>.</param>
    /// <param name="continuation">The continuation prompt's transcript and patches, or <see langword="null"/>.</param>
    public VoxCpm2Prompt BuildPrompt(string target, float[]? reference, (string Text, float[] Latents)? continuation)
    {
        ArgumentNullException.ThrowIfNull(target);
        int size = _patch * _feature;
        var tokens = new List<int>();
        var audio = new List<bool>();
        var patches = new List<float[]>();
        void AddPatches(float[] latents)
        {
            for (int first = 0; first + size <= latents.Length; first += size)
            {
                tokens.Add(0);
                audio.Add(true);
                patches.Add(latents.AsSpan(first, size).ToArray());
            }
        }

        if (reference is not null)
        {
            tokens.Add(_referenceStart);
            audio.Add(false);
            AddPatches(reference);
            tokens.Add(_referenceEnd);
            audio.Add(false);
        }

        foreach (int token in _text.Encode((continuation?.Text ?? "") + target))
        {
            tokens.Add(token);
            audio.Add(false);
        }

        tokens.Add(_audioStart);
        audio.Add(false);
        int continuationPatches = 0;
        if (continuation is { } prompt)
        {
            int before = patches.Count;
            AddPatches(prompt.Latents);
            continuationPatches = patches.Count - before;
        }

        return new VoxCpm2Prompt([.. tokens], [.. audio], patches, _text.Encode(target).Length, continuationPatches);
    }

    /// <summary>The text a request speaks: normalized (when the options ask), whitespace folded, and the voice description of <see cref="SpeechSynthesisOptions.Instructions"/> in parentheses before it.</summary>
    /// <param name="text">The request's text.</param>
    /// <param name="options">The request's options, or <see langword="null"/>.</param>
    public static string TargetText(string text, SpeechSynthesisOptions? options)
    {
        string target = VoxCpm2Text.Fold(options.NormalizedText(text)).Trim();
        return options?.Instructions is { Length: > 0 } instructions ? $"({instructions.Trim()}){target}" : target;
    }

    /// <summary>
    /// Generates the patches of a request one by one, each with the audio the AudioVAE decoded from it.
    /// </summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Stops generation between patches.</param>
    /// <param name="forcing">Replaces the diffusion's noise or patches, or the stop decision (teacher forcing for verification), or <see langword="null"/>.</param>
    /// <param name="observe">Receives each step's states, stop logits and patch, or <see langword="null"/>.</param>
    public IEnumerable<VoxCpm2Patch> GeneratePatches(string text, SpeechSynthesisOptions? options, CancellationToken cancellationToken, VoxCpm2Forcing? forcing = null,
        Action<VoxCpm2Step>? observe = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        ObjectDisposedException.ThrowIf(_disposed, this);
        string target = TargetText(text, options);
        if (target.Length == 0)
        {
            yield break;
        }

        var (reference, continuation) = ResolveVoice(options, cancellationToken);
        var prompt = BuildPrompt(target, reference, continuation);
        var baseModel = _base ??= CreateSession(WellKnown.DecodeMultimodalGraph);
        var residualModel = _residual ??= CreateSession(VoxCpm2Adapter.ResidualGraph);
        baseModel.Reset();
        residualModel.Reset();
        int rows = prompt.Tokens.Length;
        int h = _hidden;

        // The prompt: the local encoder's rows of the audio rows, the text-semantic model over all rows, the junction, the residual model over its rows.
        var features = EncodePatches(prompt.Patches);
        var featureRows = new float[rows * h];
        var mask = new float[rows];
        for (int row = 0, audio = 0; row < rows; row++)
        {
            if (prompt.Audio[row])
            {
                features.AsSpan(audio++ * h, h).CopyTo(featureRows.AsSpan(row * h, h));
                mask[row] = 1f;
            }
        }

        var hidden = ProcessRows(baseModel, prompt.Tokens, prompt.Audio, features);
        var (lm, residualInput) = Fuse(hidden, featureRows, mask, rows);
        var residual = ProcessRows(residualModel, new int[rows], [.. Enumerable.Repeat(true, rows)], residualInput);
        var lmHidden = lm.AsSpan((rows - 1) * h, h).ToArray();
        var residualHidden = residual.AsSpan((rows - 1) * h, h).ToArray();

        int size = _patch * _feature;
        var previous = prompt.Audio[^1] ? prompt.Patches[^1] : new float[size];
        var decoder = _vaeDecoder ??= _model.CompileComponent(VoxCpm2Adapter.VaeDecoderRole, VoxCpm2Adapter.VaeDecodeGraph);
        var caches = _decoderCaches ??= new VibeVoiceConvCaches(decoder, AudioVaeV2Graph.DecoderCachePrefix);
        caches.Reset();
        // As the reference: with a continuation prompt the decoder first sees its last audio patches (at most ContextPatches), whose audio is dropped.
        for (int i = prompt.ContinuationPatches > 0 ? Math.Max(0, prompt.Patches.Count - ContextPatches) : prompt.Patches.Count; i < prompt.Patches.Count; i++)
        {
            Decode(prompt.Patches[i]);
        }

        int steps = options?.DiffusionSteps ?? _defaultSteps;
        float guidance = options?.GuidanceScale ?? _defaultGuidance;
        int seed = options?.Sampling.Seed ?? Random.Shared.Next();
        int budget = Math.Min((int)(prompt.TargetTokens * LengthRatio) + LengthSlack, forcing?.MaxSteps ?? MaxPatches);
        budget = forcing?.Steps ?? Math.Min(budget, Math.Min(baseModel.ContextLength, residualModel.ContextLength) - rows);
        for (int step = 0; step < budget; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (mu, stop) = Condition(lmHidden, residualHidden);
            var noise = forcing?.Noise is { } given && step < given.Count ? Transposed(given[step], _feature, _patch) : SpeechNoise.Gaussian(size, unchecked(seed + step));
            var patch = Diffuse(mu, previous, noise, steps, guidance);
            var used = forcing?.Patches is { } fed && step < fed.Count ? fed[step] : patch;
            var embedding = EncodePatches([used]);
            observe?.Invoke(new VoxCpm2Step(step, lmHidden, residualHidden, stop, patch, embedding));
            yield return new VoxCpm2Patch(used, Decode(used));
            if (forcing?.Steps is null && step > MinPatches && stop[1] > stop[0])
            {
                yield break;
            }

            if (step + 1 == budget)
            {
                yield break;
            }

            var next = ProcessRows(baseModel, [0], [true], embedding);
            var (quantized, nextResidual) = Fuse(next, embedding, [1f], 1);
            lmHidden = quantized;
            residualHidden = ProcessRows(residualModel, [0], [true], nextResidual);
            previous = used;
        }
    }

    /// <summary>
    /// The latent patches of 16 kHz audio ([patches · patch, feature]): padded with silence to whole patches (at the start for a continuation prompt, at the end for a
    /// reference, as the reference's <c>_encode_wav</c>), then the AudioVAE encoder's means.
    /// </summary>
    /// <param name="samples">The audio at the model's input rate.</param>
    /// <param name="padStart">Whether the padding goes before the audio.</param>
    /// <param name="cancellationToken">Stops between runs of the encoder.</param>
    public float[] EncodeAudio(float[] samples, bool padStart, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(samples);
        int patchSamples = _patch * _encoderHop;
        int padded = (samples.Length + patchSamples - 1) / patchSamples * patchSamples;
        var audio = new float[padded];
        samples.CopyTo(audio, padStart ? padded - samples.Length : 0);
        var encoder = _vaeEncoder ??= _model.CompileComponent(VoxCpm2Adapter.VaeEncoderRole, VoxCpm2Adapter.VaeEncodeGraph);
        var caches = _encoderCaches ??= new VibeVoiceConvCaches(encoder, AudioVaeV2Graph.EncoderCachePrefix);
        caches.Reset();
        int frames = padded / _encoderHop;
        int block = VoxCpm2Adapter.EncodeLatentFrames;
        var input = encoder.Graph.Inputs.First(i => i.Input!.Name == AudioVaeV2Graph.AudioInput);
        var latents = new float[frames * _feature];
        var output = new float[block * _feature];
        for (int first = 0; first < frames; first += block)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var chunk = new float[block * _encoderHop];
            audio.AsSpan(first * _encoderHop, Math.Min(chunk.Length, padded - (first * _encoderHop))).CopyTo(chunk);
            encoder.SetInput<float>(input, chunk);
            caches.Bind();
            encoder.Run(new RunParameters());
            encoder.GetOutput<float>(AudioVaeV2Graph.LatentsName, output);
            caches.Collect();
            output.AsSpan(0, Math.Min(block, frames - first) * _feature).CopyTo(latents.AsSpan(first * _feature));
        }

        return latents;
    }

    /// <summary>Decodes latent frames ([frames, feature], a whole number of patches) to output-rate audio from a fresh decoder state (decoder verification).</summary>
    /// <param name="latents">The latents.</param>
    public float[] DecodeLatents(float[] latents)
    {
        ArgumentNullException.ThrowIfNull(latents);
        var decoder = _vaeDecoder ??= _model.CompileComponent(VoxCpm2Adapter.VaeDecoderRole, VoxCpm2Adapter.VaeDecodeGraph);
        (_decoderCaches ??= new VibeVoiceConvCaches(decoder, AudioVaeV2Graph.DecoderCachePrefix)).Reset();
        int size = _patch * _feature;
        var audio = new List<float>();
        for (int first = 0; first + size <= latents.Length; first += size)
        {
            audio.AddRange(Decode(latents.AsSpan(first, size).ToArray()));
        }

        return [.. audio];
    }

    /// <summary>The local encoder's language model rows ([patches · hidden]) of latent patches.</summary>
    /// <param name="patches">The patches ([patch, feature] row-major each).</param>
    public float[] EncodePatches(IReadOnlyList<float[]> patches)
    {
        ArgumentNullException.ThrowIfNull(patches);
        var result = new float[patches.Count * _hidden];
        if (patches.Count == 0)
        {
            return result;
        }

        var encoder = _encoder ??= _model.CompileComponent(VoxCpm2Adapter.EncoderRole, VoxCpm2Graphs.EncodeGraph);
        var inputs = encoder.Graph.Inputs.ToDictionary(i => i.Input!.Name, i => i, StringComparer.Ordinal);
        int sequence = _patch + 1;
        for (int first = 0; first < patches.Count; first += VoxCpm2Adapter.MaxEncodePatches)
        {
            int count = Math.Min(VoxCpm2Adapter.MaxEncodePatches, patches.Count - first);
            int rows = count * sequence;
            var values = new float[rows * _feature];
            var classMask = new float[rows];
            var positions = new int[rows];
            var spans = new int[rows * 2];
            var classRows = new int[count];
            for (int p = 0; p < count; p++)
            {
                int start = p * sequence;
                classMask[start] = 1f;
                classRows[p] = start;
                patches[first + p].CopyTo(values, (start + 1) * _feature);
                for (int r = 0; r < sequence; r++)
                {
                    positions[start + r] = r;
                    spans[(2 * (start + r)) + 0] = start;
                    spans[(2 * (start + r)) + 1] = sequence;
                }
            }

            encoder.SetInput<float>(inputs["rows"], values);
            encoder.SetInput<float>(inputs["class_mask"], classMask);
            encoder.SetInput<int>(inputs["positions"], positions);
            encoder.SetInput<int>(inputs["spans"], spans);
            encoder.SetInput<int>(inputs["class_rows"], classRows);
            var parameters = new RunParameters();
            parameters.Symbols[WellKnown.TokenSymbol] = rows;
            parameters.Symbols[VoxCpm2Graphs.PatchSymbol] = count;
            encoder.Run(parameters);
            encoder.GetOutput<float>("features", result.AsSpan(first * _hidden, count * _hidden));
        }

        return result;
    }

    /// <summary>
    /// The flow-matching sampler of one patch (the reference's <c>UnifiedCFM.solve_euler</c>): from <paramref name="noise"/> at t = 1 to t = 0 over <paramref name="steps"/>
    /// Euler steps of the sway-sampled schedule, the velocity guided between the conditional and the unconditional (zero condition) run with the zero-star rescaling, the
    /// first step's velocity zero.
    /// </summary>
    /// <param name="mu">The condition [2 · width] (the two language models' projections).</param>
    /// <param name="previous">The previous patch ([patch, feature]; zeros before the first).</param>
    /// <param name="noise">The start noise ([patch, feature]).</param>
    /// <param name="steps">The Euler steps.</param>
    /// <param name="guidance">The guidance scale.</param>
    public float[] Diffuse(float[] mu, float[] previous, float[] noise, int steps, float guidance)
    {
        var dit = _dit ??= _model.CompileComponent(VoxCpm2Adapter.DitRole, VoxCpm2Graphs.DitGraph);
        var inputs = dit.Graph.Inputs.ToDictionary(i => i.Input!.Name, i => i, StringComparer.Ordinal);
        int size = _patch * _feature;
        var muRows = new float[2 * mu.Length];
        mu.CopyTo(muRows, 0);
        dit.SetInput<float>(inputs["mu"], muRows);
        dit.SetInput<float>(inputs["cond"], previous);
        var schedule = new float[steps + 1];
        for (int k = 0; k <= steps; k++)
        {
            float t = 1f - ((float)k / steps);
            schedule[k] = t + (MathF.Cos(MathF.PI / 2 * t) - 1f + t);
        }

        var x = (float[])noise.Clone();
        var velocity = new float[2 * size];
        float time = schedule[0];
        float dt = schedule[0] - schedule[1];
        int zeroSteps = Math.Max(1, (int)((steps + 1) * 0.04));
        for (int step = 1; step <= steps; step++)
        {
            if (step > zeroSteps)
            {
                dit.SetInput<float>(inputs["x"], x);
                dit.SetInput<float>(inputs["time"], TimeEmbedding(time));
                dit.SetInput<float>(inputs["delta"], TimeEmbedding(_meanMode ? dt : 0f));
                dit.Run(new RunParameters());
                dit.GetOutput<float>("velocity", velocity);
                double dot = 0;
                double norm = 0;
                for (int i = 0; i < size; i++)
                {
                    dot += (double)velocity[i] * velocity[size + i];
                    norm += (double)velocity[size + i] * velocity[size + i];
                }

                float scale = (float)(dot / (norm + 1e-8));
                for (int i = 0; i < size; i++)
                {
                    float unconditional = velocity[size + i] * scale;
                    x[i] -= dt * (unconditional + (guidance * (velocity[i] - unconditional)));
                }
            }

            time -= dt;
            if (step < steps)
            {
                dt = time - schedule[step + 1];
            }
        }

        return x;
    }

    /// <summary>The sinusoidal embedding of a diffusion time (the reference's <c>SinusoidalPosEmb</c>, scale 1000): sines then cosines of <c>1000 · t · exp(-ln(10⁴) · k / (half - 1))</c>.</summary>
    /// <param name="time">The time.</param>
    public float[] TimeEmbedding(float time)
    {
        int half = _ditWidth / 2;
        float exponent = (float)(Math.Log(10000) / (half - 1));
        var embedding = new float[_ditWidth];
        for (int k = 0; k < half; k++)
        {
            float angle = 1000f * time * MathF.Exp(k * -exponent);
            embedding[k] = MathF.Sin(angle);
            embedding[half + k] = MathF.Cos(angle);
        }

        return embedding;
    }

    // The diffusion condition and the stop logits of a step.
    private (float[] Mu, float[] Stop) Condition(float[] lm, float[] residual)
    {
        var condition = _condition ??= _model.CompileComponent(VoxCpm2Adapter.ProjectionRole, VoxCpm2Graphs.ConditionGraph);
        var inputs = condition.Graph.Inputs.ToDictionary(i => i.Input!.Name, i => i, StringComparer.Ordinal);
        condition.SetInput<float>(inputs["lm"], lm);
        condition.SetInput<float>(inputs["residual"], residual);
        condition.Run(new RunParameters());
        var mu = new float[2 * _ditWidth];
        var stop = new float[2];
        condition.GetOutput<float>("mu", mu);
        condition.GetOutput<float>("stop", stop);
        return (mu, stop);
    }

    // The junction over rows: the quantized text-semantic states and the residual model's input rows.
    private (float[] Lm, float[] ResidualInput) Fuse(float[] hidden, float[] features, float[] mask, int rows)
    {
        var fuse = _fuse ??= _model.CompileComponent(VoxCpm2Adapter.ProjectionRole, VoxCpm2Graphs.FuseGraph);
        var inputs = fuse.Graph.Inputs.ToDictionary(i => i.Input!.Name, i => i, StringComparer.Ordinal);
        int h = _hidden;
        var lm = new float[rows * h];
        var residual = new float[rows * h];
        for (int first = 0; first < rows; first += VoxCpm2Adapter.MaxFuseRows)
        {
            int count = Math.Min(VoxCpm2Adapter.MaxFuseRows, rows - first);
            fuse.SetInput<float>(inputs["hidden"], hidden.AsSpan(first * h, count * h));
            fuse.SetInput<float>(inputs["features"], features.AsSpan(first * h, count * h));
            fuse.SetInput<float>(inputs["audio_mask"], mask.AsSpan(first, count));
            var parameters = new RunParameters();
            parameters.Symbols[WellKnown.TokenSymbol] = count;
            fuse.Run(parameters);
            fuse.GetOutput<float>("lm", lm.AsSpan(first * h, count * h));
            fuse.GetOutput<float>("residual_input", residual.AsSpan(first * h, count * h));
        }

        return (lm, residual);
    }

    // One patch through the AudioVAE decoder (its caches continue the signal).
    private float[] Decode(float[] patch)
    {
        var decoder = _vaeDecoder!;
        decoder.SetInput<float>(decoder.Graph.Inputs.First(i => i.Input!.Name == AudioVaeV2Graph.LatentsName), patch);
        _decoderCaches!.Bind();
        decoder.Run(new RunParameters());
        var audio = new float[SamplesPerPatch];
        decoder.GetOutput<float>(AudioVaeV2Graph.AudioOutput, audio);
        _decoderCaches.Collect();
        return audio;
    }

    // The request's voice: a saved voice or a reference recording (with its transcript, also the continuation prompt).
    private (float[]? Reference, (string Text, float[] Latents)? Continuation) ResolveVoice(SpeechSynthesisOptions? options, CancellationToken cancellationToken)
    {
        if (options?.VoiceProfile is { } profile)
        {
            if (profile.Architecture != VoxCpm2Adapter.Family)
            {
                throw new ArgumentException($"The voice '{profile.Name}' belongs to '{profile.Architecture}', not to VoxCPM2.");
            }

            var reference = profile.Tensors.TryGetValue(ReferenceKey, out var r) ? r.ToFloats() : null;
            (string, float[])? continuation = profile.ReferenceText is { Length: > 0 } transcript && profile.Tensors.TryGetValue(PromptKey, out var c) ? (transcript, c.ToFloats()) : null;
            return (reference, continuation);
        }

        if (options?.ReferenceAudio is { } recording)
        {
            var samples = recording.ToModelInput(_inputRate).Samples.ToArray();
            var reference = EncodeAudio(samples, padStart: false, cancellationToken);
            (string, float[])? continuation = string.IsNullOrWhiteSpace(options.ReferenceText) ? null : (options.ReferenceText, EncodeAudio(samples, padStart: true, cancellationToken));
            return (reference, continuation);
        }

        return (null, null);
    }

    // Rows through a language model session (text rows by token, the others from the given rows in order), in passes of at most the session's batch; the hidden states of all rows.
    private float[] ProcessRows(InferenceSession session, int[] tokens, bool[] embedded, float[] embeddings)
    {
        int h = _hidden;
        int rows = tokens.Length;
        var hidden = new float[rows * h];
        int chunk = Math.Max(1, session.MaxBatchTokens);
        int embeddedBefore = 0;
        for (int first = 0; first < rows; first += chunk)
        {
            int count = Math.Min(chunk, rows - first);
            int position = session.RopePosition;
            var index = new int[count];
            var row = new int[count];
            for (int i = 0; i < count; i++)
            {
                if (embedded[first + i])
                {
                    row[i] = embeddedBefore++;
                }
                else
                {
                    index[i] = -1;
                }
            }

            var prompt = new MultimodalPrompt
            {
                Tokens = tokens.AsSpan(first, count).ToArray(),
                ImageIndex = index,
                ImageRow = row,
                Positions = [.. Enumerable.Range(position, count)],
                PositionAxes = 1,
                NextPosition = position + count,
            };
            session.Process(prompt, [embeddings]);
            session.LastHidden.CopyTo(hidden.AsSpan(first * h, count * h));
        }

        return hidden;
    }

    private static float[] Transposed(float[] values, int rows, int columns)
    {
        var result = new float[values.Length];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                result[(c * rows) + r] = values[(r * columns) + c];
            }
        }

        return result;
    }

    private InferenceSession CreateSession(string graph)
    {
        var session = _llm.CreateSession(graph);
        session.CaptureHidden = true;
        return session;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _base?.Dispose();
        _residual?.Dispose();
        _encoder?.Dispose();
        _dit?.Dispose();
        _fuse?.Dispose();
        _condition?.Dispose();
        _vaeDecoder?.Dispose();
        _vaeEncoder?.Dispose();
        _gate.Dispose();
    }
}
