using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Sharplight.Audio;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Speech;
using Sharplight.Tasks;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The <see cref="ISpeechSynthesizer"/> of OmniVoice (<see cref="OmniVoiceAdapter"/>), following the official package's <c>generate</c>: the prompt is the style tokens (a denoise marker with
/// a voice-cloning reference, the language, the voice-design instruction), the text (the reference transcript first when cloning), the reference's codes and as many masked target
/// frames as the rule-based length estimate gives; the bidirectional language model unmasks the eight codebooks of the target in <see cref="OmniVoiceGenerationSettings.Steps"/> steps,
/// each time revealing the positions it is most confident about (scores penalized by codebook index) under classifier-free guidance against the target alone; the tokenizer's decoder
/// turns the codes into 24 kHz audio, which is trimmed of long silences, matched to the reference's loudness and faded. Long text is cut at punctuation and spoken chunk by chunk.
/// </summary>
internal sealed partial class OmniVoiceSynthesizer : ISpeechSynthesizer, IDisposable
{
    private const double LongTextSeconds = 30;
    private const double ChunkSeconds = 15;

    private readonly TaskModel _model;
    private readonly LlmModel _llm;
    private readonly int _codebooks;
    private readonly int _audioVocab;
    private readonly int _maskId;
    private readonly int _frameRate;
    private readonly int _samplesPerFrame;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly Lazy<Tables> _tables;
    private CompiledModel? _denoise;
    private CompiledModel? _encoder;
    private CompiledModel? _decoder;
    private bool _disposed;

    /// <summary>Creates the synthesizer; nothing is compiled until the first request.</summary>
    /// <param name="model">A loaded OmniVoice model.</param>
    public OmniVoiceSynthesizer(TaskModel model)
    {
        _model = model;
        _llm = LlmModel.FromTaskModel(model);
        var text = model.Description.Component(OmniVoiceAdapter.TextModelRole);
        _codebooks = text.ConfigValue("codebooks", 8);
        _audioVocab = text.ConfigValue("audioVocab", 1025);
        _maskId = text.ConfigValue("maskId", 1024);
        _frameRate = text.ConfigValue("frameRate", 25);
        var codec = model.Description.Component(OmniVoiceAdapter.CodecRole);
        SampleRate = codec.ConfigValue(AudioCodecGraphs.SampleRateKey, 24000);
        _samplesPerFrame = codec.ConfigValue(AudioCodecGraphs.SamplesPerFrameKey, 960);
        _tables = new Lazy<Tables>(() => new Tables(model.Definition, _codebooks, _audioVocab));
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    /// <inheritdoc/>
    public TaskClientMetadata Metadata { get; }

    /// <inheritdoc/>
    public int SampleRate { get; }

    /// <summary>OmniVoice has no built-in speakers: it speaks with the voice of a reference recording, a designed voice (<see cref="SpeechSynthesisOptions.Instructions"/>) or one it chooses.</summary>
    public IReadOnlyList<string> Voices => [];

    /// <summary>The languages a request can name (<see cref="SpeechSynthesisOptions.Language"/>), as ids; names are accepted too (<see cref="LanguageNames"/>).</summary>
    public IReadOnlyList<string> Languages => OmniVoiceLanguages.Ids;

    /// <summary>The language names a request can use.</summary>
    public IReadOnlyList<string> LanguageNames => OmniVoiceLanguages.Names;

    /// <summary>Whether <paramref name="model"/> is an OmniVoice model (<see cref="OmniVoiceAdapter"/>).</summary>
    /// <param name="model">A loaded speech-synthesis model.</param>
    public static bool Handles(TaskModel model)
        => model.Description.TryGetComponent(OmniVoiceAdapter.TextModelRole, out var text) && text.ConfigValue("family", "") == OmniVoiceAdapter.Family
           && model.Description.TryGetComponent(OmniVoiceAdapter.CodecRole, out _);

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
        ArgumentException.ThrowIfNullOrWhiteSpace(text);
        string normalized = options.NormalizedText(text);
        var audio = await BackgroundCompute.RunAsync(_gate, () => Synthesize(normalized, options, cancellationToken), cancellationToken).ConfigureAwait(false);
        return new SpeechSynthesisResult(audio, options?.Voice ?? "");
    }

    /// <inheritdoc/>
    /// <remarks>Generation unmasks a whole utterance at once, so the audio arrives as one buffer when it is ready.</remarks>
    public IAsyncEnumerable<AudioBuffer> SynthesizeStreamingAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
        => BackgroundCompute.StreamAsync(token => StreamCoreAsync(text, options, token), cancellationToken);

    private async IAsyncEnumerable<AudioBuffer> StreamCoreAsync(string text, SpeechSynthesisOptions? options, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var result = await SynthesizeAsync(text, options, cancellationToken).ConfigureAwait(false);
        yield return result.Audio;
    }

    /// <summary>Speaks a text: reference, chunking, unmasking, decoding and post-processing.</summary>
    private AudioBuffer Synthesize(string text, SpeechSynthesisOptions? options, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        string? language = OmniVoiceLanguages.Resolve(options?.Language);
        string? instruct = OmniVoiceInstruct.Resolve(options?.Instructions, OmniVoiceInstruct.HasIdeograph(text));
        OmniVoiceReference? reference = null;
        if (options?.ReferenceAudio is { } recording)
        {
            string transcript = options.ReferenceText ?? options.VoiceProfile?.ReferenceText
                ?? throw new ArgumentException("OmniVoice clones a voice from a recording together with its transcript (SpeechSynthesisOptions.ReferenceText).", nameof(options));
            reference = EncodeReference(recording, transcript, cancellationToken);
        }

        var sampling = options?.Sampling;
        bool greedy = sampling is not null && sampling with { Seed = null } != new Llm.SamplingOptions() && sampling.Temperature <= 0f;
        var settings = (greedy ? OmniVoiceGenerationSettings.Greedy : new OmniVoiceGenerationSettings()) with { Seed = sampling?.Seed };
        float speed = options?.Speed ?? 1f;
        var chunks = GenerateChunks(text, reference, language, instruct, speed, settings, cancellationToken);
        var decoded = chunks.Select(codes => Decode(codes, cancellationToken)).ToList();
        var audio = decoded.Count == 1 ? decoded[0] : OmniVoiceAudio.CrossFade(decoded, SampleRate);
        return new AudioBuffer(PostProcess(audio, reference?.Rms), SampleRate);
    }

    /// <summary>The target codes of the text, one array [codebooks, frames] per chunk (long text is spoken in chunks of about <c>15</c> s).</summary>
    private List<int[]> GenerateChunks(string text, OmniVoiceReference? reference, string? language, string? instruct, float speed, OmniVoiceGenerationSettings settings,
        CancellationToken cancellationToken)
    {
        int target = EstimateFrames(text, reference?.Text, reference?.Frames, speed);
        if (target <= LongTextSeconds * _frameRate)
        {
            var prompt = BuildPrompt(text, reference, language, instruct, target, settings.Denoise);
            return [GenerateTokens(prompt, settings, null, cancellationToken)];
        }

        double perCharacter = (double)target / text.Length;
        int chunkLength = (int)(ChunkSeconds * _frameRate / perCharacter);
        var texts = OmniVoiceText.Chunk(text, chunkLength, 3);
        var results = new List<int[]>();
        OmniVoiceReference? current = reference;
        for (int i = 0; i < texts.Count; i++)
        {
            int frames = EstimateFrames(texts[i], current?.Text, current?.Frames, speed);
            var prompt = BuildPrompt(texts[i], current, language, instruct, frames, settings.Denoise);
            var codes = GenerateTokens(prompt, settings, null, cancellationToken);
            results.Add(codes);
            if (reference is null && i == 0)
            {
                // Without a reference the first chunk's voice is the reference of the others.
                current = new OmniVoiceReference(codes, codes.Length / _codebooks, texts[0], 0f);
            }
        }

        return results;
    }

    private int EstimateFrames(string text, string? referenceText, int? referenceFrames, float speed)
    {
        string reference = referenceText is { Length: > 0 } && referenceFrames is > 0 ? referenceText : "Nice to meet you.";
        int frames = referenceFrames is > 0 && referenceText is { Length: > 0 } ? referenceFrames.Value : 25;
        double estimate = OmniVoiceDurationEstimator.Estimate(text, reference, frames);
        if (speed > 0 && speed != 1f)
        {
            estimate /= speed;
        }

        return Math.Max(1, (int)estimate);
    }

    /// <summary>The voice-cloning reference of a recording: the codes of its first frames (at most the codec's limit) and its transcript with an end mark.</summary>
    /// <param name="recording">The recording (any rate, any channels).</param>
    /// <param name="transcript">What is said in it.</param>
    /// <param name="cancellationToken">Stops the encoding.</param>
    public OmniVoiceReference EncodeReference(AudioBuffer recording, string transcript, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(recording);
        ArgumentNullException.ThrowIfNull(transcript);
        var samples = recording.ToModelInput(SampleRate).Samples.ToArray();
        double sum = 0;
        foreach (float s in samples)
        {
            sum += (double)s * s;
        }

        float rms = samples.Length == 0 ? 0f : (float)Math.Sqrt(sum / samples.Length);
        if (rms is > 0f and < 0.1f)
        {
            float gain = 0.1f / rms;
            for (int i = 0; i < samples.Length; i++)
            {
                samples[i] *= gain;
            }
        }

        samples = OmniVoiceAudio.RemoveSilence(samples, SampleRate, 200, 100, 200);
        return EncodeCodes(samples, transcript, rms, cancellationToken);
    }

    /// <summary>The reference of already prepared 24 kHz samples (no silence removal or boost): its codes and the transcript with an end mark.</summary>
    /// <param name="samples">Mono samples at the codec's rate.</param>
    /// <param name="transcript">What is said in them.</param>
    /// <param name="rms">The loudness of the original recording.</param>
    /// <param name="cancellationToken">Stops the encoding.</param>
    public OmniVoiceReference EncodeCodes(float[] samples, string transcript, float rms, CancellationToken cancellationToken = default)
    {
        int frames = Math.Min(samples.Length / _samplesPerFrame, HiggsAudioCodecAdapter.MaxFrames);
        if (frames == 0)
        {
            throw new ArgumentException("The reference recording is empty (after removing its silence).", nameof(samples));
        }

        cancellationToken.ThrowIfCancellationRequested();
        var encoder = _encoder ??= _model.CompileComponent(OmniVoiceAdapter.CodecEncoderRole, AudioCodecGraphs.EncodeGraph);
        var audio = samples.AsSpan(0, frames * _samplesPerFrame).ToArray();
        encoder.SetInput<float>(encoder.Graph.Inputs.First(v => v.Input!.Name == AudioCodecGraphs.AudioInput), audio);
        var parameters = new RunParameters();
        parameters.Symbols[WellKnown.TokenSymbol] = audio.Length;
        encoder.Run(parameters);
        var codes = new int[_codebooks * frames];
        var part = new int[frames];
        for (int q = 0; q < _codebooks; q++)
        {
            encoder.GetOutput<int>(AudioCodecGraphs.CodesName(q), part);
            part.CopyTo(codes.AsSpan(q * frames, frames));
        }

        return new OmniVoiceReference(codes, frames, OmniVoiceText.AddPunctuation(transcript), rms);
    }

    /// <summary>The style tokens of a request: <c>&lt;|denoise|&gt;</c> (voice cloning), the language and the instruction.</summary>
    /// <param name="denoise">Whether the prompt carries the denoise marker.</param>
    /// <param name="language">The language id, or <see langword="null"/>.</param>
    /// <param name="instruct">The normalized voice-design instruction, or <see langword="null"/>.</param>
    public int[] StyleTokens(bool denoise, string? language, string? instruct)
    {
        string style = (denoise ? "<|denoise|>" : "") + $"<|lang_start|>{language ?? "None"}<|lang_end|><|instruct_start|>{instruct ?? "None"}<|instruct_end|>";
        return _llm.Tokenizer.Encode(style, addSpecial: false, parseSpecial: true);
    }

    /// <summary>The text tokens: <c>&lt;|text_start|&gt;</c>, the text, <c>&lt;|text_end|&gt;</c>; each non-verbal tag (<c>[laughter]</c>...) is tokenized on its own.</summary>
    /// <param name="combinedText">The reference transcript and the target text, as <see cref="OmniVoiceText.Combine"/> joins them.</param>
    public int[] TextTokens(string combinedText)
    {
        string wrapped = $"<|text_start|>{combinedText}<|text_end|>";
        var ids = new List<int>();
        int last = 0;
        foreach (Match match in NonVerbalTag().Matches(wrapped))
        {
            if (match.Index > last)
            {
                ids.AddRange(_llm.Tokenizer.Encode(wrapped[last..match.Index], addSpecial: false, parseSpecial: true));
            }

            ids.AddRange(_llm.Tokenizer.Encode(match.Value, addSpecial: false, parseSpecial: true));
            last = match.Index + match.Length;
        }

        if (last < wrapped.Length)
        {
            ids.AddRange(_llm.Tokenizer.Encode(wrapped[last..], addSpecial: false, parseSpecial: true));
        }

        return [.. ids];
    }

    [GeneratedRegex(@"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|surprise-yo|dissatisfaction-hnn)\]")]
    private static partial Regex NonVerbalTag();

    /// <summary>Builds the conditional input of a request.</summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="reference">The reference to clone, or <see langword="null"/>.</param>
    /// <param name="language">The language id, or <see langword="null"/>.</param>
    /// <param name="instruct">The normalized instruction, or <see langword="null"/>.</param>
    /// <param name="targetFrames">The number of frames to generate.</param>
    /// <param name="denoise">Whether the prompt carries the denoise marker (when there is a reference).</param>
    public OmniVoicePrompt BuildPrompt(string text, OmniVoiceReference? reference, string? language, string? instruct, int targetFrames, bool denoise = true)
    {
        var style = StyleTokens(denoise && reference is not null, language, instruct);
        var words = TextTokens(OmniVoiceText.Combine(text, reference?.Text));
        int referenceFrames = reference?.Frames ?? 0;
        int length = style.Length + words.Length + referenceFrames + targetFrames;
        var ids = new int[_codebooks * length];
        var mask = new bool[length];
        for (int q = 0; q < _codebooks; q++)
        {
            var row = ids.AsSpan(q * length, length);
            style.CopyTo(row);
            words.CopyTo(row[style.Length..]);
            int audioStart = style.Length + words.Length;
            if (reference is not null)
            {
                reference.Codes.AsSpan(q * referenceFrames, referenceFrames).CopyTo(row[audioStart..]);
            }

            row[(audioStart + referenceFrames)..].Fill(_maskId);
        }

        mask.AsSpan(style.Length + words.Length).Fill(true);
        return new OmniVoicePrompt(ids, mask, targetFrames, referenceFrames, _codebooks);
    }

    /// <summary>
    /// The language model's logits for a prompt with the given target codes: [2 · rows, codebooks · audio vocabulary] where the first <c>length</c> rows are the conditional sequence
    /// and the next <c>target</c> rows the unconditional one (the target alone).
    /// </summary>
    /// <param name="prompt">The prompt.</param>
    /// <param name="target">The target codes [codebooks, frames] (the mask id where nothing is revealed yet).</param>
    public float[] Forward(OmniVoicePrompt prompt, ReadOnlySpan<int> target)
    {
        var tables = _tables.Value;
        int h = tables.Hidden;
        int t = prompt.TargetLength;
        int c = prompt.Length;
        int rows = c + t;
        var embeddings = new float[(long)rows * h];
        var ids = prompt.Ids;
        for (int i = 0; i < c - t; i++)
        {
            var row = embeddings.AsSpan(i * h, h);
            if (prompt.AudioMask[i])
            {
                for (int q = 0; q < prompt.Codebooks; q++)
                {
                    Add(row, tables.AudioRow(q, ids[(q * c) + i]));
                }
            }
            else
            {
                tables.TextRow(ids[i], row);
            }
        }

        for (int i = 0; i < t; i++)
        {
            var row = embeddings.AsSpan((c - t + i) * h, h);
            for (int q = 0; q < prompt.Codebooks; q++)
            {
                Add(row, tables.AudioRow(q, target[(q * t) + i]));
            }

            row.CopyTo(embeddings.AsSpan((c + i) * h, h));
        }

        var positions = new int[rows];
        var spans = new int[2 * rows];
        for (int i = 0; i < c; i++)
        {
            positions[i] = i;
            spans[2 * i] = 0;
            spans[(2 * i) + 1] = c;
        }

        for (int i = 0; i < t; i++)
        {
            positions[c + i] = i;
            spans[2 * (c + i)] = c;
            spans[(2 * (c + i)) + 1] = t;
        }

        var denoise = _denoise ??= _model.CompileComponent(OmniVoiceAdapter.TextModelRole, WellKnown.DenoiseGraph);
        denoise.SetInput<float>(denoise.Graph.Inputs.First(v => v.Input!.Name == "embeddings"), embeddings);
        denoise.SetInput<int>(denoise.Graph.Inputs.First(v => v.Input!.Name == PackedSequences.PositionsInput), positions);
        denoise.SetInput<int>(denoise.Graph.Inputs.First(v => v.Input!.Name == PackedSequences.RowSpansInput), spans);
        var parameters = new RunParameters();
        parameters.Symbols[WellKnown.TokenSymbol] = rows;
        denoise.Run(parameters);
        var logits = new float[(long)rows * _codebooks * _audioVocab];
        denoise.GetOutput<float>(WellKnown.LogitsOutput, logits);
        return logits;
    }

    /// <summary>The number of frames revealed at each step (the official schedule: a time-shifted linear one over all masked codes, rounded up, the last step taking the rest).</summary>
    /// <param name="targetFrames">The number of target frames.</param>
    /// <param name="settings">The settings.</param>
    public int[] Schedule(int targetFrames, OmniVoiceGenerationSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);
        int n = settings.Steps;
        var times = new double[n + 1];
        float step = 1f / n;
        for (int i = 0; i <= n; i++)
        {
            // torch.linspace in float32: forward from the start for the first half, backward from the end for the second.
            float value = i < (n + 1) / 2 ? step * i : 1f - (step * (n - i));
            float shifted = (float)settings.TimeShift * value / (1f + ((float)(settings.TimeShift - 1) * value));
            times[i] = shifted;
        }

        int total = targetFrames * _codebooks;
        int remaining = total;
        var schedule = new int[n];
        for (int s = 0; s < n; s++)
        {
            int count = s == n - 1 ? remaining : Math.Min((int)Math.Ceiling(total * (times[s + 1] - times[s])), remaining);
            schedule[s] = count;
            remaining -= count;
        }

        return schedule;
    }

    /// <summary>Unmasks the target of a prompt.</summary>
    /// <param name="prompt">The prompt.</param>
    /// <param name="settings">The settings.</param>
    /// <param name="observe">Receives the target codes before every step, or <see langword="null"/>.</param>
    /// <param name="cancellationToken">Stops between steps.</param>
    /// <returns>The target codes [codebooks, frames] row-major.</returns>
    public int[] GenerateTokens(OmniVoicePrompt prompt, OmniVoiceGenerationSettings settings, Action<int, int[]>? observe, CancellationToken cancellationToken)
    {
        int t = prompt.TargetLength;
        int cl = _codebooks;
        var state = new int[cl * t];
        state.AsSpan().Fill(_maskId);
        var schedule = Schedule(t, settings);
        var random = new Random(settings.Seed ?? Random.Shared.Next());
        int c = prompt.Length;
        int stride = _codebooks * _audioVocab;
        var scores = new float[cl * t];
        var predicted = new int[cl * t];
        for (int step = 0; step < settings.Steps; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            observe?.Invoke(step, (int[])state.Clone());
            int k = schedule[step];
            if (k <= 0)
            {
                continue;
            }

            var logits = Forward(prompt, state);
            Predict(logits, c, t, stride, settings, predicted, scores);
            for (int q = 0; q < cl; q++)
            {
                for (int i = 0; i < t; i++)
                {
                    int index = (q * t) + i;
                    float score = scores[index] - (q * settings.LayerPenalty);
                    if (settings.PositionTemperature > 0f)
                    {
                        double u = random.NextDouble();
                        score = (score / settings.PositionTemperature) + (float)-Math.Log((-Math.Log(u + 1e-10)) + 1e-10);
                    }

                    scores[index] = state[index] != _maskId ? float.NegativeInfinity : score;
                }
            }

            var order = Enumerable.Range(0, cl * t).ToArray();
            Array.Sort(order, (x, y) => scores[y] != scores[x] ? scores[y].CompareTo(scores[x]) : x.CompareTo(y));
            for (int j = 0; j < k && j < order.Length; j++)
            {
                if (float.IsNegativeInfinity(scores[order[j]]))
                {
                    break;
                }

                state[order[j]] = predicted[order[j]];
            }
        }

        return state;
    }

    // The guided prediction of every (codebook, position): argmax and max of log_softmax(log p_c + g · (log p_c − log p_u)) with the mask id excluded.
    private void Predict(float[] logits, int c, int t, int stride, OmniVoiceGenerationSettings settings, int[] predicted, float[] confidence)
    {
        int v = _audioVocab;
        var cond = new float[v];
        var uncond = new float[v];
        var mixed = new float[v];
        for (int q = 0; q < _codebooks; q++)
        {
            for (int i = 0; i < t; i++)
            {
                var conditional = logits.AsSpan(((c - t + i) * stride) + (q * v), v);
                LogSoftmax(conditional, cond);
                if (settings.GuidanceScale != 0f)
                {
                    LogSoftmax(logits.AsSpan(((c + i) * stride) + (q * v), v), uncond);
                    for (int j = 0; j < v; j++)
                    {
                        mixed[j] = cond[j] + (settings.GuidanceScale * (cond[j] - uncond[j]));
                    }

                    LogSoftmax(mixed, cond);
                }

                cond[_maskId] = float.NegativeInfinity;
                int best = 0;
                for (int j = 1; j < v; j++)
                {
                    if (cond[j] > cond[best])
                    {
                        best = j;
                    }
                }

                predicted[(q * t) + i] = best;
                confidence[(q * t) + i] = cond[best];
            }
        }
    }

    private static void LogSoftmax(ReadOnlySpan<float> x, Span<float> y)
    {
        float max = float.NegativeInfinity;
        foreach (float value in x)
        {
            max = MathF.Max(max, value);
        }

        float sum = 0f;
        foreach (float value in x)
        {
            sum += MathF.Exp(value - max);
        }

        float shift = max + MathF.Log(sum);
        for (int i = 0; i < x.Length; i++)
        {
            y[i] = x[i] - shift;
        }
    }

    /// <summary>Decodes target codes [codebooks, frames] into 24 kHz audio.</summary>
    /// <param name="codes">The codes.</param>
    /// <param name="cancellationToken">Stops between segments.</param>
    public float[] Decode(int[] codes, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(codes);
        int frames = codes.Length / _codebooks;
        var decoder = _decoder ??= _model.CompileComponent(OmniVoiceAdapter.CodecRole, AudioCodecGraphs.DecodeGraph);
        var mask = new float[_codebooks];
        mask.AsSpan().Fill(1f);
        decoder.SetInput<float>(decoder.Graph.Inputs.First(v => v.Input!.Name == AudioCodecGraphs.MaskInput), mask);
        var audio = new float[frames * _samplesPerFrame];
        int segment = HiggsAudioCodecAdapter.MaxFrames;
        for (int start = 0; start < frames; start += segment)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(segment, frames - start);
            for (int q = 0; q < _codebooks; q++)
            {
                decoder.SetInput<int>(decoder.Graph.Inputs.First(v => v.Input!.Name == AudioCodecGraphs.CodesName(q)), codes.AsSpan((q * frames) + start, count).ToArray());
            }

            var parameters = new RunParameters();
            parameters.Symbols[WellKnown.TokenSymbol] = count;
            decoder.Run(parameters);
            var part = new float[count * _samplesPerFrame];
            decoder.GetOutput<float>(AudioCodecGraphs.AudioOutput, part);
            part.CopyTo(audio.AsSpan(start * _samplesPerFrame));
        }

        return audio;
    }

    /// <summary>The finishing of decoded audio: long silences removed and the edges trimmed, the level matched to the reference (or peak-normalized without one), fades and padding.</summary>
    /// <param name="audio">The decoded audio.</param>
    /// <param name="referenceRms">The reference recording's loudness, or <see langword="null"/> without a reference.</param>
    /// <param name="cleanUp">Whether to remove silence and to fade and pad (the level is always matched).</param>
    public float[] PostProcess(float[] audio, float? referenceRms, bool cleanUp = true)
    {
        if (cleanUp)
        {
            audio = OmniVoiceAudio.RemoveSilence(audio, SampleRate, 500, 100, 100);
        }

        if (referenceRms is < 0.1f)
        {
            float gain = referenceRms.Value / 0.1f;
            for (int i = 0; i < audio.Length; i++)
            {
                audio[i] *= gain;
            }
        }
        else if (referenceRms is null)
        {
            float peak = audio.Length == 0 ? 0f : audio.Max(MathF.Abs);
            if (peak > 1e-6f)
            {
                float gain = 0.5f / peak;
                for (int i = 0; i < audio.Length; i++)
                {
                    audio[i] *= gain;
                }
            }
        }

        return cleanUp ? OmniVoiceAudio.FadeAndPad(audio, 0.1, 0.1, SampleRate) : audio;
    }

    private static void Add(Span<float> target, ReadOnlySpan<float> values)
    {
        for (int i = 0; i < target.Length; i++)
        {
            target[i] += values[i];
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
        _denoise?.Dispose();
        _encoder?.Dispose();
        _decoder?.Dispose();
        _gate.Dispose();
    }

    /// <summary>The host-side tables: the text embedding (rows read on demand) and the audio embedding.</summary>
    private sealed class Tables
    {
        private readonly WeightInfo _text;
        private readonly float[] _audio;
        private readonly int _vocab;

        public Tables(ModelDefinition definition, int codebooks, int audioVocab)
        {
            var w = definition.Weights;
            _text = w["token_embedding.weight"];
            Hidden = (int)_text.Shape[1].Fixed;
            _audio = WeightConversions.ReadFloats(w[OmniVoiceAdapter.Architecture + ".audio_embedding.weight"]);
            _vocab = audioVocab;
            if (_audio.Length != (long)codebooks * audioVocab * Hidden)
            {
                throw new InvalidDataException("OmniVoice: the audio embedding does not match the codebook count and vocabulary.");
            }
        }

        public int Hidden { get; }

        // The embedding of the code of a codebook.
        public ReadOnlySpan<float> AudioRow(int codebook, int code) => _audio.AsSpan(((codebook * _vocab) + code) * Hidden, Hidden);

        public void TextRow(int token, Span<float> destination)
        {
            if (_text.Type == DataType.F32)
            {
                _text.Source.Read((long)token * Hidden * sizeof(float), MemoryMarshal.AsBytes(destination));
            }
            else if (_text.Type == DataType.BF16 || _text.Type == DataType.F16)
            {
                var raw = new ushort[Hidden];
                _text.Source.Read((long)token * Hidden * sizeof(ushort), MemoryMarshal.AsBytes(raw.AsSpan()));
                for (int i = 0; i < raw.Length; i++)
                {
                    destination[i] = _text.Type == DataType.BF16 ? BitConverter.Int32BitsToSingle(raw[i] << 16) : (float)BitConverter.UInt16BitsToHalf(raw[i]);
                }
            }
            else
            {
                throw new NotSupportedException($"OmniVoice: text embedding of type {_text.Type} is not supported.");
            }
        }
    }
}
