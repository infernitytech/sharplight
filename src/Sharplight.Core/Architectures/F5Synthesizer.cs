using System.Text;
using System.Text.Json;
using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Architectures;

/// <summary>
/// The <see cref="ISpeechSynthesizer"/> of F5-TTS (<see cref="F5Adapter"/>), following the reference's <c>infer_process</c>: the reference recording
/// (<see cref="SpeechSynthesisOptions.ReferenceAudio"/>, RMS raised to 0.1 when quieter, 24 kHz) becomes the log-mel condition; the reference text plus
/// the text, one character per frame and filler after, is the text condition; the duration is the reference's frames plus the text's share of them
/// (by UTF-8 bytes, divided by the speed); 32 Euler steps over sway-sampled times integrate the guided field
/// <c>pred + 2 · (pred − pred without reference and text)</c> from Gaussian noise; the frames after the reference go through Vocos. One request runs at a time.
/// </summary>
internal sealed class F5Synthesizer : ISpeechSynthesizer, IVoiceCloner
{
    private const float TargetRms = 0.1f;

    private readonly TaskModel _model;
    private readonly IAudioFeatureExtractor _features;
    private readonly Dictionary<string, int> _characters;
    private readonly int _melBins;
    private readonly int _steps;
    private readonly float _guidance;
    private readonly float _sway;
    private readonly int _fftSize;
    private readonly int _hop;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private CompiledModel? _transformer;
    private CompiledModel? _vocoder;
    private bool _disposed;

    /// <summary>Creates the synthesizer; nothing is compiled until the first request.</summary>
    /// <param name="model">A loaded F5-TTS model.</param>
    public F5Synthesizer(TaskModel model)
    {
        _model = model;
        _features = AudioFeatureExtractors.Create(model);
        var characters = JsonSerializer.Deserialize(model.ReadAttachment(ModelComponentRoles.Tokenizer, "characters"), F5JsonContext.Default.StringArray) ?? [];
        _characters = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < characters.Length; i++)
        {
            _characters.TryAdd(characters[i], i);
        }

        var transformer = model.Description.Component(F5Adapter.TransformerRole);
        _melBins = transformer.ConfigValue("melBins", 100);
        _steps = transformer.ConfigValue("steps", 32);
        _guidance = (float)transformer.ConfigValue("guidance", 2.0);
        _sway = (float)transformer.ConfigValue("sway", -1.0);
        var vocoder = model.Description.Component(F5Adapter.VocoderRole);
        _fftSize = vocoder.ConfigValue("fftSize", 1024);
        _hop = vocoder.ConfigValue("hop", 256);
        SampleRate = model.Description.ConfigValue("sampleRate", 24000);
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    /// <inheritdoc/>
    public TaskClientMetadata Metadata { get; }

    /// <inheritdoc/>
    public int SampleRate { get; }

    /// <inheritdoc/>
    /// <remarks>None: the voice is the reference recording of each request.</remarks>
    public IReadOnlyList<string> Voices { get; } = [];

    /// <summary>Replaces the source of the starting noise (standard-normal values, <c>count</c> of them).</summary>
    public Func<int, float[]>? NoiseSource { get; set; }

    /// <summary>Whether <paramref name="model"/> is an F5-TTS model.</summary>
    /// <param name="model">A loaded speech-synthesis model.</param>
    public static bool Handles(TaskModel model)
        => model.Description.TryGetComponent(F5Adapter.TransformerRole, out var transformer) && transformer.Graphs.Contains(F5TransformerGraph.GraphName);

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
        var audio = await BackgroundCompute.RunAsync(_gate, () => Synthesize(text, options ?? SpeechSynthesisOptions.Default, cancellationToken), cancellationToken)
            .ConfigureAwait(false);
        return new SpeechSynthesisResult(audio, null);
    }

    /// <inheritdoc/>
    /// <remarks>The whole utterance as one buffer.</remarks>
    public async IAsyncEnumerable<AudioBuffer> SynthesizeStreamingAsync(string text, SpeechSynthesisOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var result = await SynthesizeAsync(text, options, cancellationToken).ConfigureAwait(false);
        yield return result.Audio;
    }

    /// <inheritdoc/>
    public string VoiceArchitecture => "f5-tts";

    /// <inheritdoc/>
    /// <remarks>F5-TTS conditions on the recording itself: the voice is the recording (mono, at the model's rate) with its words, which are required.</remarks>
    public Task<VoiceProfile> CreateVoiceAsync(string name, AudioBuffer recording, string? referenceText = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(recording);
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (string.IsNullOrWhiteSpace(referenceText))
        {
            throw new ArgumentException("F5-TTS needs the words of the recording to make a voice.", nameof(referenceText));
        }

        return Task.FromResult(new VoiceProfile
        {
            Name = name,
            Architecture = VoiceArchitecture,
            ReferenceText = referenceText,
            SampleRate = SampleRate,
            Tensors = new Dictionary<string, VoiceTensor> { ["reference_audio"] = VoiceTensor.From(recording.ToModelInput(SampleRate).Samples.ToArray()) },
        });
    }

    /// <summary>The prepared request: text ids (one per frame, −1 after the text), the reference mel [reference frames, mels], the duration in frames, the frames before the generated part and the reference's RMS.</summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="options">The request with the reference recording and its text.</param>
    public (int[] Text, float[] Cond, int Duration, int ReferenceFrames, float Rms) Prepare(string text, SpeechSynthesisOptions options)
    {
        if (options.VoiceProfile is { } profile)
        {
            profile.RequireArchitecture(VoiceArchitecture);
            options = options with
            {
                ReferenceAudio = new AudioBuffer(profile.Tensor("reference_audio").ToFloats(), SampleRate),
                ReferenceText = options.ReferenceText ?? profile.ReferenceText,
                VoiceProfile = null,
            };
        }

        var reference = options.ReferenceAudio ?? throw new NotSupportedException(
            "F5-TTS speaks with the voice of a reference recording: set SpeechSynthesisOptions.ReferenceAudio and ReferenceText.");
        string referenceText = options.ReferenceText ?? throw new NotSupportedException("F5-TTS needs the words of the reference recording (SpeechSynthesisOptions.ReferenceText).");
        text = options.NormalizedText(text);
        var samples = reference.ToModelInput(SampleRate).Samples.ToArray();
        double energy = 0;
        foreach (float s in samples)
        {
            energy += (double)s * s;
        }

        float rms = (float)Math.Sqrt(energy / Math.Max(1, samples.Length));
        if (rms < TargetRms && rms > 0)
        {
            for (int i = 0; i < samples.Length; i++)
            {
                samples[i] *= TargetRms / rms;
            }
        }

        var mel = _features.Extract(samples);
        var cond = new float[mel.Frames * _melBins];
        for (int t = 0; t < mel.Frames; t++)
        {
            for (int m = 0; m < _melBins; m++)
            {
                cond[(t * _melBins) + m] = mel.Values[((long)m * mel.Frames) + t];
            }
        }

        if (Encoding.UTF8.GetByteCount(referenceText[^1..]) == 1)
        {
            referenceText += " ";
        }

        int referenceFrames = samples.Length / _hop;
        int referenceBytes = Encoding.UTF8.GetByteCount(referenceText);
        int textBytes = Encoding.UTF8.GetByteCount(text);
        float speed = textBytes < 10 ? 0.3f : options.Speed;
        int duration = referenceFrames + (int)(referenceFrames / (double)referenceBytes * textBytes / speed);
        var ids = Characters(referenceText + text).Select(c => _characters.GetValueOrDefault(c, 0)).ToList();

        duration = Math.Min(Math.Max(Math.Max(ids.Count, mel.Frames) + 1, duration), F5Adapter.MaxFrames);
        return ([.. ids], cond, duration, referenceFrames, rms);
    }

    /// <summary>The sampled mel [duration, mels] with the reference frames restored, from the given starting noise (or the noise source's).</summary>
    /// <param name="ids">The character ids.</param>
    /// <param name="cond">The reference mel.</param>
    /// <param name="duration">The frames.</param>
    /// <param name="noise">The starting noise [duration, mels], or <see langword="null"/>.</param>
    /// <param name="cancellationToken">Stops between steps.</param>
    public float[] Sample(int[] ids, float[] cond, int duration, float[]? noise, CancellationToken cancellationToken)
    {
        var transformer = _transformer ??= _model.CompileComponent(F5Adapter.TransformerRole);
        int width = _melBins;
        int condFrames = cond.Length / width;
        var condition = new float[duration * width];
        cond.AsSpan(0, Math.Min(cond.Length, condition.Length)).CopyTo(condition);
        var text = new int[duration];
        var mask = new float[duration];
        for (int i = 0; i < duration; i++)
        {
            int id = i < ids.Length ? ids[i] : -1;
            text[i] = id + 1;
            mask[i] = id >= 0 ? 1f : 0f;
        }

        var positions = Enumerable.Range(0, duration).ToArray();
        var x = (float[])(noise ?? (NoiseSource ?? (n => SpeechNoise.Gaussian(n, 0)))(duration * width)).Clone();
        var times = new double[_steps + 1];
        for (int i = 0; i <= _steps; i++)
        {
            double t = (double)i / _steps;
            times[i] = t + (_sway * (Math.Cos(Math.PI / 2 * t) - 1 + t));
        }

        var zeros = new float[duration * width];
        var noText = new int[duration];
        var conditioned = new float[duration * width];
        var unconditioned = new float[duration * width];
        for (int step = 0; step < _steps; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            float t = (float)times[step];
            Evaluate(transformer, x, condition, text, mask, positions, t, conditioned);
            Evaluate(transformer, x, zeros, noText, mask, positions, t, unconditioned);
            float dt = (float)(times[step + 1] - times[step]);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] += dt * (conditioned[i] + ((conditioned[i] - unconditioned[i]) * _guidance));
            }
        }

        condition.AsSpan(0, Math.Min(condFrames, duration) * width).CopyTo(x);
        return x;
    }

    /// <summary>One evaluation of the field [duration, mels]: with the reference and the text, or without both (<paramref name="unconditional"/>).</summary>
    /// <param name="x">The current sample [duration, mels].</param>
    /// <param name="ids">The character ids.</param>
    /// <param name="cond">The reference mel.</param>
    /// <param name="t">The time.</param>
    /// <param name="unconditional">Drop the reference audio and the text.</param>
    public float[] Velocity(float[] x, int[] ids, float[] cond, float t, bool unconditional)
    {
        var transformer = _transformer ??= _model.CompileComponent(F5Adapter.TransformerRole);
        int duration = x.Length / _melBins;
        var condition = new float[x.Length];
        if (!unconditional)
        {
            cond.AsSpan(0, Math.Min(cond.Length, condition.Length)).CopyTo(condition);
        }

        var text = new int[duration];
        var mask = new float[duration];
        for (int i = 0; i < duration; i++)
        {
            int id = i < ids.Length ? ids[i] : -1;
            text[i] = unconditional ? 0 : id + 1;
            mask[i] = id >= 0 ? 1f : 0f;
        }

        var output = new float[x.Length];
        Evaluate(transformer, x, condition, text, mask, [.. Enumerable.Range(0, duration)], t, output);
        return output;
    }

    /// <summary>Vocos over mel frames [frames, mels]: the waveform.</summary>
    /// <param name="mel">The mel frames.</param>
    public float[] Vocode(float[] mel)
    {
        int frames = mel.Length / _melBins;
        var vocoder = _vocoder ??= _model.CompileComponent(F5Adapter.VocoderRole);
        vocoder.SetInput<float>(vocoder.Graph.Inputs.First(v => v.Input!.Name == VocosGraph.MelInput), mel);
        var run = new RunParameters();
        run.Symbols[WellKnown.TokenSymbol] = frames;
        vocoder.Run(run);
        var spectrum = new float[frames * _fftSize];
        vocoder.GetOutput<float>(VocosGraph.FramesOutput, spectrum);
        return OverlapAdd.Synthesize(spectrum, frames, _fftSize, _hop, Window.Hann(_fftSize), _fftSize / 2);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _transformer?.Dispose();
        _vocoder?.Dispose();
        _gate.Dispose();
    }

    /// <summary>
    /// The reference's <c>convert_char_to_pinyin</c> for text without Chinese characters: <c>;</c> and curly quotes replaced, one entry per code point,
    /// and a space before a word of several letters or digits that directly follows punctuation other than <c>:</c> and quotes (the word boundary
    /// the reference's segmenter adds).
    /// </summary>
    /// <param name="text">The text.</param>
    public static List<string> Characters(string text)
    {
        text = text.Replace(';', ',').Replace('“', '"').Replace('”', '"').Replace('‘', '\'').Replace('’', '\'');
        var result = new List<string>();
        var runes = text.EnumerateRunes().ToArray();
        for (int i = 0; i < runes.Length; i++)
        {
            var rune = runes[i];
            bool wordStart = Rune.IsLetterOrDigit(rune) && rune.IsAscii && i + 1 < runes.Length && runes[i + 1].IsAscii && Rune.IsLetterOrDigit(runes[i + 1])
                             && (i == 0 || !Rune.IsLetterOrDigit(runes[i - 1]));
            if (wordStart && result.Count > 0 && result[^1] is not (" " or ":" or "'" or "\""))
            {
                result.Add(" ");
            }

            result.Add(rune.ToString());
        }

        return result;
    }

    private AudioBuffer Synthesize(string text, SpeechSynthesisOptions options, CancellationToken cancellationToken)
    {
        var (ids, cond, duration, referenceFrames, rms) = Prepare(text, options);
        var mel = Sample(ids, cond, duration, null, cancellationToken);
        var samples = Vocode(mel[(referenceFrames * _melBins)..]);
        if (rms < TargetRms && rms > 0)
        {
            for (int i = 0; i < samples.Length; i++)
            {
                samples[i] *= rms / TargetRms;
            }
        }

        return new AudioBuffer(samples, SampleRate);
    }

    private void Evaluate(CompiledModel transformer, float[] x, float[] cond, int[] text, float[] mask, int[] positions, float t, float[] output)
    {
        Value Input(string name) => transformer.Graph.Inputs.First(v => v.Input!.Name == name);
        transformer.SetInput<float>(Input(F5TransformerGraph.SampleInput), x);
        transformer.SetInput<float>(Input(F5TransformerGraph.CondInput), cond);
        transformer.SetInput<int>(Input(F5TransformerGraph.TextInput), text);
        transformer.SetInput<float>(Input(F5TransformerGraph.TextMaskInput), mask);
        transformer.SetInput<int>(Input(F5TransformerGraph.PositionsInput), positions);
        transformer.SetInput<float>(Input(F5TransformerGraph.TimeInput), [t]);
        var run = new RunParameters();
        run.Symbols[WellKnown.TokenSymbol] = x.Length / _melBins;
        transformer.Run(run);
        transformer.GetOutput<float>(F5TransformerGraph.VelocityOutput, output);
    }
}
