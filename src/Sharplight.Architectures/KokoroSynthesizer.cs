using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;
using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Architectures;

/// <summary>
/// The <see cref="ISpeechSynthesizer"/> of a Kokoro model (<see cref="KokoroAdapter"/>): the text graph predicts the duration of every phoneme symbol, the host
/// rounds the durations and expands the prosody and acoustic features to frames (the length regulator), the acoustic graph produces the inverse-DFT frames on the
/// same devices, and the host overlap-adds them. The streaming form yields one buffer per sentence.
/// </summary>
internal sealed class KokoroSynthesizer : ISpeechSynthesizer
{
    private readonly TaskModel _model;
    private readonly KokoroParameters _p;
    private readonly Dictionary<string, int> _vocabulary;
    private readonly Dictionary<string, string> _voices;
    private readonly Dictionary<string, string> _lexicons;
    private readonly int _context;
    private readonly long _maxFrames;
    private readonly Dictionary<string, KokoroPhonemizer> _phonemizers = new(StringComparer.Ordinal);
    private CompiledModel? _text;
    private CompiledModel? _acoustic;
    private bool _disposed;

    /// <summary>Creates the synthesizer; nothing is compiled until the first request.</summary>
    /// <param name="model">A loaded Kokoro model.</param>
    public KokoroSynthesizer(TaskModel model)
    {
        _model = model;
        var text = model.Description.Component(KokoroAdapter.TextRole);
        var acoustic = model.Description.Component(KokoroAdapter.AcousticRole);
        SampleRate = acoustic.ConfigValue("sampleRate", 24000);
        _p = KokoroParameters.From(model.Source);
        _context = text.ConfigValue("contextLength", 512);
        _maxFrames = acoustic.SymbolBounds.GetValueOrDefault(WellKnown.TokenSymbol, KokoroAdapter.MaxFrames);
        _vocabulary = JsonSerializer.Deserialize(model.ReadAttachment(KokoroAdapter.TextRole, KokoroAdapter.VocabularyKey), KokoroJsonContext.Default.DictionaryStringInt32) ?? [];
        _voices = text.Attachments.Where(a => a.Key.StartsWith(KokoroAdapter.VoicePrefix, StringComparison.Ordinal) && a.Key.EndsWith(KokoroAdapter.VoiceSuffix, StringComparison.Ordinal))
            .ToDictionary(a => a.Key[KokoroAdapter.VoicePrefix.Length..^KokoroAdapter.VoiceSuffix.Length], a => a.Key, StringComparer.Ordinal);
        _lexicons = text.Attachments.Where(a => a.Key.StartsWith(KokoroAdapter.LexiconPrefix, StringComparison.Ordinal)).ToDictionary(a => a.Key, a => a.Value, StringComparer.Ordinal);
        Voices = [.. _voices.Keys.Order(StringComparer.Ordinal)];
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    /// <inheritdoc/>
    public TaskClientMetadata Metadata { get; }

    /// <inheritdoc/>
    public int SampleRate { get; }

    /// <inheritdoc/>
    public IReadOnlyList<string> Voices { get; }

    /// <summary>
    /// Replaces the source of the excitation noise (standard-normal values, <c>count × harmonics</c>): a run is reproducible when the source is. Defaults to a
    /// generator seeded per request.
    /// </summary>
    public Func<int, float[]>? NoiseSource { get; set; }

    /// <summary>Whether <paramref name="model"/> is a Kokoro model (<see cref="KokoroAdapter"/>).</summary>
    /// <param name="model">A loaded speech-synthesis model.</param>
    public static bool Handles(TaskModel model)
        => model.Description.TryGetComponent(KokoroAdapter.TextRole, out var text) && text.ConfigValue("family", "") == KokoroAdapter.Family
           && model.Description.TryGetComponent(KokoroAdapter.AcousticRole, out _);

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
        var parts = new List<float>();
        await foreach (var buffer in SynthesizeStreamingAsync(text, options, cancellationToken).ConfigureAwait(false))
        {
            parts.AddRange(buffer.Samples);
        }

        return new SpeechSynthesisResult(new AudioBuffer([.. parts], SampleRate), options?.Voice);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<AudioBuffer> SynthesizeStreamingAsync(string text, SpeechSynthesisOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        ObjectDisposedException.ThrowIf(_disposed, this);
        options ??= SpeechSynthesisOptions.Default;
        foreach (string phonemes in PhonemizerFor(options).Chunks(text, options.Language, _context - 2))
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return await Task.Run(() => SynthesizePhonemes(phonemes, options.Voice, options.Speed), cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>Synthesizes a sequence of phoneme symbols (the symbols of the vocabulary; others are dropped).</summary>
    /// <param name="phonemes">The phonemes of at most <c>context - 2</c> symbols.</param>
    /// <param name="voice">The voice name.</param>
    /// <param name="speed">The speaking rate (1 is the voice's natural rate).</param>
    public AudioBuffer SynthesizePhonemes(string phonemes, string? voice, float speed = 1f)
    {
        var run = Run(phonemes, voice, speed);
        return new AudioBuffer(run.Samples, SampleRate);
    }

    /// <summary>One synthesis with its intermediate results (the symbols, rounded durations, F0 curve and samples).</summary>
    /// <param name="phonemes">The phonemes.</param>
    /// <param name="voice">The voice name.</param>
    /// <param name="speed">The speaking rate.</param>
    public KokoroRun Run(string phonemes, string? voice, float speed = 1f)
    {
        ArgumentNullException.ThrowIfNull(phonemes);
        var symbols = new List<int>();
        foreach (var rune in phonemes.EnumerateRunes())
        {
            if (_vocabulary.TryGetValue(rune.ToString(), out int id))
            {
                symbols.Add(id);
            }
        }

        if (symbols.Count == 0)
        {
            return new KokoroRun([], [], [], [], []);
        }

        if (symbols.Count + 2 > _context)
        {
            throw new ArgumentException($"A Kokoro utterance holds at most {_context - 2} phoneme symbols, got {symbols.Count}.", nameof(phonemes));
        }

        var style = VoiceStyle(voice, symbols.Count);
        int[] ids = [0, .. symbols, 0];
        int t = ids.Length;
        _text ??= _model.Compile(_model.ComponentGraph(KokoroAdapter.TextRole), null, _context);
        _text.SetInput<int>(Input(_text, KokoroTextGraph.IdsInput), ids);
        _text.SetInput<int>(Input(_text, KokoroTextGraph.PositionsInput), [.. Enumerable.Range(0, t)]);
        _text.SetInput<float>(Input(_text, KokoroTextGraph.StyleInput), style.AsSpan(_p.StyleDim, _p.StyleDim));
        var parameters = new RunParameters();
        parameters.Symbols[WellKnown.TokenSymbol] = t;
        _text.Run(parameters);
        var raw = new float[t];
        var prosody = new float[t * (_p.Hidden + _p.StyleDim)];
        var acoustic = new float[t * _p.Hidden];
        _text.GetOutput<float>(KokoroTextGraph.DurationOutput, raw);
        _text.GetOutput<float>(KokoroTextGraph.ProsodyOutput, prosody);
        _text.GetOutput<float>(KokoroTextGraph.AcousticOutput, acoustic);

        var durations = new int[t];
        long frames = 0;
        for (int i = 0; i < t; i++)
        {
            durations[i] = Math.Max(1, (int)MathF.Round(raw[i] / speed));
            frames += durations[i];
        }

        if (frames > _maxFrames)
        {
            throw new ArgumentException($"The utterance needs {frames} decoder frames, the model runs at most {_maxFrames}; split the text.", nameof(phonemes));
        }

        var (samples, f0) = Decode(prosody, acoustic, durations, (int)frames, style);
        return new KokoroRun(ids, durations, raw, f0, samples);
    }

    private (float[] Samples, float[] F0) Decode(float[] prosody, float[] acoustic, int[] durations, int frames, float[] style)
    {
        var en = Expand(prosody, _p.Hidden + _p.StyleDim, durations, frames);
        var asr = Expand(acoustic, _p.Hidden, durations, frames);
        int count = frames * (int)_p.SamplesPerFrame * _p.Harmonics;
        var noise = (NoiseSource ?? (n => Gaussian(n, 0)))(count);
        _acoustic ??= _model.Compile(_model.ComponentGraph(KokoroAdapter.AcousticRole), null, _maxFrames);
        _acoustic.SetInput<float>(Input(_acoustic, KokoroAcousticGraph.ProsodyInput), en);
        _acoustic.SetInput<float>(Input(_acoustic, KokoroAcousticGraph.AcousticInput), asr);
        _acoustic.SetInput<float>(Input(_acoustic, KokoroAcousticGraph.DecoderStyleInput), style.AsSpan(0, _p.StyleDim));
        _acoustic.SetInput<float>(Input(_acoustic, KokoroAcousticGraph.PredictorStyleInput), style.AsSpan(_p.StyleDim, _p.StyleDim));
        _acoustic.SetInput<float>(Input(_acoustic, KokoroAcousticGraph.NoiseInput), noise);
        var parameters = new RunParameters();
        parameters.Symbols[WellKnown.TokenSymbol] = frames;
        _acoustic.Run(parameters);
        int samplesPerFrame = (int)_p.SamplesPerFrame;
        int spectrumFrames = (frames * samplesPerFrame / _p.Hop) + 1;
        var spectrum = new float[spectrumFrames * _p.NFft];
        _acoustic.GetOutput<float>(KokoroAcousticGraph.FramesOutput, spectrum);
        if (Environment.GetEnvironmentVariable("SHARPLIGHT_KOKORO_DEBUG") == "1")
        {
            foreach (string name in new[] { "source", "decoder", "stft", "har", "stage0", "stage1", "post" })
            {
                var shape = _acoustic.OutputShape("debug." + name, parameters);
                var data = new float[shape.Aggregate(1L, (a, c) => a * c)];
                _acoustic.GetOutput<float>("debug." + name, data);
                File.WriteAllBytes(Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_KOKORO_DEBUG_DIR") ?? ".", name + ".bin"), MemoryMarshal.AsBytes(data.AsSpan()).ToArray());
            }
        }

        var f0 = new float[2 * frames];
        _acoustic.GetOutput<float>(KokoroAcousticGraph.F0Output, f0);
        return (OverlapAdd.Synthesize(spectrum, spectrumFrames, _p.NFft, _p.Hop, Window.Hann(_p.NFft), _p.NFft / 2), f0);
    }

    /// <summary>The length regulator: row <c>i</c> of <paramref name="rows"/> repeated <c>durations[i]</c> times.</summary>
    private static float[] Expand(float[] rows, int width, int[] durations, int frames)
    {
        var expanded = new float[frames * width];
        int frame = 0;
        for (int i = 0; i < durations.Length; i++)
        {
            for (int d = 0; d < durations[i]; d++)
            {
                Array.Copy(rows, i * width, expanded, frame * width, width);
                frame++;
            }
        }

        return expanded;
    }

    /// <summary>The front end for a request: American or British English lexicons, chosen by the language option or else by the voice (<c>b…</c> voices are British).</summary>
    private KokoroPhonemizer PhonemizerFor(SpeechSynthesisOptions options)
    {
        string language = options.Language?.ToLowerInvariant() ?? string.Empty;
        string voice = options.Voice ?? (Voices.Count > 0 ? Voices[0] : string.Empty);
        string variant = language is "b" or "en-gb" || (language.Length == 0 && voice.StartsWith('b')) ? "gb" : "us";
        if (_phonemizers.TryGetValue(variant, out var cached))
        {
            return cached;
        }

        KokoroLexicon? Load(string name)
        {
            string key = $"{KokoroAdapter.LexiconPrefix}{variant}_{name}.json";
            if (!_lexicons.TryGetValue(key, out var attachment))
            {
                return null;
            }

            var lexicon = new KokoroLexicon();
            lexicon.Add(_model.ReadAttachment(KokoroAdapter.TextRole, attachment));
            return lexicon;
        }

        var gold = Load("gold") ?? new KokoroLexicon();
        return _phonemizers[variant] = new KokoroPhonemizer(gold, Load("silver"));
    }

    private float[] VoiceStyle(string? voice, int symbols)
    {
        string name = voice ?? (Voices.Count > 0 ? Voices[0] : throw new InvalidOperationException("The model has no voices."));
        if (!_voices.TryGetValue(name, out var attachment))
        {
            throw new ArgumentException($"Unknown voice '{name}'. The model has: {string.Join(", ", Voices)}.", nameof(voice));
        }

        var bytes = _model.ReadAttachment(KokoroAdapter.TextRole, attachment);
        int row = Math.Min(symbols, bytes.Length / (2 * _p.StyleDim * sizeof(float))) - 1;
        return MemoryMarshal.Cast<byte, float>(bytes.AsSpan(row * 2 * _p.StyleDim * sizeof(float), 2 * _p.StyleDim * sizeof(float))).ToArray();
    }

    private static Value Input(CompiledModel model, string name) => model.Graph.Inputs.First(i => i.Name == name);

    /// <summary>Standard-normal values (Box-Muller over a seeded generator).</summary>
    /// <param name="count">The number of values.</param>
    /// <param name="seed">The seed.</param>
    internal static float[] Gaussian(int count, int seed)
    {
        var random = new Random(seed);
        var values = new float[count];
        for (int i = 0; i < count; i += 2)
        {
            double radius = Math.Sqrt(-2 * Math.Log(1 - random.NextDouble()));
            double angle = 2 * Math.PI * random.NextDouble();
            values[i] = (float)(radius * Math.Cos(angle));
            if (i + 1 < count)
            {
                values[i + 1] = (float)(radius * Math.Sin(angle));
            }
        }

        return values;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _text?.Dispose();
        _acoustic?.Dispose();
    }
}
