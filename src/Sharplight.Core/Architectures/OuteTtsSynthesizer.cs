using System.Runtime.CompilerServices;
using Sharplight.Audio;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Architectures;

/// <summary>
/// The <see cref="ISpeechSynthesizer"/> of an OuteTTS model (<see cref="OuteTtsAdapter"/>): the language model generates codec tokens after
/// the <see cref="OuteTtsPrompt"/> through the text-generation machinery (session, sampler), the WavTokenizer decoder graph turns the tokens of the
/// whole utterance into inverse-DFT frames on the same devices, and the host overlap-adds them into the waveform. The streaming form decodes
/// every spoken word (its <c>&lt;|code_start|&gt;</c> … <c>&lt;|code_end|&gt;</c> group) on its own as soon as the language model has produced it.
/// </summary>
internal sealed class OuteTtsSynthesizer : ISpeechSynthesizer
{
    private const string CodeEnd = "<|code_end|>";
    private const string CodeStart = "<|code_start|>";

    private readonly TaskModel _model;
    private readonly LlmModel _llm;
    private readonly Dictionary<string, string> _voiceAttachments;
    private readonly int _vocabSize;
    private readonly int _nFft;
    private readonly int _hop;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private InferenceSession? _session;
    private CompiledModel? _codec;
    private bool _disposed;

    /// <summary>Creates the synthesizer; nothing is compiled until the first request.</summary>
    /// <param name="model">A loaded OuteTTS model.</param>
    public OuteTtsSynthesizer(TaskModel model)
    {
        _model = model;
        _llm = LlmModel.FromTaskModel(model);
        var codec = model.Description.Component(OuteTtsAdapter.CodecRole);
        SampleRate = codec.ConfigValue("sampleRate", 24000);
        _nFft = codec.ConfigValue("nFft", 1280);
        _hop = codec.ConfigValue("hopLength", 320);
        _vocabSize = codec.ConfigValue("vocabSize", 4096);
        const string prefix = "voices/";
        _voiceAttachments = model.Description.Component(OuteTtsAdapter.TextModelRole).Attachments
            .Where(a => a.Key.StartsWith(prefix, StringComparison.Ordinal) && a.Key.EndsWith(".json", StringComparison.Ordinal))
            .ToDictionary(a => a.Key[prefix.Length..^5], a => a.Key, StringComparer.Ordinal);
        Voices = [.. _voiceAttachments.Keys.Order(StringComparer.Ordinal)];
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    /// <inheritdoc/>
    public TaskClientMetadata Metadata { get; }

    /// <inheritdoc/>
    public int SampleRate { get; }

    /// <inheritdoc/>
    public IReadOnlyList<string> Voices { get; }

    /// <summary>Whether <paramref name="model"/> is an OuteTTS model (<see cref="OuteTtsAdapter"/>).</summary>
    /// <param name="model">A loaded speech-synthesis model.</param>
    public static bool Handles(TaskModel model)
        => model.Description.TryGetComponent(OuteTtsAdapter.TextModelRole, out var text) && text.ConfigValue("family", "") == OuteTtsAdapter.Family
           && model.Description.TryGetComponent(OuteTtsAdapter.CodecRole, out _);

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
    /// <remarks>Returns at once: synthesis runs on a background task (<see cref="BackgroundCompute"/>), one request at a time per client.</remarks>
    public async Task<SpeechSynthesisResult> SynthesizeAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
    {
        var codes = await GenerateCodesAsync(text, options, cancellationToken).ConfigureAwait(false);
        var audio = await BackgroundCompute.RunAsync(_gate, () => Decode(codes), cancellationToken).ConfigureAwait(false);
        return new SpeechSynthesisResult(audio, options?.Voice);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Returns at once: synthesis runs on a background task (<see cref="BackgroundCompute"/>), one request at a time per client, and queues
    /// each word's audio for the caller without waiting for it to be consumed.
    /// </remarks>
    public IAsyncEnumerable<AudioBuffer> SynthesizeStreamingAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
        => BackgroundCompute.StreamAsync(token => SynthesizeCoreAsync(text, options, token), cancellationToken);

    /// <summary>Generates the codec tokens of the utterance and returns them in one list (the language-model half of <see cref="SynthesizeAsync"/>).</summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Stops generation between tokens.</param>
    public async Task<int[]> GenerateCodesAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
    {
        var codes = new List<int>();
        await foreach (var word in BackgroundCompute.StreamAsync(token => GenerateWordsGatedAsync(text, options, token), cancellationToken).ConfigureAwait(false))
        {
            codes.AddRange(word);
        }

        return [.. codes];
    }

    private async IAsyncEnumerable<AudioBuffer> SynthesizeCoreAsync(string text, SpeechSynthesisOptions? options, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await foreach (var word in GenerateWordsGatedAsync(text, options, cancellationToken).ConfigureAwait(false))
        {
            yield return await Task.Run(() => Decode(word), cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary><see cref="GenerateWordsAsync"/> holding the client's gate (one request at a time: the session is shared).</summary>
    private async IAsyncEnumerable<int[]> GenerateWordsGatedAsync(string text, SpeechSynthesisOptions? options, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await foreach (var word in GenerateWordsAsync(text, options, cancellationToken).ConfigureAwait(false))
            {
                yield return word;
            }
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>The prompt a request runs on (see <see cref="OuteTtsPrompt"/>).</summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="voice">The voice name, or <see langword="null"/>.</param>
    public string PromptFor(string text, string? voice = null)
        => OuteTtsPrompt.Build(OuteTtsPrompt.Words(text), voice is null ? null : LoadVoice(voice));

    /// <summary>Runs the codec decoder over <paramref name="codes"/> and returns the waveform.</summary>
    /// <param name="codes">The codec tokens, one per frame; tokens outside the codebook are dropped.</param>
    public AudioBuffer Decode(IReadOnlyList<int> codes)
    {
        var valid = codes.Where(c => c >= 0 && c < _vocabSize).ToArray();
        if (valid.Length == 0)
        {
            return new AudioBuffer([], SampleRate);
        }

        _codec ??= _model.Compile(_model.ComponentGraph(OuteTtsAdapter.CodecRole), null, OuteTtsAdapter.MaxCodecFrames);
        var input = _codec.Graph.TryInput(InputRole.TokenIds) ?? throw new InvalidOperationException("The codec graph has no token input.");
        var parameters = new RunParameters();
        parameters.Symbols[WavTokenizerGraph.FrameSymbol] = valid.Length;
        _codec.SetInput<int>(input, valid);
        _codec.Run(parameters);
        var frames = new float[valid.Length * _nFft];
        _codec.GetOutput<float>(WavTokenizerGraph.FramesOutput, frames);
        var samples = OverlapAdd.Synthesize(frames, valid.Length, _nFft, _hop, Window.Hann(_nFft), (_nFft - _hop) / 2);
        return new AudioBuffer(samples, SampleRate);
    }

    private OuteTtsVoice LoadVoice(string name)
        => _voiceAttachments.TryGetValue(name, out var attachment)
            ? OuteTtsVoice.Parse(_model.ReadAttachment(OuteTtsAdapter.TextModelRole, attachment))
            : throw new ArgumentException($"Unknown voice '{name}'. The model has: {(Voices.Count == 0 ? "no voices" : string.Join(", ", Voices))}.", nameof(name));

    /// <summary>Generates the codec tokens of one spoken word at a time (the groups between <c>&lt;|code_start|&gt;</c> and <c>&lt;|code_end|&gt;</c>).</summary>
    private async IAsyncEnumerable<int[]> GenerateWordsAsync(string text, SpeechSynthesisOptions? options, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(text);
        ObjectDisposedException.ThrowIf(_disposed, this);
        var words = OuteTtsPrompt.Words(text);
        if (words.Count == 0)
        {
            yield break;
        }

        string prompt = OuteTtsPrompt.Build(words, options?.Voice is { } voice ? LoadVoice(voice) : null);
        var sampling = options.SamplingOr(new SamplingOptions { Temperature = 0.4f, TopK = 40, TopP = 0.9f, RepetitionPenalty = 1.1f });
        _session ??= await Task.Run(() => _llm.CreateSession(), cancellationToken).ConfigureAwait(false);
        _session.Reset();
        var generator = new Generator(_session, _llm.Tokenizer);
        int promptTokens = _llm.Tokenizer.Encode(prompt, addSpecial: true, parseSpecial: true).Length;
        int budget = Math.Max(1, Math.Min(_session.ContextLength - promptTokens, OuteTtsAdapter.MaxCodecFrames + (words.Count * 8)));
        var generation = new GenerationOptions { MaxNewTokens = budget, Sampling = sampling, StopAtEog = true };
        using var tokens = generator.Generate(prompt, generation).GetEnumerator();
        var word = new List<int>();
        while (await Task.Run(tokens.MoveNext, cancellationToken).ConfigureAwait(false))
        {
            string piece = tokens.Current.Text;
            if (piece == OuteTtsPrompt.AudioEnd)
            {
                break;
            }

            if (piece == CodeStart)
            {
                word.Clear();
            }
            else if (piece == CodeEnd)
            {
                yield return [.. word];
                word.Clear();
            }
            else if (OuteTtsPrompt.CodeOf(piece) is >= 0 and var code)
            {
                word.Add(code);
            }
        }

        if (word.Count > 0)
        {
            yield return [.. word];
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
        _session?.Dispose();
        _codec?.Dispose();
        _gate.Dispose();
    }
}
