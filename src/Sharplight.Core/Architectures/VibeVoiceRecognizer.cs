using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using Sharplight.Audio;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Speech;
using Sharplight.Tasks;
using Sharplight.Tokenizers;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// The <see cref="ISpeechRecognizer"/> of a VibeVoice-ASR model (<see cref="VibeVoiceAsrAdapter"/>), as transformers' <c>VibeVoiceAsrForConditionalGeneration.generate</c>: the audio is normalized
/// to -25 dBFS, encoded by the two speech tokenizers (64 frames per run, the convolutions' contexts carried from run to run; the acoustic latents are the encoder's mean, without the
/// training-time sampling noise) and the connectors into language model rows, which replace the audio placeholders of the chat prompt; the language model generates (greedy unless the
/// request samples) a JSON array of segments with start and end times, speaker numbers and content, which becomes the <see cref="TranscriptionSegment"/>s (with their <see cref="TranscriptionSegment.Speaker"/>),
/// yielded as soon as each object is complete. One pass covers as much audio as the context holds (about 40 % of it for the audio rows); a longer recording runs in windows whose speaker numbers are
/// independent.
/// </summary>
internal sealed class VibeVoiceRecognizer : ISpeechRecognizer
{
    private readonly TaskModel _model;
    private readonly LlmModel _llm;
    private readonly ITokenizer _tokenizer;
    private readonly int _audioToken;
    private readonly int _imEnd;
    private readonly int _eos;
    private readonly int _hidden;
    private readonly int _hop;
    private readonly float _normalizeDbfs;
    private readonly string _template;
    private readonly string _contextTemplate;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private CompiledModel? _encoder;
    private VibeVoiceConvCaches? _acousticCaches;
    private VibeVoiceConvCaches? _semanticCaches;
    private InferenceSession? _session;
    private bool _disposed;

    /// <summary>Creates the recognizer; nothing is compiled until the first request.</summary>
    /// <param name="model">A loaded VibeVoice-ASR model.</param>
    public VibeVoiceRecognizer(TaskModel model)
    {
        _model = model;
        _llm = LlmModel.FromTaskModel(model);
        _tokenizer = _llm.Tokenizer;
        var text = model.Description.Component(VibeVoiceAsrAdapter.TextModelRole);
        _audioToken = text.ConfigValue("audioToken", 151648);
        _imEnd = text.ConfigValue("imEnd", 151645);
        _eos = text.ConfigValue("eos", 151643);
        _hidden = text.ConfigValue("hidden", 3584);
        _hop = text.ConfigValue("hop", VibeVoiceAdapter.Hop);
        _normalizeDbfs = (float)text.ConfigValue("normalizeDbfs", -25.0);
        _template = text.ConfigValue("prompt.template", "");
        _contextTemplate = text.ConfigValue("prompt.contextTemplate", "");
        SampleRate = text.ConfigValue("sampleRate", 24000);
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    /// <inheritdoc/>
    public TaskClientMetadata Metadata { get; }

    /// <inheritdoc/>
    public int SampleRate { get; }

    /// <summary>Whether <paramref name="model"/> is a VibeVoice-ASR model (<see cref="VibeVoiceAsrAdapter"/>).</summary>
    /// <param name="model">A loaded speech-recognition model.</param>
    public static bool Handles(TaskModel model)
        => model.Description.TryGetComponent(VibeVoiceAsrAdapter.TextModelRole, out var text) && text.ConfigValue("family", "") == VibeVoiceAsrAdapter.Family
           && model.Description.TryGetComponent(VibeVoiceAsrAdapter.AudioEncoderRole, out _);

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
    public async Task<Transcription> TranscribeAsync(AudioBuffer audio, SpeechRecognitionOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        var segments = new List<TranscriptionSegment>();
        await foreach (var segment in TranscribeStreamingAsync(audio, options, cancellationToken).ConfigureAwait(false))
        {
            segments.Add(segment);
        }

        string text = string.Join(" ", segments.Select(s => s.Text.Trim()).Where(t => t.Length > 0));
        return new Transcription(text, options?.Language, segments, audio.Duration);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<TranscriptionSegment> TranscribeStreamingAsync(AudioBuffer audio, SpeechRecognitionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ObjectDisposedException.ThrowIf(_disposed, this);
        options ??= SpeechRecognitionOptions.Default;
        if (options.Translate)
        {
            throw new NotSupportedException("This model transcribes only; it does not translate.");
        }

        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var samples = audio.ToModelInput(SampleRate).Samples.ToArray();
            _encoder ??= await Task.Run(() => _model.CompileComponent(VibeVoiceAsrAdapter.AudioEncoderRole, VibeVoiceAsrAdapter.EncodeGraph), cancellationToken).ConfigureAwait(false);
            _session ??= await Task.Run(() => _llm.CreateSession(WellKnown.DecodeMultimodalGraph), cancellationToken).ConfigureAwait(false);
            int windowRows = Math.Max(VibeVoiceAsrAdapter.EncodeFrames, (int)(_session.ContextLength * 0.4));
            long windowSamples = (long)windowRows * _hop;
            for (long first = 0; first < samples.Length; first += windowSamples)
            {
                int length = (int)Math.Min(windowSamples, samples.Length - first);
                var window = samples.AsSpan((int)first, length).ToArray();
                double offset = (double)first / SampleRate;
                using var enumerator = Recognize(window, offset, options, cancellationToken).GetEnumerator();
                while (await Task.Run(enumerator.MoveNext, cancellationToken).ConfigureAwait(false))
                {
                    yield return enumerator.Current;
                }
            }
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>The audio embedding rows [rows, hidden] of a recording: normalized, encoded in runs of <see cref="VibeVoiceAsrAdapter.EncodeFrames"/> frames (a shorter tail is padded with silence).</summary>
    /// <param name="samples">24 kHz mono samples.</param>
    public float[] Encode(float[] samples)
    {
        ArgumentNullException.ThrowIfNull(samples);
        var encoder = _encoder ??= _model.CompileComponent(VibeVoiceAsrAdapter.AudioEncoderRole, VibeVoiceAsrAdapter.EncodeGraph);
        var caches = _acousticCaches ??= new VibeVoiceConvCaches(encoder, VibeVoiceAdapter.AcousticPrefix);
        var semanticCaches = _semanticCaches ??= new VibeVoiceConvCaches(encoder, VibeVoiceAdapter.SemanticPrefix);
        var graph = encoder.Graph;
        var input = graph.Inputs.First(i => i.Input!.Name == AudioCodecGraphs.AudioInput);
        var normalized = VibeVoiceLoudness.Normalize((float[])samples.Clone(), _normalizeDbfs);
        int rows = (normalized.Length + _hop - 1) / _hop;
        int block = VibeVoiceAsrAdapter.EncodeFrames;
        caches.Reset();
        semanticCaches.Reset();
        var result = new float[(long)rows * _hidden];
        var output = new float[block * _hidden];
        for (int first = 0; first < rows; first += block)
        {
            var run = new float[block * _hop];
            normalized.AsSpan(first * _hop, Math.Min(run.Length, Math.Max(0, normalized.Length - (first * _hop)))).CopyTo(run);
            encoder.SetInput<float>(input, run);
            caches.Bind();
            semanticCaches.Bind();
            encoder.Run(new RunParameters());
            caches.Collect();
            semanticCaches.Collect();
            int count = Math.Min(block, rows - first);
            encoder.GetOutput<float>("rows", output);
            output.AsSpan(0, count * _hidden).CopyTo(result.AsSpan(first * _hidden));
        }

        return result;
    }

    /// <summary>The prompt of a window: the chat template with <paramref name="rows"/> audio placeholders and the window's duration.</summary>
    /// <param name="rows">The audio rows.</param>
    /// <param name="seconds">The duration of the window's audio.</param>
    /// <param name="context">Extra context for the transcription, or <see langword="null"/>.</param>
    public int[] PromptTokens(int rows, double seconds, string? context)
    {
        string template = string.IsNullOrEmpty(context) ? _template : _contextTemplate.Replace("{context}", context, StringComparison.Ordinal);
        var audio = new StringBuilder();
        string token = _tokenizer.TokenString(_audioToken);
        for (int i = 0; i < rows; i++)
        {
            audio.Append(token);
        }

        string prompt = template.Replace("{audio}", audio.ToString(), StringComparison.Ordinal).Replace("{duration}", seconds.ToString("0.00", CultureInfo.InvariantCulture), StringComparison.Ordinal);
        return _tokenizer.Encode(prompt, addSpecial: false, parseSpecial: true);
    }

    /// <summary>Recognizes one window: encodes it, prompts the language model, generates and yields the segments as they complete (times shifted by <paramref name="offset"/>).</summary>
    /// <param name="window">24 kHz mono samples.</param>
    /// <param name="offset">The window's start in the recording, in seconds.</param>
    /// <param name="options">The request.</param>
    /// <param name="cancellationToken">Stops between tokens.</param>
    public IEnumerable<TranscriptionSegment> Recognize(float[] window, double offset, SpeechRecognitionOptions options, CancellationToken cancellationToken)
    {
        var embeddings = Encode(window);
        var tokens = PromptTokens(embeddings.Length / _hidden, (double)window.Length / SampleRate, options.Prompt);
        var generated = new List<int>();
        var parser = new VibeVoiceSegmentParser();
        foreach (int id in GenerateTokens(embeddings, tokens, options, cancellationToken))
        {
            generated.Add(id);
            if (_tokenizer.TokenString(id).Contains('}', StringComparison.Ordinal))
            {
                foreach (var segment in parser.Feed(_tokenizer.Decode([.. generated], skipSpecial: true)))
                {
                    yield return ToSegment(segment, offset, options);
                }
            }
        }

        foreach (var segment in parser.Feed(_tokenizer.Decode([.. generated], skipSpecial: true)))
        {
            yield return ToSegment(segment, offset, options);
        }
    }

    /// <summary>
    /// The tokens the language model generates for a prompt whose audio placeholders take <paramref name="embeddings"/> (greedy unless the request samples), up to the first end-of-turn token
    /// (<c>&lt;|im_end|&gt;</c> or end of text; not included).
    /// </summary>
    /// <param name="embeddings">The audio rows [rows, hidden].</param>
    /// <param name="tokens">The prompt (<see cref="PromptTokens"/>).</param>
    /// <param name="options">The request (sampling).</param>
    /// <param name="cancellationToken">Stops between tokens.</param>
    /// <param name="forced">Feeds these tokens instead of the generated ones (teacher forcing), or <see langword="null"/>.</param>
    /// <param name="observe">Receives each step's logits, or <see langword="null"/>.</param>
    public IEnumerable<int> GenerateTokens(float[] embeddings, int[] tokens, SpeechRecognitionOptions options, CancellationToken cancellationToken, IReadOnlyList<int>? forced = null,
        Action<int, float[]>? observe = null)
    {
        var session = _session ??= _llm.CreateSession(WellKnown.DecodeMultimodalGraph);
        session.Reset();
        var sampler = new Sampler(options.Sampling);
        foreach (int t in tokens)
        {
            sampler.Accept(t);
        }

        var logits = session.Process(Assemble(tokens, session.RopePosition), [embeddings]);
        int budget = Math.Max(1, session.ContextLength - tokens.Length - 1);
        for (int i = 0; i < budget; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            observe?.Invoke(i, logits);
            int id = sampler.Sample(logits);
            if (forced is not null)
            {
                if (i >= forced.Count)
                {
                    yield break;
                }

                id = forced[i];
            }

            if (id == _eos || id == _imEnd || _tokenizer.IsEndOfGeneration(id))
            {
                yield break;
            }

            sampler.Accept(id);
            yield return id;
            logits = session.Process([id]);
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
        _encoder?.Dispose();
        _gate.Dispose();
    }

    private static TranscriptionSegment ToSegment(VibeVoiceSegment segment, double offset, SpeechRecognitionOptions options) => new()
    {
        Start = TimeSpan.FromSeconds(offset + segment.Start),
        End = TimeSpan.FromSeconds(offset + segment.End),
        Text = segment.Content.Trim(),
        Speaker = segment.Speaker.ToString(CultureInfo.InvariantCulture),
        Language = options.Language,
    };

    // The prompt as the multimodal graph's rows: text tokens, and the audio rows (one per embedding row) at the placeholders.
    private MultimodalPrompt Assemble(int[] tokens, int startPosition)
    {
        var imageIndex = new int[tokens.Length];
        var imageRow = new int[tokens.Length];
        var positions = new int[tokens.Length];
        int row = 0;
        for (int i = 0; i < tokens.Length; i++)
        {
            bool audio = tokens[i] == _audioToken;
            imageIndex[i] = audio ? 0 : -1;
            imageRow[i] = audio ? row++ : 0;
            positions[i] = startPosition + i;
        }

        return new MultimodalPrompt
        {
            Tokens = tokens,
            ImageIndex = imageIndex,
            ImageRow = imageRow,
            Positions = positions,
            PositionAxes = 1,
            NextPosition = startPosition + tokens.Length,
        };
    }
}
