using System.Runtime.CompilerServices;
using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Architectures;

/// <summary>
/// The <see cref="ISpeechSynthesizer"/> of an Orpheus model (<see cref="OrpheusAdapter"/>): the language model generates SNAC codes after the prompt
/// <c>[start of human] "{voice}: {text}" [end of text, end of human]</c> through the text-generation machinery (session, sampler), seven tokens per codec frame
/// (one coarse code, two medium, four fine, each offset by its position in the frame), and the SNAC decoder graph turns the codes into the waveform on the same devices.
/// The streaming form decodes every <see cref="StreamFrames"/> frames with <see cref="StreamContext"/> frames of context before them.
/// </summary>
internal sealed class OrpheusSynthesizer : ISpeechSynthesizer
{
    /// <summary>The token that ends the speech the model generated.</summary>
    public const int EndOfSpeech = 128258;

    /// <summary>The token that opens the human turn of the prompt.</summary>
    public const int StartOfHuman = 128259;

    /// <summary>The token that closes the human turn of the prompt.</summary>
    public const int EndOfHuman = 128260;

    /// <summary>The end-of-text token before the end of the human turn.</summary>
    public const int EndOfText = 128009;

    /// <summary>The token id of the first codec code (code 0 of the coarse codebook).</summary>
    public const int AudioOffset = 128266;

    /// <summary>Tokens per codec frame.</summary>
    public const int TokensPerFrame = 7;

    /// <summary>Frames decoded per streamed buffer.</summary>
    public const int StreamFrames = 8;

    /// <summary>Frames of already generated speech the decoder sees before each streamed buffer.</summary>
    public const int StreamContext = 2;

    private readonly TaskModel _model;
    private readonly LlmModel _llm;
    private readonly int _codebookSize;
    private readonly int _frameSamples;
    private readonly bool _noise;
    private readonly long[] _strides;
    private readonly long[] _rates;
    private InferenceSession? _session;
    private CompiledModel? _codec;
    private bool _disposed;

    /// <summary>Creates the synthesizer; nothing is compiled until the first request.</summary>
    /// <param name="model">A loaded Orpheus model.</param>
    public OrpheusSynthesizer(TaskModel model)
    {
        _model = model;
        _llm = LlmModel.FromTaskModel(model);
        var codec = model.Description.Component(OrpheusAdapter.CodecRole);
        SampleRate = codec.ConfigValue("sampleRate", 24000);
        _codebookSize = codec.ConfigValue("codebookSize", 4096);
        _frameSamples = codec.ConfigValue("samplesPerFrame", 2048);
        _noise = codec.ConfigValue("noise", true);
        _strides = [.. codec.ConfigValue("strides", "4,2,1").Split(',').Select(long.Parse)];
        _rates = [.. codec.ConfigValue("rates", "8,8,4,2").Split(',').Select(long.Parse)];
        Voices = OrpheusAdapter.DefaultVoices;
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    /// <inheritdoc/>
    public TaskClientMetadata Metadata { get; }

    /// <inheritdoc/>
    public int SampleRate { get; }

    /// <inheritdoc/>
    public IReadOnlyList<string> Voices { get; }

    /// <summary>Replaces the source of the decoder noise (standard-normal values, <c>count</c> of them per call). Defaults to a seeded generator.</summary>
    public Func<int, float[]>? NoiseSource { get; set; }

    /// <summary>Whether <paramref name="model"/> is an Orpheus model (<see cref="OrpheusAdapter"/>).</summary>
    /// <param name="model">A loaded speech-synthesis model.</param>
    public static bool Handles(TaskModel model)
        => model.Description.TryGetComponent(OrpheusAdapter.TextModelRole, out var text) && text.ConfigValue("family", "") == OrpheusAdapter.Family
           && model.Description.TryGetComponent(OrpheusAdapter.CodecRole, out _);

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
        var codes = await GenerateCodesAsync(text, options, cancellationToken).ConfigureAwait(false);
        var audio = await Task.Run(() => Decode(codes), cancellationToken).ConfigureAwait(false);
        return new SpeechSynthesisResult(audio, options?.Voice);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<AudioBuffer> SynthesizeStreamingAsync(string text, SpeechSynthesisOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var codes = new List<int>();
        int decoded = 0;
        await foreach (int code in GenerateCodeStreamAsync(text, options, cancellationToken).ConfigureAwait(false))
        {
            codes.Add(code);
            int frames = codes.Count / TokensPerFrame;
            if (frames - decoded >= StreamFrames)
            {
                yield return await Task.Run(() => DecodeSlice(codes, decoded, frames), cancellationToken).ConfigureAwait(false);
                decoded = frames;
            }
        }

        int total = codes.Count / TokensPerFrame;
        if (total > decoded)
        {
            yield return await Task.Run(() => DecodeSlice(codes, decoded, total), cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>The token ids of the prompt of a request.</summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="voice">The voice name (one of <see cref="Voices"/>), or <see langword="null"/> for the first.</param>
    public int[] PromptFor(string text, string? voice = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        string name = voice ?? Voices[0];
        var ids = _llm.Tokenizer.Encode($"{name}: {text}", addSpecial: true, parseSpecial: false);
        return [StartOfHuman, .. ids, EndOfText, EndOfHuman];
    }

    /// <summary>Generates the codec codes of the utterance (seven per frame).</summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Stops generation between tokens.</param>
    public async Task<int[]> GenerateCodesAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
    {
        var codes = new List<int>();
        await foreach (int code in GenerateCodeStreamAsync(text, options, cancellationToken).ConfigureAwait(false))
        {
            codes.Add(code);
        }

        return [.. codes];
    }

    /// <summary>Runs the codec decoder over <paramref name="codes"/> (seven per frame; a trailing partial frame and codes outside the codebook are dropped).</summary>
    /// <param name="codes">The codes, each already reduced by its position's offset (<c>token - 128266 - (i mod 7) · 4096</c>).</param>
    public AudioBuffer Decode(IReadOnlyList<int> codes) => DecodeSlice(codes, 0, codes.Count / TokensPerFrame, 0);

    private AudioBuffer DecodeSlice(IReadOnlyList<int> codes, int first, int end) => DecodeSlice(codes, first, end, Math.Min(StreamContext, first));

    /// <summary>Decodes frames <c>[first, end)</c> with <paramref name="context"/> frames before them whose audio is dropped.</summary>
    private AudioBuffer DecodeSlice(IReadOnlyList<int> codes, int first, int end, int context)
    {
        int start = first - context;
        int frames = end - start;
        if (frames <= 0)
        {
            return new AudioBuffer([], SampleRate);
        }

        if (frames > OrpheusAdapter.MaxCodecFrames)
        {
            throw new ArgumentException($"A decode covers at most {OrpheusAdapter.MaxCodecFrames} frames, got {frames}.", nameof(codes));
        }

        var layers = _strides.Select(s => new int[frames * (_strides[0] / s)]).ToArray();
        for (int f = 0; f < frames; f++)
        {
            int[] frame = [.. Enumerable.Range(0, TokensPerFrame).Select(i => Math.Clamp(codes[((start + f) * TokensPerFrame) + i], 0, _codebookSize - 1))];
            layers[0][f] = frame[0];
            layers[1][2 * f] = frame[1];
            layers[1][(2 * f) + 1] = frame[4];
            layers[2][4 * f] = frame[2];
            layers[2][(4 * f) + 1] = frame[3];
            layers[2][(4 * f) + 2] = frame[5];
            layers[2][(4 * f) + 3] = frame[6];
        }

        _codec ??= _model.Compile(_model.ComponentGraph(OrpheusAdapter.CodecRole), null, OrpheusAdapter.MaxCodecFrames);
        for (int i = 0; i < layers.Length; i++)
        {
            _codec.SetInput<int>(Input(SnacGraph.CodesInput(i)), layers[i]);
        }

        if (_noise)
        {
            long scale = _strides[0];
            for (int i = 0; i < _rates.Length; i++)
            {
                scale *= _rates[i];
                int count = (int)(scale * frames);
                _codec.SetInput<float>(Input(SnacGraph.NoiseInput(i)), (NoiseSource ?? (n => KokoroSynthesizer.Gaussian(n, i)))(count));
            }
        }

        var parameters = new RunParameters();
        parameters.Symbols[WellKnown.TokenSymbol] = frames;
        _codec.Run(parameters);
        var samples = new float[frames * _frameSamples];
        _codec.GetOutput<float>(SnacGraph.AudioOutput, samples);
        return new AudioBuffer(samples[(context * _frameSamples)..], SampleRate);
    }

    private Value Input(string name) => _codec!.Graph.Inputs.First(i => i.Name == name);

    /// <summary>Generates the prompt's continuation and yields the position-reduced codec codes one by one.</summary>
    private async IAsyncEnumerable<int> GenerateCodeStreamAsync(string text, SpeechSynthesisOptions? options, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var prompt = PromptFor(text, options?.Voice);
        var sampling = options?.Sampling ?? new SamplingOptions { Temperature = 0.6f, TopK = 0, TopP = 0.8f, RepetitionPenalty = 1.3f, RepetitionWindow = 4096 };
        _session ??= await Task.Run(() => _llm.CreateSession(), cancellationToken).ConfigureAwait(false);
        _session.Reset();
        var generator = new Generator(_session, _llm.Tokenizer);
        int budget = Math.Max(1, Math.Min(_session.ContextLength - prompt.Length, (OrpheusAdapter.MaxCodecFrames * TokensPerFrame) + 16));
        var generation = new GenerationOptions { MaxNewTokens = budget, Sampling = sampling, StopAtEog = false };
        using var tokens = generator.Generate(prompt, generation).GetEnumerator();
        int index = 0;
        while (await Task.Run(tokens.MoveNext, cancellationToken).ConfigureAwait(false))
        {
            int id = tokens.Current.Id;
            if (id == EndOfSpeech)
            {
                yield break;
            }

            if (id >= AudioOffset)
            {
                yield return id - AudioOffset - ((index % TokensPerFrame) * _codebookSize);
                index++;
            }
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
    }
}
