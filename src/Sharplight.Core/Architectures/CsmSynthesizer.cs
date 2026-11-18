using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Speech;
using Sharplight.Tasks;
using Sharplight.Tensors;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// The <see cref="ISpeechSynthesizer"/> of a CSM model (<see cref="CsmAdapter"/>), as transformers' <c>CsmForConditionalGeneration.generate</c> without audio context: the
/// backbone's prompt is the text turn <c>&lt;|begin_of_text|&gt;[speaker]text&lt;|end_of_text|&gt;</c> (embedding rows of the text tokens); every step the backbone predicts a
/// frame's first codebook, the depth decoder codebooks 1 to 31 (its inputs: the backbone's hidden state, then each code's embedding, all projected to its width), and the sum of the
/// frame's 32 codebook embeddings is the backbone's next input. A frame whose first 31 codes are all zero ends the audio. The Mimi decoder turns the frames into 24 kHz audio
/// in chunks of <see cref="CsmAdapter.CodecChunkFrames"/> frames with <see cref="CsmAdapter.CodecContextFrames"/> frames of context.
/// </summary>
internal sealed class CsmSynthesizer : ISpeechSynthesizer
{
    /// <summary>Frames decoded per streamed buffer.</summary>
    public const int StreamFrames = 25;

    private readonly TaskModel _model;
    private readonly LlmModel _llm;
    private readonly int _codebooks;
    private readonly int _vocabulary;
    private readonly int _frameSamples;
    private readonly int _positionsPerFrame;
    private readonly int _maxFrames;
    private readonly SamplingOptions _defaults;
    private readonly SamplingOptions _depthDefaults;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly Lazy<Tables> _tables;
    private InferenceSession? _backbone;
    private InferenceSession? _depth;
    private CompiledModel? _codec;
    private bool _disposed;

    /// <summary>Creates the synthesizer; nothing is compiled until the first request.</summary>
    /// <param name="model">A loaded CSM model.</param>
    public CsmSynthesizer(TaskModel model)
    {
        _model = model;
        _llm = LlmModel.FromTaskModel(model);
        var text = model.Description.Component(CsmAdapter.TextModelRole);
        _codebooks = text.ConfigValue("codebooks", 32);
        _vocabulary = text.ConfigValue("codebookVocabulary", 2051);
        _maxFrames = (int)text.ConfigValue("sampling.max_frames", 125.0);
        _defaults = new SamplingOptions
        {
            Temperature = (float)text.ConfigValue("sampling.temperature", 0.9), TopK = (int)text.ConfigValue("sampling.top_k", 50.0), TopP = (float)text.ConfigValue("sampling.top_p", 1.0),
            MinP = 0f, RepetitionPenalty = 1f,
        };
        _depthDefaults = new SamplingOptions
        {
            Temperature = (float)text.ConfigValue("sampling.depth.temperature", 0.9), TopK = (int)text.ConfigValue("sampling.depth.top_k", 50.0),
            TopP = (float)text.ConfigValue("sampling.depth.top_p", 1.0), MinP = 0f,
        };
        var codec = model.Description.Component(CsmAdapter.CodecRole);
        SampleRate = codec.ConfigValue("sampleRate", 24000);
        _frameSamples = codec.ConfigValue("samplesPerFrame", 1920);
        _positionsPerFrame = codec.ConfigValue(AudioCodecGraphs.PositionsNumeratorKey, 2);
        Voices = ["0", "1"];
        _tables = new Lazy<Tables>(() => new Tables(model.Definition, _codebooks, _vocabulary));
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    /// <inheritdoc/>
    public TaskClientMetadata Metadata { get; }

    /// <inheritdoc/>
    public int SampleRate { get; }

    /// <summary>The speaker numbers the model was trained with (any other non-negative integer is accepted).</summary>
    public IReadOnlyList<string> Voices { get; }

    /// <summary>Whether <paramref name="model"/> is a CSM model (<see cref="CsmAdapter"/>).</summary>
    /// <param name="model">A loaded speech-synthesis model.</param>
    public static bool Handles(TaskModel model)
        => model.Description.TryGetComponent(CsmAdapter.TextModelRole, out var text) && text.ConfigValue("family", "") == CsmAdapter.Family
           && model.Description.TryGetComponent(CsmAdapter.CodecRole, out _);

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
        string normalized = options.NormalizedText(text);
        var audio = await BackgroundCompute.RunAsync(_gate, () =>
        {
            var frames = new List<int[]>();
            foreach (var frame in GenerateFrames(normalized, options, cancellationToken))
            {
                frames.Add(frame);
            }

            return Decode(frames, cancellationToken);
        }, cancellationToken).ConfigureAwait(false);
        return new SpeechSynthesisResult(audio, options?.Voice ?? Voices[0]);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Returns at once: generation and decoding run on a background task (<see cref="BackgroundCompute"/>), one request at a time per client; every
    /// <see cref="StreamFrames"/> frames are decoded with up to <see cref="CsmAdapter.CodecContextFrames"/> frames of context before them.
    /// </remarks>
    public IAsyncEnumerable<AudioBuffer> SynthesizeStreamingAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
        => BackgroundCompute.StreamAsync(token => SynthesizeCoreAsync(text, options, token), cancellationToken);

    private async IAsyncEnumerable<AudioBuffer> SynthesizeCoreAsync(string text, SpeechSynthesisOptions? options, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            string normalized = options.NormalizedText(text);
            var frames = new List<int[]>();
            int decoded = 0;
            using var generation = GenerateFrames(normalized, options, cancellationToken).GetEnumerator();
            while (await Task.Run(generation.MoveNext, cancellationToken).ConfigureAwait(false))
            {
                frames.Add(generation.Current);
                if (frames.Count - decoded >= StreamFrames)
                {
                    int end = frames.Count;
                    yield return await Task.Run(() => DecodeSlice(frames, decoded, end, Math.Min(CsmAdapter.CodecContextFrames, decoded)), cancellationToken).ConfigureAwait(false);
                    decoded = end;
                }
            }

            if (frames.Count > decoded)
            {
                yield return await Task.Run(() => DecodeSlice(frames, decoded, frames.Count, Math.Min(CsmAdapter.CodecContextFrames, decoded)), cancellationToken).ConfigureAwait(false);
            }
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>The token ids of the request's text turn: <c>&lt;|begin_of_text|&gt;[speaker]text&lt;|end_of_text|&gt;</c>.</summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="voice">The speaker number, or <see langword="null"/> for 0.</param>
    public int[] TokensFor(string text, string? voice = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        string speaker = voice ?? Voices[0];
        if (!int.TryParse(speaker, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out _))
        {
            throw new ArgumentException($"CSM voices are speaker numbers such as 0 and 1, not '{speaker}'.", nameof(voice));
        }

        return _llm.Tokenizer.Encode($"<|begin_of_text|>[{speaker}]{text}<|end_of_text|>", addSpecial: false, parseSpecial: true);
    }

    /// <summary>The backbone's prompt rows [rows, hidden] for a request.</summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="voice">The speaker number, or <see langword="null"/> for 0.</param>
    public float[] PromptFor(string text, string? voice = null)
    {
        var tables = _tables.Value;
        var ids = TokensFor(text, voice);
        var prompt = new float[ids.Length * tables.Hidden];
        for (int i = 0; i < ids.Length; i++)
        {
            tables.TextRow(ids[i]).CopyTo(prompt, i * tables.Hidden);
        }

        return prompt;
    }

    /// <summary>
    /// Generates the codec frames of a request (32 codes each), one by one; the backbone's first-codebook logits of each step and the depth decoder's logits of the frame go to
    /// <paramref name="observe"/> when given.
    /// </summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="options">Request settings (voice, sampling), or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Stops generation between frames.</param>
    /// <param name="forced">Feeds these frames instead of the generated ones (teacher forcing), or <see langword="null"/>.</param>
    /// <param name="observe">Receives each step's backbone logits, hidden state and the depth decoder's logits of the frame, or <see langword="null"/>.</param>
    public IEnumerable<int[]> GenerateFrames(string text, SpeechSynthesisOptions? options, CancellationToken cancellationToken, IReadOnlyList<int[]>? forced = null,
        Action<float[], float[], IReadOnlyList<float[]>>? observe = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var tables = _tables.Value;
        int h = tables.Hidden;
        var sampling = options.SamplingOr(_defaults);
        var depthSampling = sampling.Temperature <= 0f ? new SamplingOptions { Temperature = 0f, RepetitionPenalty = 1f } : _depthDefaults with { Seed = sampling.Seed };
        var sampler = new Sampler(sampling with { RepetitionPenalty = 1f });
        var depthSampler = new Sampler(depthSampling);
        var backbone = _backbone ??= CreateSession(WellKnown.DecodeMultimodalGraph);
        var depth = _depth ??= CreateSession(CsmAdapter.DepthGraph);
        backbone.Reset();
        var prompt = PromptFor(text, options?.Voice);
        var logits = Process(backbone, prompt, prompt.Length / h);
        var hidden = backbone.LastHidden[^h..].ToArray();
        int budget = Math.Min(_maxFrames, backbone.ContextLength - (prompt.Length / h) - 1);
        for (int step = 0; step < budget; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var depthLogits = observe is null ? null : new List<float[]>();
            int first = forced is not null ? (step < forced.Count ? forced[step][0] : 0) : sampler.Sample(logits);
            var frame = new int[_codebooks];
            frame[0] = first;
            depth.Reset();
            var rows = new float[2 * tables.DepthHidden];
            tables.Project(hidden).CopyTo(rows, 0);
            tables.Project(tables.Embedding(0, first)).CopyTo(rows, tables.DepthHidden);
            Process(depth, rows, 2);
            for (int g = 1; g < _codebooks; g++)
            {
                var head = tables.Head(g - 1, depth.LastHidden[^tables.DepthHidden..]);
                depthLogits?.Add(head);
                frame[g] = forced is not null ? (step < forced.Count ? forced[step][g] : 0) : depthSampler.Sample(head);
                if (g < _codebooks - 1)
                {
                    Process(depth, tables.Project(tables.Embedding(g, frame[g])), 1);
                }
            }

            observe?.Invoke(logits, hidden, depthLogits!);
            if (frame.AsSpan(0, _codebooks - 1).ToArray().All(c => c == 0))
            {
                yield break;
            }

            yield return frame;
            var next = new float[h];
            for (int g = 0; g < _codebooks; g++)
            {
                Add(next, tables.Embedding(g, frame[g]));
            }

            logits = Process(backbone, next, 1);
            hidden = backbone.LastHidden[^h..].ToArray();
        }
    }

    /// <summary>Decodes frames of 32 codes into audio (chunks of <see cref="CsmAdapter.CodecChunkFrames"/> with <see cref="CsmAdapter.CodecContextFrames"/> frames of context).</summary>
    /// <param name="frames">The frames.</param>
    /// <param name="cancellationToken">Stops between chunks.</param>
    public AudioBuffer Decode(IReadOnlyList<int[]> frames, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(frames);
        var samples = new List<float>(frames.Count * _frameSamples);
        for (int start = 0; start < frames.Count; start += CsmAdapter.CodecChunkFrames)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int end = Math.Min(frames.Count, start + CsmAdapter.CodecChunkFrames);
            samples.AddRange(DecodeSlice(frames, start, end, Math.Min(CsmAdapter.CodecContextFrames, start)).Samples.ToArray());
        }

        return new AudioBuffer([.. samples], SampleRate);
    }

    /// <summary>Decodes frames <c>[first, end)</c> with <paramref name="context"/> frames before them whose audio is dropped.</summary>
    private AudioBuffer DecodeSlice(IReadOnlyList<int[]> frames, int first, int end, int context)
    {
        int start = first - context;
        int count = end - start;
        if (count <= 0)
        {
            return new AudioBuffer([], SampleRate);
        }

        _codec ??= _model.CompileComponent(CsmAdapter.CodecRole, AudioCodecGraphs.DecodeGraph);
        var mask = new float[_codebooks];
        mask.AsSpan().Fill(1f);
        _codec.SetInput<float>(Input(AudioCodecGraphs.MaskInput), mask);
        for (int q = 0; q < _codebooks; q++)
        {
            var codes = new int[count];
            for (int f = 0; f < count; f++)
            {
                codes[f] = Math.Max(0, frames[start + f][q]);
            }

            _codec.SetInput<int>(Input(AudioCodecGraphs.CodesName(q)), codes);
        }

        int rows = count * _positionsPerFrame;
        var positions = new int[rows];
        var spans = new int[2 * rows];
        for (int i = 0; i < rows; i++)
        {
            positions[i] = i;
            spans[(2 * i) + 1] = rows;
        }

        _codec.SetInput<int>(Input(AudioCodecGraphs.PositionsInput), positions);
        _codec.SetInput<int>(Input(AudioCodecGraphs.SpansInput), spans);
        var parameters = new RunParameters();
        parameters.Symbols[WellKnown.TokenSymbol] = count;
        _codec.Run(parameters);
        var samples = new float[count * _frameSamples];
        _codec.GetOutput<float>(AudioCodecGraphs.AudioOutput, samples);
        return new AudioBuffer(samples[(context * _frameSamples)..], SampleRate);
    }

    private Value Input(string name) => _codec!.Graph.Inputs.First(i => i.Input!.Name == name);

    private InferenceSession CreateSession(string graph)
    {
        var session = _llm.CreateSession(graph);
        session.CaptureHidden = true;
        return session;
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

    private static void Add(Span<float> target, ReadOnlySpan<float> values)
    {
        for (int i = 0; i < target.Length; i++)
        {
            target[i] += values[i];
        }
    }

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var sum = Vector<float>.Zero;
        int i = 0;
        for (; i <= a.Length - Vector<float>.Count; i += Vector<float>.Count)
        {
            sum += new Vector<float>(a[i..]) * new Vector<float>(b[i..]);
        }

        float total = Vector.Sum(sum);
        for (; i < a.Length; i++)
        {
            total += a[i] * b[i];
        }

        return total;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _backbone?.Dispose();
        _depth?.Dispose();
        _codec?.Dispose();
        _gate.Dispose();
    }

    /// <summary>The host-side tables: the text embedding and the codebook embedding (rows read on demand), the depth decoder's projector and its codebook heads.</summary>
    private sealed class Tables
    {
        private const string Prefix = CsmAdapter.Architecture + ".";
        private readonly WeightInfo _text;
        private readonly WeightInfo _audio;
        private readonly float[] _projector;
        private readonly WeightInfo _headWeight;
        private readonly float[]?[] _heads;
        private readonly int _vocabulary;

        public Tables(ModelDefinition definition, int codebooks, int vocabulary)
        {
            var w = definition.Weights;
            _vocabulary = vocabulary;
            _text = w[Prefix + "text_embedding.weight"];
            _audio = w[Prefix + "audio_embedding.weight"];
            Hidden = (int)_text.Shape[1].Fixed;
            _projector = WeightConversions.ReadFloats(w[Prefix + "depth.projector.weight"]);
            DepthHidden = _projector.Length / Hidden;
            _headWeight = w[Prefix + "depth.heads.weight"];
            _heads = new float[codebooks - 1][];
        }

        public int Hidden { get; }

        public int DepthHidden { get; }

        public float[] TextRow(int token) => ReadRow(_text, token, Hidden);

        // The embedding of a code of codebook <paramref name="codebook"/>.
        public float[] Embedding(int codebook, int code) => ReadRow(_audio, (codebook * _vocabulary) + code, Hidden);

        public float[] Project(ReadOnlySpan<float> row)
        {
            var y = new float[DepthHidden];
            for (int o = 0; o < y.Length; o++)
            {
                y[o] = Dot(_projector.AsSpan(o * Hidden, Hidden), row);
            }

            return y;
        }

        // The logits of codebook head <paramref name="head"/> (stored [depth width, vocabulary]) for a hidden state.
        public float[] Head(int head, ReadOnlySpan<float> hidden)
        {
            var weight = _heads[head] ??= ReadHead(head);
            var y = new float[_vocabulary];
            for (int k = 0; k < hidden.Length; k++)
            {
                float v = hidden[k];
                var row = weight.AsSpan(k * _vocabulary, _vocabulary);
                int i = 0;
                var scale = new Vector<float>(v);
                for (; i <= y.Length - Vector<float>.Count; i += Vector<float>.Count)
                {
                    (new Vector<float>(y.AsSpan(i)) + (scale * new Vector<float>(row[i..]))).CopyTo(y.AsSpan(i));
                }

                for (; i < y.Length; i++)
                {
                    y[i] += v * row[i];
                }
            }

            return y;
        }

        private float[] ReadHead(int head)
        {
            long count = (long)DepthHidden * _vocabulary;
            return ReadFloats(_headWeight, head * count, count);
        }

        private static float[] ReadRow(WeightInfo weight, int row, int width) => ReadFloats(weight, (long)row * width, width);

        private static float[] ReadFloats(WeightInfo weight, long first, long count)
        {
            var values = new float[count];
            if (weight.Type == DataType.F32)
            {
                weight.Source.Read(first * sizeof(float), MemoryMarshal.AsBytes(values.AsSpan()));
            }
            else
            {
                var raw = new ushort[count];
                weight.Source.Read(first * sizeof(ushort), MemoryMarshal.AsBytes(raw.AsSpan()));
                if (weight.Type == DataType.BF16)
                {
                    for (int i = 0; i < raw.Length; i++)
                    {
                        values[i] = BitConverter.Int32BitsToSingle(raw[i] << 16);
                    }
                }
                else if (weight.Type == DataType.F16)
                {
                    for (int i = 0; i < raw.Length; i++)
                    {
                        values[i] = (float)BitConverter.UInt16BitsToHalf(raw[i]);
                    }
                }
                else
                {
                    throw new NotSupportedException($"CSM: weight '{weight.Name}' of type {weight.Type} is not supported.");
                }
            }

            return values;
        }
    }
}
