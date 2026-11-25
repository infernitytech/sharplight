using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Tokenizers;

namespace Sharplight.Audio;

/// <summary>The built-in <see cref="IMusicGenerator"/> over the <see cref="MusicGenerationGraphs"/> contract: codebook streams with a delay pattern, classifier-free guidance, codec decoding.</summary>
internal sealed class MusicGeneratorClient : IMusicGenerator
{
    private const int StreamChunkFrames = 50;
    private const int StreamContextFrames = 25;
    private const int StreamLookaheadFrames = 10;

    private readonly TaskModel _model;
    private readonly ITokenizer _tokenizer;
    private readonly CompiledModel _text;
    private readonly CompiledModel _step;
    private readonly CompiledModel _unconditional;
    private readonly CompiledModel _codec;
    private readonly Graph.Value _tokens;
    private readonly Graph.Value _condition;
    private readonly Graph.Value[] _stepCodes;
    private readonly Graph.Value[] _unconditionalCodes;
    private readonly Graph.Value _stepPositions;
    private readonly Graph.Value _unconditionalPositions;
    private readonly Graph.Value _codecMask;
    private readonly Graph.Value[] _codecCodes;
    private readonly int _codebooks;
    private readonly int _vocabulary;
    private readonly int _maxSteps;
    private readonly int _maxTokens;
    private readonly int _conditionWidth;
    private readonly int _samplesPerFrame;
    private readonly int _codecFrames;
    private readonly int _codecCodebooks;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _disposed;

    public MusicGeneratorClient(TaskModel model)
    {
        _model = model;
        var decoder = model.Description.Component(ModelComponentRoles.Decoder);
        var text = model.Description.Component(ModelComponentRoles.TextEncoder);
        var codec = model.Description.Component(ModelComponentRoles.Vocoder);
        _codebooks = decoder.ConfigValue(MusicGenerationGraphs.CodebooksKey, 0);
        _vocabulary = decoder.ConfigValue(MusicGenerationGraphs.VocabularyKey, 0);
        _maxSteps = decoder.ConfigValue(MusicGenerationGraphs.MaxStepsKey, 0);
        Defaults = new MusicGenerationDefaults((float)decoder.ConfigValue(MusicGenerationGraphs.GuidanceKey, 3.0), Temperature: 1f, TopK: 250, TopP: 1f);
        SampleRate = decoder.ConfigValue(MusicGenerationGraphs.SampleRateKey, 0);
        FrameRate = decoder.ConfigValue(MusicGenerationGraphs.FrameRateKey, 0.0);
        _maxTokens = text.ConfigValue(MusicGenerationGraphs.MaxTokensKey, 512);
        _conditionWidth = text.ConfigValue(MusicGenerationGraphs.ConditionWidthKey, 0);
        _samplesPerFrame = codec.ConfigValue(AudioCodecGraphs.SamplesPerFrameKey, 0);
        _codecCodebooks = codec.ConfigValue(AudioCodecGraphs.CodebooksKey, 0);
        _codecFrames = (int)codec.SymbolBounds.GetValueOrDefault(WellKnown.TokenSymbol, 1500);
        _tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
        _text = model.CompileComponent(ModelComponentRoles.TextEncoder, MusicGenerationGraphs.TextGraph);
        _step = model.CompileComponent(ModelComponentRoles.Decoder, MusicGenerationGraphs.StepGraph);
        _unconditional = model.CompileComponent(ModelComponentRoles.Decoder, MusicGenerationGraphs.UnconditionalStepGraph);
        _codec = model.CompileComponent(ModelComponentRoles.Vocoder, AudioCodecGraphs.DecodeGraph);
        _tokens = _text.Graph.Inputs.First(v => v.Input!.Name == MusicGenerationGraphs.TokensInput);
        _condition = _step.Graph.Inputs.First(v => v.Input!.Name == MusicGenerationGraphs.ConditionInput);
        _stepCodes = [.. Enumerable.Range(0, _codebooks).Select(q => _step.Graph.Inputs.First(v => v.Input!.Name == MusicGenerationGraphs.CodeInput(q)))];
        _unconditionalCodes = [.. Enumerable.Range(0, _codebooks).Select(q => _unconditional.Graph.Inputs.First(v => v.Input!.Name == MusicGenerationGraphs.CodeInput(q)))];
        _stepPositions = _step.Graph.Inputs.First(v => v.Input!.Name == "positions");
        _unconditionalPositions = _unconditional.Graph.Inputs.First(v => v.Input!.Name == "positions");
        _codecMask = _codec.Graph.Inputs.First(v => v.Input!.Name == AudioCodecGraphs.MaskInput);
        _codecCodes = [.. Enumerable.Range(0, _codecCodebooks).Select(q => _codec.Graph.Inputs.First(v => v.Input!.Name == AudioCodecGraphs.CodesName(q)))];
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public MusicGenerationDefaults Defaults { get; }

    public int SampleRate { get; }

    public double FrameRate { get; }

    public TimeSpan MaxDuration => TimeSpan.FromSeconds((_maxSteps - _codebooks) / FrameRate);

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    public Task<CodecCodes> GenerateCodesAsync(MusicGenerationRequest request, MusicGenerationOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ObjectDisposedException.ThrowIf(_disposed, this);
        options ??= MusicGenerationOptions.Default;
        long seed = AudioGenerationStreams.ResolveSeed(options.Seed);
        return BackgroundCompute.RunAsync(_gate, () =>
        {
            var session = new Session(this, request, options, seed, cancellationToken);
            session.StepUntil(session.Frames);
            return session.Codes(0, session.Frames);
        }, cancellationToken);
    }

    public Task<AudioGenerationResult> GenerateAsync(MusicGenerationRequest request, MusicGenerationOptions? options = null, CancellationToken cancellationToken = default)
        => AudioGenerationStreams.CompleteAsync(Stream(request, options, pieces: false, cancellationToken));

    public IAsyncEnumerable<AudioGenerationUpdate> GenerateStreamingAsync(MusicGenerationRequest request, MusicGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
        => Stream(request, options, pieces: true, cancellationToken);

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _text.Dispose();
        _step.Dispose();
        _unconditional.Dispose();
        _codec.Dispose();
        _gate.Dispose();
    }

    /// <summary>
    /// Generates in chunks of <see cref="StreamChunkFrames"/> frames with an update after each; with <paramref name="pieces"/> every update carries the audio of
    /// its chunk (decoded with context and lookahead frames). The last update carries the whole audio, decoded at once.
    /// </summary>
    private IAsyncEnumerable<AudioGenerationUpdate> Stream(MusicGenerationRequest request, MusicGenerationOptions? options, bool pieces, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ObjectDisposedException.ThrowIf(_disposed, this);
        options ??= MusicGenerationOptions.Default;
        long seed = AudioGenerationStreams.ResolveSeed(options.Seed);
        return AudioGenerationStreams.Stream(_gate, (emit, token) =>
        {
            var session = new Session(this, request, options, seed, token);
            int emitted = 0;
            while (emitted < session.Frames)
            {
                int target = Math.Min(session.Frames, emitted + StreamChunkFrames + (pieces ? StreamLookaheadFrames : 0));
                session.StepUntil(target);
                int end = target == session.Frames || !pieces ? target : target - StreamLookaheadFrames;
                AudioBuffer? piece = null;
                if (pieces)
                {
                    int from = Math.Max(0, emitted - StreamContextFrames);
                    var audio = Decode(session.Codes(from, target - from), token);
                    int skip = (emitted - from) * _samplesPerFrame;
                    int count = (end - emitted) * _samplesPerFrame;
                    piece = new AudioBuffer(audio.Samples.AsSpan(skip, count).ToArray(), SampleRate);
                }

                emit(new AudioGenerationUpdate(end, session.Frames, Audio: piece));
                emitted = end;
            }

            emit(new AudioGenerationUpdate(session.Frames, session.Frames, new AudioGenerationResult(Decode(session.Codes(0, session.Frames), token), seed)));
        }, cancellationToken);
    }

    private AudioBuffer Decode(CodecCodes codes, CancellationToken cancellationToken)
    {
        var mask = new float[_codecCodebooks];
        mask.AsSpan(0, Math.Min(codes.Codebooks, _codecCodebooks)).Fill(1f);
        var audio = new float[codes.Frames * _samplesPerFrame];
        for (int start = 0; start < codes.Frames; start += _codecFrames)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(_codecFrames, codes.Frames - start);
            _codec.SetInput<float>(_codecMask, mask);
            var zero = new int[count];
            for (int q = 0; q < _codecCodebooks; q++)
            {
                _codec.SetInput<int>(_codecCodes[q], q < codes.Codebooks ? codes.Codebook(q).Slice(start, count).ToArray() : zero);
            }

            var parameters = new RunParameters();
            parameters.Symbols[WellKnown.TokenSymbol] = count;
            _codec.Run(parameters);
            var part = new float[count * _samplesPerFrame];
            _codec.GetOutput<float>(AudioCodecGraphs.AudioOutput, part);
            part.CopyTo(audio.AsSpan(start * _samplesPerFrame));
        }

        return new AudioBuffer(audio, SampleRate);
    }

    /// <summary>The state of one generation: the delayed code streams and the step the decoders are at.</summary>
    private sealed class Session
    {
        private readonly MusicGeneratorClient _client;
        private readonly MusicGenerationRequest _request;
        private readonly bool _greedy;
        private readonly int _topK;
        private readonly float _topP;
        private readonly float _temperature;
        private readonly CancellationToken _cancellationToken;
        private readonly int[][] _delayed;
        private readonly Random _random;
        private readonly float _guidance;
            private readonly int _conditionRows;
        private float[] _condition = [];
        private readonly float[] _guided;
        private readonly float[] _conditionalLogits;
        private readonly float[] _unconditionalLogits;
        private readonly int _length;
        private int _step;

        public Session(MusicGeneratorClient client, MusicGenerationRequest request, MusicGenerationOptions options, long seed, CancellationToken cancellationToken)
        {
            _client = client;
            _request = request;
            _cancellationToken = cancellationToken;
            int k = client._codebooks;
            Frames = (int)Math.Round(request.Duration.TotalSeconds * client.FrameRate);
            if (Frames < 1 || Frames + k > client._maxSteps)
            {
                throw new ArgumentOutOfRangeException(nameof(request), $"The duration must be between one frame and {client.MaxDuration.TotalSeconds:0.##} s.");
            }

            _length = Frames + k;
            _random = Seeds.CreateRandom(seed);
            _guidance = options.GuidanceScale ?? client.Defaults.GuidanceScale;
            _greedy = options.Greedy;
            _topK = options.TopK ?? client.Defaults.TopK;
            _topP = options.TopP ?? client.Defaults.TopP;
            _temperature = options.Temperature ?? client.Defaults.Temperature;
            _delayed = new int[k][];
            for (int q = 0; q < k; q++)
            {
                _delayed[q] = new int[_length];
                Array.Fill(_delayed[q], client._vocabulary);
            }

            _guided = new float[client._vocabulary];
            _conditionalLogits = new float[k * client._vocabulary];
            _unconditionalLogits = new float[k * client._vocabulary];
            _conditionRows = Encode();
        }

        public int Frames { get; }

        /// <summary>Runs decoder steps until the first <paramref name="frames"/> frames of every codebook are complete.</summary>
        public void StepUntil(int frames)
        {
            int k = _client._codebooks;
            int steps = Math.Min(_length - 1, frames + k - 1);
            while (_step < steps)
            {
                _cancellationToken.ThrowIfCancellationRequested();
                Step();
            }
        }

        /// <summary>The codes of <paramref name="count"/> frames from <paramref name="start"/> with the delay pattern removed.</summary>
        public CodecCodes Codes(int start, int count)
        {
            int k = _client._codebooks;
            var values = new int[k * count];
            for (int q = 0; q < k; q++)
            {
                _delayed[q].AsSpan(start + q + 1, count).CopyTo(values.AsSpan(q * count, count));
            }

            return new CodecCodes(k, count, values);
        }

        private int Encode()
        {
            var ids = _client._tokenizer.Encode(_request.Prompt);
            if (ids.Length > _client._maxTokens)
            {
                int eos = _client._tokenizer.EosId ?? -1;
                ids = ids[.._client._maxTokens];
                if (eos >= 0 && _client._tokenizer.AddEos)
                {
                    ids[^1] = eos;
                }
            }

            _client._text.SetInput<int>(_client._tokens, ids);
            var parameters = new RunParameters();
            parameters.Symbols[WellKnown.TokenSymbol] = ids.Length;
            _client._text.Run(parameters);
            var condition = new float[ids.Length * _client._conditionWidth];
            _client._text.GetOutput<float>(MusicGenerationGraphs.ConditionOutput, condition);
            _condition = condition;
            return ids.Length;
        }

        private void Step()
        {
            int k = _client._codebooks;
            int vocabulary = _client._vocabulary;
            int t = _step;
            var parameters = new RunParameters();
            parameters.Symbols[WellKnown.TokenSymbol] = 1;
            parameters.Symbols[WellKnown.EncoderSymbol] = _conditionRows;
            parameters.Scalars[WellKnown.PastScalar] = t;
            for (int q = 0; q < k; q++)
            {
                _client._step.SetInput<int>(_client._stepCodes[q], [_delayed[q][t]]);
                _client._unconditional.SetInput<int>(_client._unconditionalCodes[q], [_delayed[q][t]]);
            }

            _client._step.SetInput<float>(_client._condition, _condition);
            _client._step.SetInput<int>(_client._stepPositions, [t]);
            _client._unconditional.SetInput<int>(_client._unconditionalPositions, [t]);
            var plain = new RunParameters();
            plain.Symbols[WellKnown.TokenSymbol] = 1;
            plain.Scalars[WellKnown.PastScalar] = t;
            _client._step.Run(parameters);
            _client._step.GetOutput<float>(MusicGenerationGraphs.LogitsOutput, _conditionalLogits);
            bool guide = _guidance != 1f;
            if (guide)
            {
                _client._unconditional.Run(plain);
                _client._unconditional.GetOutput<float>(MusicGenerationGraphs.LogitsOutput, _unconditionalLogits);
            }

            int next = t + 1;
            for (int q = 0; q < k; q++)
            {
                // The delay pattern: the first codebook.Index + 1 positions and the trailing positions are padding.
                if (next <= q || next >= _length - k + 1 + q)
                {
                    _delayed[q][next] = vocabulary;
                    continue;
                }

                for (int v = 0; v < vocabulary; v++)
                {
                    float c = _conditionalLogits[(q * vocabulary) + v];
                    _guided[v] = guide ? _unconditionalLogits[(q * vocabulary) + v] + (_guidance * (c - _unconditionalLogits[(q * vocabulary) + v])) : c;
                }

                _delayed[q][next] = _greedy ? Sampler.ArgMax(_guided) : Sample(_guided);
            }

            _step++;
        }

        private int Sample(float[] logits)
        {
            int n = logits.Length;
            var order = new int[n];
            for (int i = 0; i < n; i++)
            {
                order[i] = i;
            }

            Array.Sort(order, (a, b) => logits[b].CompareTo(logits[a]));
            int keep = _topK > 0 ? Math.Min(_topK, n) : n;
            float temperature = Math.Max(_temperature, 1e-4f);
            var probabilities = new double[keep];
            double max = logits[order[0]];
            double sum = 0;
            for (int i = 0; i < keep; i++)
            {
                probabilities[i] = Math.Exp((logits[order[i]] - max) / temperature);
                sum += probabilities[i];
            }

            double cumulative = 0;
            int count = keep;
            for (int i = 0; i < keep; i++)
            {
                cumulative += probabilities[i] / sum;
                if (_topP < 1f && cumulative >= _topP)
                {
                    count = i + 1;
                    break;
                }
            }

            double total = 0;
            for (int i = 0; i < count; i++)
            {
                total += probabilities[i];
            }

            double draw = _random.NextDouble() * total;
            for (int i = 0; i < count; i++)
            {
                draw -= probabilities[i];
                if (draw <= 0)
                {
                    return order[i];
                }
            }

            return order[count - 1];
        }
    }
}
