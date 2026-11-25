using System.Runtime.InteropServices;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Tokenizers;

namespace Sharplight.Audio;

/// <summary>
/// The built-in <see cref="IAudioGenerator"/> over the <see cref="AudioGenerationGraphs"/> contract: caption and lyric conditioning, a flow-matching Euler loop over the
/// latents (the source latents are the model's silence latent, the chunk mask is the model's constant), and the autoencoder's decoder, run in overlapping windows for long audio.
/// </summary>
internal sealed class AudioGeneratorClient : IAudioGenerator
{
    private readonly TaskModel _model;
    private readonly ITokenizer _tokenizer;
    private readonly CompiledModel _text;
    private readonly CompiledModel _lyric;
    private readonly CompiledModel _condition;
    private readonly CompiledModel _denoise;
    private readonly CompiledModel _decode;
    private readonly float[] _silence;
    private readonly string _instruction;
    private readonly int _maxTextTokens;
    private readonly int _maxLyricTokens;
    private readonly int _textWidth;
    private readonly int _conditionWidth;
    private readonly int _latentChannels;
    private readonly int _contextChannels;
    private readonly int _silenceFrames;
    private readonly int _referenceFrames;
    private readonly int _patch;
    private readonly int _maxFrames;
    private readonly int _samplesPerFrame;
    private readonly int _decodeFrames;
    private readonly int _decodeOverlap;
    private readonly float _chunkMask;
    private readonly float _defaultShift;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _disposed;

    public AudioGeneratorClient(TaskModel model)
    {
        _model = model;
        var text = model.Description.Component(ModelComponentRoles.TextEncoder);
        var denoiser = model.Description.Component(ModelComponentRoles.Denoiser);
        var encoder = model.Description.Component(ModelComponentRoles.Encoder);
        var vae = model.Description.Component(ModelComponentRoles.Vae);
        _maxTextTokens = text.ConfigValue(AudioGenerationGraphs.MaxTextTokensKey, 256);
        _maxLyricTokens = text.ConfigValue(AudioGenerationGraphs.MaxLyricTokensKey, 2048);
        _textWidth = text.ConfigValue(AudioGenerationGraphs.TextWidthKey, 0);
        _conditionWidth = encoder.ConfigValue(AudioGenerationGraphs.ConditionWidthKey, 0);
        _instruction = text.Config.TryGetValue(AudioGenerationGraphs.InstructionKey, out var instruction) ? instruction.GetString() ?? string.Empty : string.Empty;
        SampleRate = denoiser.ConfigValue(AudioGenerationGraphs.SampleRateKey, 0);
        Channels = denoiser.ConfigValue(AudioGenerationGraphs.ChannelsKey, 1);
        _samplesPerFrame = denoiser.ConfigValue(AudioGenerationGraphs.SamplesPerFrameKey, 1);
        LatentChannels = denoiser.ConfigValue(AudioGenerationGraphs.LatentChannelsKey, 0);
        _latentChannels = LatentChannels;
        _contextChannels = denoiser.ConfigValue(AudioGenerationGraphs.ContextChannelsKey, 0);
        _silenceFrames = denoiser.ConfigValue(AudioGenerationGraphs.SilenceFramesKey, 0);
        _chunkMask = (float)denoiser.ConfigValue(AudioGenerationGraphs.ChunkMaskValueKey, 1.0);
        _patch = denoiser.ConfigValue(AudioGenerationGraphs.PatchSizeKey, 1);
        _maxFrames = denoiser.ConfigValue(AudioGenerationGraphs.MaxFramesKey, 0);
        Defaults = new AudioGenerationDefaults(denoiser.ConfigValue(AudioGenerationGraphs.DefaultStepsKey, 8), null, SupportsLyrics: true, SupportsShift: true);
        _defaultShift = (float)denoiser.ConfigValue(AudioGenerationGraphs.DefaultShiftKey, 1.0);
        _referenceFrames = encoder.ConfigValue(AudioGenerationGraphs.ReferenceFramesKey, 0);
        _decodeFrames = vae.ConfigValue(AudioGenerationGraphs.DecodeFramesKey, 256);
        _decodeOverlap = vae.ConfigValue(AudioGenerationGraphs.DecodeOverlapKey, 16);
        LatentFrameRate = (double)SampleRate / _samplesPerFrame;
        _silence = MemoryMarshal.Cast<byte, float>(model.ReadAttachment(ModelComponentRoles.Denoiser, AudioGenerationGraphs.SilenceAttachment)).ToArray();
        _tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
        _text = model.CompileComponent(ModelComponentRoles.TextEncoder, AudioGenerationGraphs.TextGraph);
        _lyric = model.CompileComponent(ModelComponentRoles.TextEncoder, AudioGenerationGraphs.LyricGraph);
        _condition = model.CompileComponent(ModelComponentRoles.Encoder, AudioGenerationGraphs.ConditionGraph);
        _denoise = model.CompileComponent(ModelComponentRoles.Denoiser, AudioGenerationGraphs.DenoiseGraph);
        _decode = model.CompileComponent(ModelComponentRoles.Vae, AudioGenerationGraphs.DecodeGraph);
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public int SampleRate { get; }

    public int Channels { get; }

    public double LatentFrameRate { get; }

    public int LatentChannels { get; }

    public AudioGenerationDefaults Defaults { get; }

    public TimeSpan MaxDuration => TimeSpan.FromSeconds(_maxFrames / LatentFrameRate);

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    public Task<AudioLatents> GenerateLatentsAsync(AudioGenerationRequest request, AudioGenerationOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ObjectDisposedException.ThrowIf(_disposed, this);
        options ??= AudioGenerationOptions.Default;
        long seed = AudioGenerationStreams.ResolveSeed(options.Seed);
        return BackgroundCompute.RunAsync(_gate, () => Denoise(request, options, seed, null, cancellationToken), cancellationToken);
    }

    public Task<AudioGenerationResult> GenerateAsync(AudioGenerationRequest request, AudioGenerationOptions? options = null, CancellationToken cancellationToken = default)
        => AudioGenerationStreams.CompleteAsync(GenerateStreamingAsync(request, options, cancellationToken));

    public IAsyncEnumerable<AudioGenerationUpdate> GenerateStreamingAsync(AudioGenerationRequest request, AudioGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ObjectDisposedException.ThrowIf(_disposed, this);
        options ??= AudioGenerationOptions.Default;
        long seed = AudioGenerationStreams.ResolveSeed(options.Seed);
        return AudioGenerationStreams.Stream(_gate, (emit, token) =>
        {
            int done = 0;
            int total = 0;
            var latents = Denoise(request, options, seed, (step, steps) =>
            {
                (done, total) = (step, steps);
                emit(new AudioGenerationUpdate(step, steps));
            }, token);
            emit(new AudioGenerationUpdate(done, total, new AudioGenerationResult(Decode(latents, token), seed)));
        }, cancellationToken);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _text.Dispose();
        _lyric.Dispose();
        _condition.Dispose();
        _denoise.Dispose();
        _decode.Dispose();
        _gate.Dispose();
    }

    private AudioLatents Denoise(AudioGenerationRequest request, AudioGenerationOptions options, long seed, Action<int, int>? progress, CancellationToken cancellationToken)
    {
        int frames = (int)Math.Round(request.Duration.TotalSeconds * LatentFrameRate);
        if (frames < 1 || frames > _maxFrames)
        {
            throw new ArgumentOutOfRangeException(nameof(request), $"The duration must be between one frame and {MaxDuration.TotalSeconds:0.##} s.");
        }

        int steps = options.Steps ?? Defaults.Steps;
        if (steps < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "The number of steps must be positive.");
        }

        int padded = (frames + _patch - 1) / _patch * _patch;
        float[] condition = Condition(request, out int conditionRows);

        float[] noise;
        if (options.InitialNoise is { } given)
        {
            if (given.Length != frames * _latentChannels)
            {
                throw new ArgumentException($"The initial noise has {given.Length} values, {frames * _latentChannels} are needed.", nameof(options));
            }

            noise = given.ToArray();
        }
        else
        {
            noise = new float[frames * _latentChannels];
            AudioNoise.Fill(AudioNoise.Create(seed), noise);
        }

        // Context: the silence latent as source audio and the chunk mask, zero rows for the padding.
        var context = new float[padded * _contextChannels];
        int sourceChannels = _contextChannels - _latentChannels;
        for (int f = 0; f < frames; f++)
        {
            for (int c = 0; c < sourceChannels; c++)
            {
                context[(f * _contextChannels) + c] = _silence[((f % _silenceFrames) * sourceChannels) + c];
            }

            for (int c = sourceChannels; c < _contextChannels; c++)
            {
                context[(f * _contextChannels) + c] = _chunkMask;
            }
        }

        var latent = new float[padded * _latentChannels];
        noise.CopyTo(latent, 0);
        var velocity = new float[padded * _latentChannels];
        var positions = new int[padded / _patch];
        for (int i = 0; i < positions.Length; i++)
        {
            positions[i] = i;
        }

        var inputs = new
        {
            Latent = _denoise.Graph.Inputs.First(v => v.Input!.Name == AudioGenerationGraphs.LatentInput),
            Context = _denoise.Graph.Inputs.First(v => v.Input!.Name == AudioGenerationGraphs.ContextInput),
            Time = _denoise.Graph.Inputs.First(v => v.Input!.Name == AudioGenerationGraphs.TimestepInput),
            TimeR = _denoise.Graph.Inputs.First(v => v.Input!.Name == AudioGenerationGraphs.TimestepRInput),
            Condition = _denoise.Graph.Inputs.First(v => v.Input!.Name == AudioGenerationGraphs.ConditionInput),
            Positions = _denoise.Graph.Inputs.First(v => v.Input!.Name == AudioGenerationGraphs.PositionsInput),
        };
        float shift = options.Shift ?? _defaultShift;
        var parameters = new RunParameters();
        parameters.Symbols[WellKnown.TokenSymbol] = padded;
        parameters.Symbols[WellKnown.EncoderSymbol] = conditionRows;
        float t = Time(0, steps, shift);
        for (int step = 0; step < steps; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            float next = step + 1 < steps ? Time(step + 1, steps, shift) : 0f;
            _denoise.SetInput<float>(inputs.Latent, latent);
            _denoise.SetInput<float>(inputs.Context, context);
            _denoise.SetInput<float>(inputs.Time, [t]);
            _denoise.SetInput<float>(inputs.TimeR, [t]);
            _denoise.SetInput<float>(inputs.Condition, condition);
            _denoise.SetInput<int>(inputs.Positions, positions);
            _denoise.Run(parameters);
            _denoise.GetOutput<float>(AudioGenerationGraphs.VelocityOutput, velocity);
            float dt = t - next;
            for (int i = 0; i < frames * _latentChannels; i++)
            {
                latent[i] -= velocity[i] * dt;
            }

            t = next;
            progress?.Invoke(step + 1, steps);
        }

        return new AudioLatents(frames, _latentChannels, latent[..(frames * _latentChannels)]);
    }

    // The timestep of step k of n: a uniform grid from 1 down, bent by the shift (shift · s / (1 + (shift - 1) · s)).
    private static float Time(int step, int steps, float shift)
    {
        float s = (float)(steps - step) / steps;
        return shift * s / (1f + ((shift - 1f) * s));
    }

    private float[] Condition(AudioGenerationRequest request, out int rows)
    {
        string metas = $"- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: {(int)request.Duration.TotalSeconds} seconds\n";
        string prompt = $"# Instruction\n{_instruction}\n\n# Caption\n{request.Prompt}\n\n# Metas\n{metas}<|endoftext|>\n";
        string lyrics = string.IsNullOrWhiteSpace(request.Lyrics) ? "[Instrumental]" : request.Lyrics;
        string language = string.IsNullOrWhiteSpace(request.Language) ? "unknown" : request.Language;
        string lyricPrompt = $"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>";
        var textIds = Truncate(_tokenizer.Encode(prompt), _maxTextTokens);
        var lyricIds = Truncate(_tokenizer.Encode(lyricPrompt), _maxLyricTokens);
        int width = _textWidth;

        var textHidden = new float[textIds.Length * width];
        var textRun = new RunParameters();
        textRun.Symbols[WellKnown.TokenSymbol] = textIds.Length;
        _text.SetInput<int>(_text.Graph.Inputs.First(v => v.Input!.Name == AudioGenerationGraphs.TokensInput), textIds);
        _text.SetInput<int>(_text.Graph.Inputs.First(v => v.Input!.Name == AudioGenerationGraphs.PositionsInput), Enumerable.Range(0, textIds.Length).ToArray());
        var spans = new int[textIds.Length * 2];
        for (int i = 0; i < textIds.Length; i++)
        {
            spans[(i * 2) + 1] = textIds.Length;
        }

        _text.SetInput<int>(_text.Graph.Inputs.First(v => v.Input!.Name == AudioGenerationGraphs.SpansInput), spans);
        _text.Run(textRun);
        _text.GetOutput<float>(AudioGenerationGraphs.HiddenOutput, textHidden);

        var lyricHidden = new float[lyricIds.Length * width];
        var lyricRun = new RunParameters();
        lyricRun.Symbols[WellKnown.TokenSymbol] = lyricIds.Length;
        _lyric.SetInput<int>(_lyric.Graph.Inputs.First(v => v.Input!.Name == AudioGenerationGraphs.TokensInput), lyricIds);
        _lyric.Run(lyricRun);
        _lyric.GetOutput<float>(AudioGenerationGraphs.HiddenOutput, lyricHidden);

        rows = lyricIds.Length + 1 + textIds.Length;
        var packed = new float[rows * _conditionWidth];
        var run = new RunParameters();
        run.Symbols[AudioGenerationGraphs.LyricSymbol] = lyricIds.Length;
        run.Symbols[AudioGenerationGraphs.TextSymbol] = textIds.Length;
        _condition.SetInput<float>(_condition.Graph.Inputs.First(v => v.Input!.Name == AudioGenerationGraphs.TextInput), textHidden);
        _condition.SetInput<float>(_condition.Graph.Inputs.First(v => v.Input!.Name == AudioGenerationGraphs.LyricInput), lyricHidden);
        _condition.SetInput<float>(_condition.Graph.Inputs.First(v => v.Input!.Name == AudioGenerationGraphs.ReferenceInput), _silence.AsSpan(0, _referenceFrames * _latentChannels));
        _condition.SetInput<int>(_condition.Graph.Inputs.First(v => v.Input!.Name == AudioGenerationGraphs.LyricPositionsInput), Enumerable.Range(0, lyricIds.Length).ToArray());
        _condition.SetInput<int>(_condition.Graph.Inputs.First(v => v.Input!.Name == AudioGenerationGraphs.ReferencePositionsInput), Enumerable.Range(0, _referenceFrames).ToArray());
        _condition.Run(run);
        _condition.GetOutput<float>(AudioGenerationGraphs.ConditionOutput, packed);
        return packed;
    }

    private static int[] Truncate(int[] ids, int max) => ids.Length > max ? ids[..max] : ids;

    private AudioBuffer Decode(AudioLatents latents, CancellationToken cancellationToken)
    {
        var samples = AudioLatentDecoder.Decode(_decode, latents.Values, latents.Frames, latents.Frames, _latentChannels, _samplesPerFrame, Channels, _decodeFrames, _decodeOverlap,
            cancellationToken);
        AudioLatentDecoder.LimitPeak(samples);
        return new AudioBuffer(samples, SampleRate, Channels);
    }
}
