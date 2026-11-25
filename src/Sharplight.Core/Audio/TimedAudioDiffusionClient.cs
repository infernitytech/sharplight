using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Tokenizers;

namespace Sharplight.Audio;

/// <summary>
/// The built-in <see cref="IAudioGenerator"/> over the <see cref="TimedAudioDiffusionGraphs"/> contract (Stable Audio Open): the prompt's encoder rows and the timing rows
/// (start 0, total the requested duration) condition a v-objective network over the model's fixed latent window, sampled with DPM-Solver++(2M) SDE
/// (<see cref="DpmSolverSdeSampler"/>) and classifier-free guidance against a negative prompt or the empty conditioning; the autoencoder decodes the frames that cover
/// the duration and the audio is cut to it.
/// </summary>
internal sealed class TimedAudioDiffusionClient : IAudioGenerator
{
    private readonly TaskModel _model;
    private readonly ITokenizer _tokenizer;
    private readonly CompiledModel _text;
    private readonly CompiledModel _timing;
    private readonly CompiledModel _denoise;
    private readonly CompiledModel _decode;
    private readonly int _promptLength;
    private readonly int _width;
    private readonly int _frames;
    private readonly int _samplesPerFrame;
    private readonly int _decodeFrames;
    private readonly int _decodeOverlap;
    private readonly double _maxSeconds;
    private readonly double _sigmaMin;
    private readonly double _sigmaMax;
    private readonly double _sigmaData;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _disposed;

    public TimedAudioDiffusionClient(TaskModel model)
    {
        _model = model;
        var text = model.Description.Component(ModelComponentRoles.TextEncoder);
        var timing = model.Description.Component(ModelComponentRoles.Encoder);
        var denoiser = model.Description.Component(ModelComponentRoles.Denoiser);
        var vae = model.Description.Component(ModelComponentRoles.Vae);
        _promptLength = text.ConfigValue(TimedAudioDiffusionGraphs.PromptLengthKey, 128);
        _width = text.ConfigValue(TimedAudioDiffusionGraphs.WidthKey, 0);
        _maxSeconds = timing.ConfigValue(TimedAudioDiffusionGraphs.MaxSecondsKey, 512.0);
        SampleRate = denoiser.ConfigValue(AudioGenerationGraphs.SampleRateKey, 0);
        Channels = denoiser.ConfigValue(AudioGenerationGraphs.ChannelsKey, 2);
        _samplesPerFrame = denoiser.ConfigValue(AudioGenerationGraphs.SamplesPerFrameKey, 1);
        LatentChannels = denoiser.ConfigValue(AudioGenerationGraphs.LatentChannelsKey, 0);
        _frames = denoiser.ConfigValue(TimedAudioDiffusionGraphs.FramesKey, 0);
        Defaults = new AudioGenerationDefaults(denoiser.ConfigValue(AudioGenerationGraphs.DefaultStepsKey, 100),
            (float)denoiser.ConfigValue(TimedAudioDiffusionGraphs.DefaultGuidanceKey, 7.0), SupportsNegativePrompt: true);
        _sigmaMin = denoiser.ConfigValue(TimedAudioDiffusionGraphs.SigmaMinKey, 0.3);
        _sigmaMax = denoiser.ConfigValue(TimedAudioDiffusionGraphs.SigmaMaxKey, 500.0);
        _sigmaData = denoiser.ConfigValue(TimedAudioDiffusionGraphs.SigmaDataKey, 1.0);
        _decodeFrames = vae.ConfigValue(AudioGenerationGraphs.DecodeFramesKey, 256);
        _decodeOverlap = vae.ConfigValue(AudioGenerationGraphs.DecodeOverlapKey, 16);
        LatentFrameRate = (double)SampleRate / _samplesPerFrame;
        _tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
        _text = model.CompileComponent(ModelComponentRoles.TextEncoder, TimedAudioDiffusionGraphs.TextGraph);
        _timing = model.CompileComponent(ModelComponentRoles.Encoder, TimedAudioDiffusionGraphs.TimingGraph);
        _denoise = model.CompileComponent(ModelComponentRoles.Denoiser, TimedAudioDiffusionGraphs.DenoiseGraph);
        _decode = model.CompileComponent(ModelComponentRoles.Vae, AudioGenerationGraphs.DecodeGraph);
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public int SampleRate { get; }

    public int Channels { get; }

    public double LatentFrameRate { get; }

    public int LatentChannels { get; }

    public AudioGenerationDefaults Defaults { get; }

    public TimeSpan MaxDuration => TimeSpan.FromSeconds(_frames / LatentFrameRate);

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    /// <inheritdoc/>
    /// <remarks>The latents cover the model's whole window (<see cref="MaxDuration"/>); the requested duration conditions them and cuts the decoded audio.</remarks>
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
            emit(new AudioGenerationUpdate(done, total, new AudioGenerationResult(Decode(latents, request.Duration, token), seed)));
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
        _timing.Dispose();
        _denoise.Dispose();
        _decode.Dispose();
        _gate.Dispose();
    }

    private AudioLatents Denoise(AudioGenerationRequest request, AudioGenerationOptions options, long seed, Action<int, int>? progress, CancellationToken cancellationToken)
    {
        double seconds = request.Duration.TotalSeconds;
        if (seconds <= 0 || request.Duration > MaxDuration || seconds > _maxSeconds)
        {
            throw new ArgumentOutOfRangeException(nameof(request), $"The duration must be between 0 and {MaxDuration.TotalSeconds:0.##} s.");
        }

        int steps = options.Steps ?? Defaults.Steps;
        if (steps < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "The number of steps must be positive.");
        }

        int count = _frames * LatentChannels;
        if (options.StepNoise is { } given && (given.Count < steps || given.Any(n => n.Length != count)))
        {
            throw new ArgumentException($"The step noise needs {steps} arrays of {count} values.", nameof(options));
        }

        float guidance = options.GuidanceScale ?? Defaults.GuidanceScale ?? 1f;
        bool guided = guidance > 1f;
        float[] timing = Timing(seconds);
        float[] condition = Condition(Encode(request.Prompt), timing);
        float[]? uncondition = null;
        if (guided)
        {
            // Without a negative prompt the unconditional pass sees all-zero rows (the timing included), as the reference pipeline does.
            uncondition = request.NegativePrompt is { } negative ? Condition(Encode(negative), timing) : new float[condition.Length];
        }

        var random = AudioNoise.Create(seed);
        var latent = new float[count];
        if (options.InitialNoise is { } initial)
        {
            if (initial.Length != count)
            {
                throw new ArgumentException($"The initial noise has {initial.Length} values, {count} are needed.", nameof(options));
            }

            initial.Span.CopyTo(latent);
        }
        else
        {
            AudioNoise.Fill(random, latent);
        }

        var sampler = new DpmSolverSdeSampler(steps, _sigmaMin, _sigmaMax, _sigmaData);
        float initialScale = (float)sampler.InitialSigma;
        for (int i = 0; i < count; i++)
        {
            latent[i] *= initialScale;
        }

        var inputs = new
        {
            Latent = _denoise.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.LatentInput),
            Time = _denoise.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.TimestepInput),
            Condition = _denoise.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.ConditionInput),
            Global = _denoise.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.GlobalInput),
            Positions = _denoise.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.PositionsInput),
        };
        var positions = Enumerable.Range(0, _frames + 1).ToArray();
        var scaled = new float[count];
        var prediction = new float[count];
        var unconditional = guided ? new float[count] : null;
        var noise = new float[count];
        var parameters = new RunParameters();
        for (int step = 0; step < steps; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            float scale = sampler.InputScale(step);
            for (int i = 0; i < count; i++)
            {
                scaled[i] = latent[i] * scale;
            }

            float time = sampler.NetworkTime(step);
            Predict(condition, prediction);
            if (unconditional is not null)
            {
                Predict(uncondition!, unconditional);
                for (int i = 0; i < count; i++)
                {
                    prediction[i] = unconditional[i] + (guidance * (prediction[i] - unconditional[i]));
                }
            }

            if (options.StepNoise is { } stepNoise)
            {
                stepNoise[step].Span.CopyTo(noise);
            }
            else
            {
                AudioNoise.Fill(random, noise);
            }

            sampler.Step(step, latent, prediction, noise);
            progress?.Invoke(step + 1, steps);

            void Predict(float[] rows, float[] output)
            {
                _denoise.SetInput<float>(inputs.Latent, scaled);
                _denoise.SetInput<float>(inputs.Time, [time]);
                _denoise.SetInput<float>(inputs.Condition, rows);
                _denoise.SetInput<float>(inputs.Global, timing);
                _denoise.SetInput<int>(inputs.Positions, positions);
                _denoise.Run(parameters);
                _denoise.GetOutput<float>(TimedAudioDiffusionGraphs.PredictionOutput, output);
            }
        }

        return new AudioLatents(_frames, LatentChannels, latent);
    }

    // The prompt's encoder rows: the fixed prompt length, the padding rows zero.
    private float[] Encode(string prompt)
    {
        int[] ids = PromptTokenizer.Tokenize(_tokenizer, _promptLength, prompt, out int tokens);
        var mask = new float[_promptLength];
        mask.AsSpan(0, tokens).Fill(1f);
        _text.SetInput<int>(_text.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.TokensInput), ids);
        _text.SetInput<float>(_text.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.MaskInput), mask);
        _text.Run(new RunParameters());
        var hidden = new float[_promptLength * _width];
        _text.GetOutput<float>(TimedAudioDiffusionGraphs.HiddenOutput, hidden);
        hidden.AsSpan(tokens * _width).Clear();
        return hidden;
    }

    // The timing rows of start 0 and total length seconds, [2, width] (side by side they are the global vector).
    private float[] Timing(double seconds)
    {
        _timing.SetInput<float>(_timing.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.SecondsInput), [0f, (float)seconds]);
        _timing.Run(new RunParameters());
        var rows = new float[2 * _width];
        _timing.GetOutput<float>(TimedAudioDiffusionGraphs.TimingOutput, rows);
        return rows;
    }

    private static float[] Condition(float[] text, float[] timing) => [.. text, .. timing];

    private AudioBuffer Decode(AudioLatents latents, TimeSpan duration, CancellationToken cancellationToken)
    {
        int length = (int)(duration.TotalSeconds * SampleRate);
        int frames = Math.Min(latents.Frames, (length + _samplesPerFrame - 1) / _samplesPerFrame);
        var decoded = AudioLatentDecoder.Decode(_decode, latents.Values, latents.Frames, frames, LatentChannels, _samplesPerFrame, Channels, _decodeFrames, _decodeOverlap,
            cancellationToken);
        var samples = decoded[..(length * Channels)];
        AudioLatentDecoder.LimitPeak(samples);
        return new AudioBuffer(samples, SampleRate, Channels);
    }
}
