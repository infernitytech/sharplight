using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Tokenizers;

namespace Sharplight.Audio;

/// <summary>
/// The built-in <see cref="IAudioGenerator"/> over the rectified-flow form of <see cref="TimedAudioDiffusionGraphs"/> (Stable Audio 3): the latent length follows the
/// requested duration plus some headroom; the prompt's rows and the seconds row condition a velocity network whose padding frames are masked; the base models
/// (<c>rectified_flow</c>) take Euler steps with adaptive projected guidance (the guidance direction made orthogonal to the conditional prediction), the post-trained
/// ones (<c>rf_denoiser</c>) ping-pong steps (denoise, then renoise to the next time); the SAME autoencoder decodes the frames in overlapping windows as the reference
/// package does, with its own random dither unless <see cref="AudioGenerationRequest.DecoderNoise"/> is 0, and the audio is cut to the duration and clamped to [−1, 1].
/// </summary>
internal sealed class FlowAudioDiffusionClient : IAudioGenerator
{
    private const string PingPong = "rf_denoiser";

    private readonly TaskModel _model;
    private readonly ITokenizer _tokenizer;
    private readonly CompiledModel _text;
    private readonly CompiledModel _seconds;
    private readonly CompiledModel _denoise;
    private readonly CompiledModel _decode;
    private readonly FlowTimeSchedule _schedule;
    private readonly string _objective;
    private readonly int _promptLength;
    private readonly int _width;
    private readonly int _maxFrames;
    private readonly int _samplesPerFrame;
    private readonly int _memoryTokens;
    private readonly int _frameAlign;
    private readonly double _headroomSeconds;
    private readonly bool _timeLogSnr;
    private readonly int _localChannels;
    private readonly int _decodeFrames;
    private readonly int _decodeOverlap;
    private readonly int _noiseRows;
    private readonly int _noiseWidth;
    private readonly float _latentNoise;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _disposed;

    public FlowAudioDiffusionClient(TaskModel model)
    {
        _model = model;
        var text = model.Description.Component(ModelComponentRoles.TextEncoder);
        var seconds = model.Description.Component(ModelComponentRoles.Encoder);
        var denoiser = model.Description.Component(ModelComponentRoles.Denoiser);
        var vae = model.Description.Component(ModelComponentRoles.Vae);
        _promptLength = text.ConfigValue(TimedAudioDiffusionGraphs.PromptLengthKey, 256);
        _width = text.ConfigValue(TimedAudioDiffusionGraphs.WidthKey, 0);
        double maxSeconds = seconds.ConfigValue(TimedAudioDiffusionGraphs.MaxSecondsKey, 384.0);
        SampleRate = denoiser.ConfigValue(AudioGenerationGraphs.SampleRateKey, 0);
        Channels = denoiser.ConfigValue(AudioGenerationGraphs.ChannelsKey, 2);
        _samplesPerFrame = denoiser.ConfigValue(AudioGenerationGraphs.SamplesPerFrameKey, 1);
        LatentChannels = denoiser.ConfigValue(AudioGenerationGraphs.LatentChannelsKey, 0);
        _maxFrames = denoiser.ConfigValue(TimedAudioDiffusionGraphs.FramesKey, 0);
        DefaultSteps = denoiser.ConfigValue(AudioGenerationGraphs.DefaultStepsKey, 50);
        _objective = denoiser.ConfigValue(TimedAudioDiffusionGraphs.ObjectiveKey, "rectified_flow");
        float guidance = (float)denoiser.ConfigValue(TimedAudioDiffusionGraphs.DefaultGuidanceKey, 1.0);
        DefaultGuidanceScale = _objective == PingPong && guidance == 1f ? null : guidance;
        _memoryTokens = denoiser.ConfigValue(TimedAudioDiffusionGraphs.MemoryTokensKey, 0);
        _frameAlign = denoiser.ConfigValue(TimedAudioDiffusionGraphs.FrameAlignKey, 1);
        _headroomSeconds = denoiser.ConfigValue(TimedAudioDiffusionGraphs.HeadroomSecondsKey, 6.0);
        _timeLogSnr = denoiser.ConfigValue(TimedAudioDiffusionGraphs.TimeLogSnrKey, false);
        _localChannels = denoiser.ConfigValue(TimedAudioDiffusionGraphs.LocalChannelsKey, 0);
        _decodeFrames = vae.ConfigValue(AudioGenerationGraphs.DecodeFramesKey, 128);
        _decodeOverlap = vae.ConfigValue(AudioGenerationGraphs.DecodeOverlapKey, 32);
        _noiseRows = vae.ConfigValue(TimedAudioDiffusionGraphs.NoiseRowsKey, 0);
        _noiseWidth = vae.ConfigValue(TimedAudioDiffusionGraphs.NoiseWidthKey, 0);
        _latentNoise = (float)vae.ConfigValue(TimedAudioDiffusionGraphs.LatentNoiseKey, 0.0);
        _schedule = new FlowTimeSchedule(denoiser);
        LatentFrameRate = (double)SampleRate / _samplesPerFrame;
        MaxDuration = TimeSpan.FromSeconds(Math.Min(maxSeconds, (double)_maxFrames * _samplesPerFrame / SampleRate));
        _tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
        _text = model.CompileComponent(ModelComponentRoles.TextEncoder, TimedAudioDiffusionGraphs.TextGraph);
        _seconds = model.CompileComponent(ModelComponentRoles.Encoder, TimedAudioDiffusionGraphs.TimingGraph);
        _denoise = model.CompileComponent(ModelComponentRoles.Denoiser, TimedAudioDiffusionGraphs.DenoiseGraph);
        _decode = model.CompileComponent(ModelComponentRoles.Vae, AudioGenerationGraphs.DecodeGraph);
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public int SampleRate { get; }

    public int Channels { get; }

    public double LatentFrameRate { get; }

    public int LatentChannels { get; }

    public int DefaultSteps { get; }

    /// <summary>The guided models (the base models) take a negative prompt; the post-trained ones sample without guidance.</summary>
    public bool SupportsNegativePrompt => DefaultGuidanceScale is not null;

    public float? DefaultGuidanceScale { get; }

    public TimeSpan MaxDuration { get; }

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    /// <inheritdoc/>
    /// <remarks>The latents cover the duration plus the model's headroom (<see cref="Frames"/>); the decoded audio is cut to the duration.</remarks>
    public Task<AudioLatents> GenerateLatentsAsync(AudioGenerationRequest request, IProgress<(int Step, int Steps)>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ObjectDisposedException.ThrowIf(_disposed, this);
        return BackgroundCompute.RunAsync(_gate, () => Denoise(request, progress, cancellationToken), cancellationToken);
    }

    /// <summary>Denoises with the chosen sampler instead of the model's own (ping-pong steps on a base model, Euler steps on a post-trained one).</summary>
    /// <param name="request">The request.</param>
    /// <param name="pingPong">Whether to take ping-pong steps.</param>
    /// <param name="cancellationToken">Stops generation between steps.</param>
    public Task<AudioLatents> GenerateLatentsAsync(AudioGenerationRequest request, bool pingPong, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ObjectDisposedException.ThrowIf(_disposed, this);
        return BackgroundCompute.RunAsync(_gate, () => Denoise(request, null, cancellationToken, pingPong), cancellationToken);
    }

    public Task<AudioBuffer> GenerateAsync(AudioGenerationRequest request, IProgress<(int Step, int Steps)>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ObjectDisposedException.ThrowIf(_disposed, this);
        return BackgroundCompute.RunAsync(_gate, () => Decode(Denoise(request, progress, cancellationToken), request, cancellationToken), cancellationToken);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _text.Dispose();
        _seconds.Dispose();
        _denoise.Dispose();
        _decode.Dispose();
        _gate.Dispose();
    }

    /// <summary>The latent frames of a duration: the duration plus the headroom, rounded up to whole frames and to the frame alignment, at most the model's frames.</summary>
    /// <param name="seconds">The duration in seconds.</param>
    public int Frames(double seconds)
    {
        long samples = (long)((seconds + _headroomSeconds) * SampleRate);
        long align = (long)_samplesPerFrame * _frameAlign;
        samples = (samples + _samplesPerFrame - 1) / _samplesPerFrame * _samplesPerFrame;
        samples = (samples + align - 1) / align * align;
        return (int)Math.Min(samples / _samplesPerFrame, _maxFrames);
    }

    private AudioLatents Denoise(AudioGenerationRequest request, IProgress<(int Step, int Steps)>? progress, CancellationToken cancellationToken, bool? sampler = null)
    {
        double seconds = request.Duration.TotalSeconds;
        if (seconds <= 0 || request.Duration > MaxDuration)
        {
            throw new ArgumentOutOfRangeException(nameof(request), $"The duration must be between 0 and {MaxDuration.TotalSeconds:0.##} s.");
        }

        int steps = request.Steps ?? DefaultSteps;
        if (steps < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "The number of steps must be positive.");
        }

        int frames = Frames(seconds);
        int count = frames * LatentChannels;
        bool pingPong = sampler ?? _objective == PingPong;
        if (pingPong && request.StepNoise is { } given && (given.Count < steps || given.Any(n => n.Length != count)))
        {
            throw new ArgumentException($"The step noise needs {steps} arrays of {count} values.", nameof(request));
        }

        // The frames of the duration, the valid frames (those plus the headroom) and the time schedule.
        int effective = (int)Math.Ceiling((double)(long)(seconds * SampleRate) / _samplesPerFrame);
        int validFrames = Math.Min(effective + (int)(_headroomSeconds * SampleRate / _samplesPerFrame), frames);
        float[] times = _schedule.Times(steps, effective);

        float guidance = request.GuidanceScale ?? DefaultGuidanceScale ?? 1f;
        bool guided = guidance != 1f;
        float[] secondsRow = SecondsRow(seconds);
        float[] condition = [.. Encode(request.Prompt, zeroPadding: false), .. secondsRow];
        float[]? uncondition = null;
        if (guided)
        {
            // Without a negative prompt the unconditional pass sees all-zero rows; a negative prompt's padding rows are zero (the reference's negative mask).
            uncondition = request.NegativePrompt is { } negative ? [.. Encode(negative, zeroPadding: true), .. secondsRow] : new float[condition.Length];
        }

        var random = AudioNoise.Create(request.Seed);
        var x = new float[count];
        if (request.InitialNoise is { } initial)
        {
            if (initial.Length != count)
            {
                throw new ArgumentException($"The initial noise has {initial.Length} values, {count} are needed ({frames} frames of {LatentChannels}).", nameof(request));
            }

            initial.Span.CopyTo(x);
        }
        else
        {
            AudioNoise.Fill(random, x);
        }

        var inputs = new
        {
            Latent = Input(_denoise, TimedAudioDiffusionGraphs.LatentInput),
            Time = Input(_denoise, TimedAudioDiffusionGraphs.TimestepInput),
            Condition = Input(_denoise, TimedAudioDiffusionGraphs.ConditionInput),
            Global = Input(_denoise, TimedAudioDiffusionGraphs.GlobalInput),
            Local = Input(_denoise, TimedAudioDiffusionGraphs.LocalInput),
            Positions = Input(_denoise, TimedAudioDiffusionGraphs.PositionsInput),
            Valid = Input(_denoise, TimedAudioDiffusionGraphs.ValidInput),
        };
        int rows = frames + _memoryTokens;
        var positions = new int[rows];
        var valid = new float[rows];
        for (int r = 0; r < rows; r++)
        {
            positions[r] = r < frames ? _memoryTokens + r : r - frames;
            valid[r] = r < validFrames || r >= frames ? 1f : 0f;
        }

        var local = new float[frames * _localChannels];
        var output = new float[rows * LatentChannels];
        var velocity = new float[count];
        var unconditional = guided ? new float[count] : null;
        var noise = pingPong ? new float[count] : null;
        var parameters = new RunParameters();
        parameters.Symbols[WellKnown.TokenSymbol] = frames;
        for (int step = 0; step < steps; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            float t = times[step];
            float next = times[step + 1];
            Predict(condition, velocity);
            if (unconditional is not null)
            {
                Predict(uncondition!, unconditional);
                ProjectedGuidance.Apply(x, velocity, unconditional, t, guidance, valid.AsSpan(0, frames), LatentChannels);
            }

            if (noise is not null)
            {
                // Ping-pong: denoise, then noise again to the next time.
                if (request.StepNoise is { } stepNoise)
                {
                    stepNoise[step].Span.CopyTo(noise);
                }
                else
                {
                    AudioNoise.Fill(random, noise);
                }

                for (int i = 0; i < count; i++)
                {
                    float denoised = x[i] - (t * velocity[i]);
                    x[i] = ((1f - next) * denoised) + (next * noise[i]);
                }
            }
            else
            {
                float dt = next - t;
                for (int i = 0; i < count; i++)
                {
                    x[i] += dt * velocity[i];
                }
            }

            progress?.Report((step + 1, steps));

            void Predict(float[] rowsIn, float[] result)
            {
                _denoise.SetInput<float>(inputs.Latent, x);
                _denoise.SetInput<float>(inputs.Time, [NetworkTime(t)]);
                _denoise.SetInput<float>(inputs.Condition, rowsIn);
                _denoise.SetInput<float>(inputs.Global, secondsRow);
                _denoise.SetInput<float>(inputs.Local, local);
                _denoise.SetInput<int>(inputs.Positions, positions);
                _denoise.SetInput<float>(inputs.Valid, valid);
                _denoise.Run(parameters);
                _denoise.GetOutput<float>(TimedAudioDiffusionGraphs.PredictionOutput, output);
                output.AsSpan(0, count).CopyTo(result);
            }
        }

        return new AudioLatents(frames, LatentChannels, x);
    }

    // The network's time: t, or its normalized log-SNR (t = 0 → 0, t = 1 → 1).
    private float NetworkTime(float t)
    {
        if (!_timeLogSnr)
        {
            return t;
        }

        float clamped = Math.Clamp(t, 1e-7f, 1f - 1e-7f);
        float logsnr = Math.Clamp(MathF.Log((1f - clamped) / clamped), -12f, 5f);
        return (5f - logsnr) / 17f;
    }

    // The prompt's rows at the fixed prompt length: the padding rows are the learned padding row, or zero (a negative prompt).
    private float[] Encode(string prompt, bool zeroPadding)
    {
        int[] ids = PromptTokenizer.Tokenize(_tokenizer, _promptLength, prompt, out int tokens, "{prompt}");
        var mask = new float[_promptLength];
        mask.AsSpan(0, tokens).Fill(1f);
        _text.SetInput<int>(Input(_text, TimedAudioDiffusionGraphs.TokensInput), ids);
        _text.SetInput<float>(Input(_text, TimedAudioDiffusionGraphs.MaskInput), mask);
        _text.Run(new RunParameters());
        var hidden = new float[_promptLength * _width];
        _text.GetOutput<float>(TimedAudioDiffusionGraphs.HiddenOutput, hidden);
        if (zeroPadding)
        {
            hidden.AsSpan(tokens * _width).Clear();
        }

        return hidden;
    }

    private float[] SecondsRow(double seconds)
    {
        _seconds.SetInput<float>(Input(_seconds, TimedAudioDiffusionGraphs.SecondsInput), [(float)seconds]);
        _seconds.Run(new RunParameters());
        var row = new float[_width];
        _seconds.GetOutput<float>(TimedAudioDiffusionGraphs.TimingOutput, row);
        return row;
    }

    /// <summary>Decodes latents to the request's duration (with the request's decoder noise and seed), as <see cref="GenerateAsync"/> does after denoising.</summary>
    /// <param name="latents">The latents of <see cref="GenerateLatentsAsync"/>.</param>
    /// <param name="request">The request (duration, seed, decoder noise).</param>
    /// <param name="cancellationToken">Stops between decoder windows.</param>
    public AudioBuffer Decode(AudioLatents latents, AudioGenerationRequest request, CancellationToken cancellationToken)
    {
        int length = (int)(request.Duration.TotalSeconds * SampleRate);
        int needed = Math.Min(latents.Frames, (length + _samplesPerFrame - 1) / _samplesPerFrame);
        float dither = request.DecoderNoise ?? 1f;
        var random = AudioNoise.Create(request.Seed is { } seed ? unchecked((seed * 31) + 17) : null);
        var values = latents.Values.ToArray();
        if (dither != 0f && _latentNoise > 0f)
        {
            var extra = new float[values.Length];
            AudioNoise.Fill(random, extra);
            for (int i = 0; i < values.Length; i++)
            {
                values[i] += dither * _latentNoise * extra[i];
            }
        }

        var samples = new float[latents.Frames * _samplesPerFrame * Channels];
        var latentInput = Input(_decode, AudioGenerationGraphs.LatentInput);
        var noiseInput = Input(_decode, TimedAudioDiffusionGraphs.NoiseInput);
        int window = _decodeFrames;
        var chunk = new float[window * LatentChannels];
        var noise = new float[window * _noiseRows * _noiseWidth];
        var decoded = new float[window * _samplesPerFrame * Channels];
        foreach (var (start, keepFrom, keepTo) in DecodeWindows(latents.Frames, needed))
        {
            cancellationToken.ThrowIfCancellationRequested();
            Array.Clear(chunk);
            int available = Math.Min(window, latents.Frames - start);
            values.AsSpan(start * LatentChannels, available * LatentChannels).CopyTo(chunk);
            if (dither != 0f)
            {
                AudioNoise.Fill(random, noise);
                if (dither != 1f)
                {
                    for (int i = 0; i < noise.Length; i++)
                    {
                        noise[i] *= dither;
                    }
                }
            }

            _decode.SetInput<float>(latentInput, chunk);
            _decode.SetInput<float>(noiseInput, noise);
            _decode.Run(new RunParameters());
            _decode.GetOutput<float>(AudioGenerationGraphs.AudioOutput, decoded);
            int perFrame = _samplesPerFrame * Channels;
            decoded.AsSpan(keepFrom * perFrame, (keepTo - keepFrom) * perFrame).CopyTo(samples.AsSpan((start + keepFrom) * perFrame));
        }

        var audio = samples[..(length * Channels)];
        for (int i = 0; i < audio.Length; i++)
        {
            audio[i] = Math.Clamp(audio[i], -1f, 1f);
        }

        return new AudioBuffer(audio, SampleRate, Channels);
    }

    // The decoder windows of the reference's chunked decode (chunks of the window size, the last one anchored at the end, half the overlap cut off the inner edges),
    // as (start frame, first kept frame, end of the kept frames) relative to the start; only the windows that reach the needed frames. Shorter latents are one window
    // padded with zero frames (the padding only changes the last frame, which lies in the headroom).
    private IEnumerable<(int Start, int KeepFrom, int KeepTo)> DecodeWindows(int frames, int needed)
    {
        int window = _decodeFrames;
        if (frames <= window)
        {
            yield return (0, 0, frames);
            yield break;
        }

        var starts = new List<int>();
        for (int s = 0; s <= frames - window; s += window - _decodeOverlap)
        {
            starts.Add(s);
        }

        if (starts[^1] != frames - window)
        {
            starts.Add(frames - window);
        }

        int half = _decodeOverlap / 2;
        int written = 0;
        for (int i = 0; i < starts.Count && written < needed; i++)
        {
            bool first = i == 0;
            bool last = i == starts.Count - 1;
            int start = last ? frames - window : starts[i];
            int keepFrom = first ? 0 : half;
            int keepTo = last ? window : window - half;
            yield return (start, keepFrom, keepTo);
            written = start + keepTo;
        }
    }

    private static Graph.Value Input(CompiledModel model, string name) => model.Graph.Inputs.First(v => v.Input!.Name == name);
}
