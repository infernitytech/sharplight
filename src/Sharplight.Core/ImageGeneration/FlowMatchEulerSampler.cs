namespace Sharplight.ImageGeneration;

/// <summary>
/// Euler's method on the rectified-flow ODE, as diffusers' <c>FlowMatchEulerDiscreteScheduler</c> driven by the FLUX / SD3 pipelines:
/// noise levels σ linearly spaced from 1 to 1 / steps, shifted (statically by <see cref="NoiseSchedule.Shift"/>, or with
/// <see cref="NoiseSchedule.DynamicShift"/> by μ interpolated from the image's token count), followed by 0; the model timestep is
/// σ · train steps; the denoiser predicts the velocity v = ε − x₀ and each step moves the sample by (σₙₑₓₜ − σ) · v. Samples are
/// x = (1 − σ) · x₀ + σ · ε, so the initial sample is the unscaled noise and the model input is the sample itself.
/// </summary>
internal sealed class FlowMatchEulerSampler : DiffusionSampler
{
    /// <summary>
    /// <see cref="NoiseSchedule.TimestepSpacing"/> of flow schedules whose pipeline passes no noise levels, so the scheduler's own
    /// spacing applies (the Wan pipelines).
    /// </summary>
    public const string SchedulerSpacing = "scheduler";

    private readonly float[] _sigmas;
    private readonly float[] _timesteps;

    /// <summary>Prepares a run.</summary>
    /// <param name="schedule">The model's schedule (prediction type <c>flow</c>).</param>
    /// <param name="steps">Denoising steps.</param>
    /// <param name="imageTokens">The denoiser's image tokens (latent pixels / patch area), for the dynamic shift.</param>
    public FlowMatchEulerSampler(NoiseSchedule schedule, int steps, int imageTokens)
        : base(schedule)
    {
        float shiftExp = (float)Math.Exp(Mu(schedule, imageTokens));
        _sigmas = new float[steps + 1];
        _timesteps = new float[steps];
        // The FLUX / SD3 pipelines pass σ = linspace(1, 1 / steps); pipelines that pass none (Wan) get the scheduler's own levels:
        // linspace from 1 to its smallest training level (1 / train steps, already shifted once), shifted again.
        bool schedulerSpacing = schedule.TimestepSpacing == SchedulerSpacing;
        double train = schedule.TrainSteps;
        double last = schedulerSpacing ? schedule.Shift / train / (1 + (schedule.Shift - 1) / train) : 1.0 / steps;
        for (int i = 0; i < steps; i++)
        {
            // np.linspace in double, then f32 as the scheduler stores it.
            double linear = steps == 1 ? 1.0 : 1.0 + (last - 1.0) * i / (steps - 1);
            float sigma = (float)linear;
            sigma = schedule.DynamicShift
                ? shiftExp / (shiftExp + (1f / sigma - 1f))
                : schedule.Shift * sigma / (1f + (schedule.Shift - 1f) * sigma);
            _sigmas[i] = sigma;
        }

        if (schedule.ShiftTerminal > 0)
        {
            // diffusers' stretch_shift_to_terminal: 1 − σ scaled so that the last level becomes the terminal one.
            float scale = (1f - _sigmas[steps - 1]) / (1f - (float)schedule.ShiftTerminal);
            for (int i = 0; i < steps; i++)
            {
                _sigmas[i] = 1f - (1f - _sigmas[i]) / scale;
            }
        }

        for (int i = 0; i < steps; i++)
        {
            _timesteps[i] = _sigmas[i] * schedule.TrainSteps;
        }
    }

    /// <inheritdoc/>
    public override IReadOnlyList<float> Timesteps => _timesteps;

    /// <inheritdoc/>
    public override float InitialNoiseScale => 1f;

    /// <summary>The noise level before each step, plus the final 0.</summary>
    public IReadOnlyList<float> Sigmas => _sigmas;

    /// <inheritdoc/>
    public override float ModelInputScale(int index) => 1f;

    /// <inheritdoc/>
    public override void Step(int index, ReadOnlySpan<float> prediction, Span<float> sample, Func<float[]> noise)
    {
        float sigma = _sigmas[index];
        float dt = _sigmas[index + 1] - sigma;
        var original = new float[sample.Length];
        for (int i = 0; i < sample.Length; i++)
        {
            original[i] = sample[i] - sigma * prediction[i];
            sample[i] += dt * prediction[i];
        }

        PredictedOriginal = original;
    }

    /// <inheritdoc/>
    public override void AddNoise(ReadOnlySpan<float> original, ReadOnlySpan<float> noise, int index, Span<float> sample)
    {
        float sigma = _sigmas[index];
        for (int i = 0; i < sample.Length; i++)
        {
            sample[i] = sigma * noise[i] + (1f - sigma) * original[i];
        }
    }

    /// <inheritdoc/>
    /// <remarks>The flow pipelines keep the fractional part: the start is int(max(steps − steps · strength, 0)).</remarks>
    public override int StartIndex(int steps, float strength)
    {
        double init = Math.Min(steps * (double)strength, steps);
        return (int)Math.Max(steps - init, 0);
    }

    /// <summary>The dynamic shift μ: linear in the image token count between the base and maximum sequence lengths.</summary>
    private static double Mu(NoiseSchedule schedule, int imageTokens)
    {
        double slope = (schedule.MaxShift - schedule.BaseShift) / (schedule.MaxSequenceLength - schedule.BaseSequenceLength);
        double intercept = schedule.BaseShift - slope * schedule.BaseSequenceLength;
        return imageTokens * slope + intercept;
    }
}
