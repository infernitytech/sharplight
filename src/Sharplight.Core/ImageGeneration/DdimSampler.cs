namespace Sharplight.ImageGeneration;

/// <summary>
/// Deterministic DDIM sampling (η = 0, Song et al. 2021) on variance-preserving samples, as diffusers' <c>DDIMScheduler</c>:
/// integer timesteps of the schedule's spacing (default <c>leading</c>); the last step goes to the first training alpha (or to 1 with
/// <see cref="NoiseSchedule.SetAlphaToOne"/>).
/// </summary>
internal sealed class DdimSampler : DiffusionSampler
{
    private readonly int[] _timesteps;
    private readonly float[] _timestepValues;
    private readonly int _stride;

    /// <summary>Prepares a run.</summary>
    /// <param name="schedule">The model's noise schedule.</param>
    /// <param name="steps">Denoising steps.</param>
    public DdimSampler(NoiseSchedule schedule, int steps)
        : base(schedule)
    {
        _timesteps = IntegerTimesteps(steps, "leading");
        _timestepValues = [.. _timesteps.Select(t => (float)t)];
        _stride = schedule.TrainSteps / steps;
        FinalAlpha = (schedule.SetAlphaToOne ?? true) ? 1f : AlphasCumulative[0];
    }

    private float FinalAlpha { get; }

    /// <inheritdoc/>
    public override IReadOnlyList<float> Timesteps => _timestepValues;

    /// <inheritdoc/>
    public override float InitialNoiseScale => 1f;

    /// <inheritdoc/>
    public override float ModelInputScale(int index) => 1f;

    /// <inheritdoc/>
    public override void Step(int index, ReadOnlySpan<float> prediction, Span<float> sample, Func<float[]> noise)
    {
        int t = _timesteps[index];
        int previous = t - _stride;
        float alpha = AlphasCumulative[t];
        float alphaPrevious = previous >= 0 ? AlphasCumulative[previous] : FinalAlpha;
        var epsilon = new float[sample.Length];
        Epsilon(prediction, sample, alpha, epsilon);
        var original = new float[sample.Length];
        float signal = MathF.Sqrt(alpha);
        float noiseLevel = MathF.Sqrt(1 - alpha);
        float signalPrevious = MathF.Sqrt(alphaPrevious);
        float direction = MathF.Sqrt(1 - alphaPrevious);
        bool clip = Schedule.ClipSample ?? true;
        for (int i = 0; i < sample.Length; i++)
        {
            original[i] = (sample[i] - noiseLevel * epsilon[i]) / signal;
            if (clip)
            {
                // Only the clean estimate is clipped; the step keeps the model's noise prediction (as the reference pipelines do).
                original[i] = Math.Clamp(original[i], -1f, 1f);
            }

            sample[i] = signalPrevious * original[i] + direction * epsilon[i];
        }

        PredictedOriginal = original;
    }

    /// <inheritdoc/>
    public override void AddNoise(ReadOnlySpan<float> original, ReadOnlySpan<float> noise, int index, Span<float> sample)
        => VariancePreserving.AddNoise(AlphasCumulative[_timesteps[index]], original, noise, sample);
}
