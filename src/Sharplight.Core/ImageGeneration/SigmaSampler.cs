namespace Sharplight.ImageGeneration;

/// <summary>
/// Base of the samplers that work on σ-scaled samples x = x₀ + σ·ε (Euler, Euler ancestral): fractional timesteps of the schedule's
/// spacing (default <c>linspace</c>), their interpolated noise levels (optionally re-spaced by the Karras schedule) followed by 0,
/// model inputs scaled by 1 / sqrt(σ² + 1).
/// </summary>
internal abstract class SigmaSampler : DiffusionSampler
{
    private readonly float[] _timesteps;

    /// <summary>Prepares a run of <paramref name="steps"/> steps.</summary>
    /// <param name="schedule">The model's noise schedule.</param>
    /// <param name="steps">Denoising steps.</param>
    /// <param name="karras">Whether the noise levels follow the Karras schedule between the first and last level.</param>
    protected SigmaSampler(NoiseSchedule schedule, int steps, bool karras)
        : base(schedule)
    {
        var training = schedule.Sigmas();
        var timesteps = FractionalTimesteps(steps);
        var sigmas = timesteps.Select(t => KarrasSigmas.Interpolate(t, training)).ToArray();
        if (karras)
        {
            sigmas = KarrasSigmas.Schedule(sigmas[^1], sigmas[0], steps);
            timesteps = [.. sigmas.Select(s => KarrasSigmas.Timestep(s, training))];
        }

        _timesteps = timesteps;
        Sigmas = [.. sigmas, 0f];
        float max = Sigmas.Max();
        string spacing = schedule.TimestepSpacing ?? "linspace";
        InitialNoiseScale = spacing is "linspace" or "trailing" ? max : MathF.Sqrt(max * max + 1);
    }

    /// <summary>Noise level before each step, plus the final 0.</summary>
    protected float[] Sigmas { get; }

    /// <inheritdoc/>
    public override IReadOnlyList<float> Timesteps => _timesteps;

    /// <inheritdoc/>
    public override float InitialNoiseScale { get; }

    /// <inheritdoc/>
    public override float ModelInputScale(int index) => 1f / MathF.Sqrt(Sigmas[index] * Sigmas[index] + 1);

    /// <inheritdoc/>
    public override void AddNoise(ReadOnlySpan<float> original, ReadOnlySpan<float> noise, int index, Span<float> sample)
    {
        float sigma = Sigmas[index];
        for (int i = 0; i < sample.Length; i++)
        {
            sample[i] = original[i] + noise[i] * sigma;
        }
    }

    private float[] FractionalTimesteps(int steps)
    {
        int train = Schedule.TrainSteps;
        var timesteps = new float[steps];
        switch (Schedule.TimestepSpacing ?? "linspace")
        {
            case "linspace":
                for (int i = 0; i < steps; i++)
                {
                    float value = steps == 1 ? 0 : (float)((train - 1) * (double)i / (steps - 1));
                    timesteps[steps - 1 - i] = value;
                }

                break;
            case "trailing":
                double ratio = train / (double)steps;
                for (int i = 0; i < steps; i++)
                {
                    timesteps[i] = (float)Math.Round(train - i * ratio, MidpointRounding.ToEven) - 1;
                }

                break;
            default:
                int stepRatio = train / steps;
                for (int i = 0; i < steps; i++)
                {
                    timesteps[steps - 1 - i] = i * stepRatio + Schedule.StepsOffset;
                }

                break;
        }

        return timesteps;
    }
}
