namespace Sharplight.ImageGeneration;

/// <summary>
/// One sampler (scheduler) run of a diffusion pipeline, following the reference implementation's state machine so results match it:
/// the pipeline starts from noise times <see cref="InitialNoiseScale"/>, and for every entry of <see cref="Timesteps"/> evaluates the
/// denoiser on the sample times <see cref="ModelInputScale"/> at that timestep and passes the prediction to <see cref="Step"/>, which
/// updates the sample in place. After the last step the sample is the clean latent. Samplers keep per-run state (multistep
/// history): create one per image.
/// </summary>
internal abstract class DiffusionSampler
{
    /// <summary>Creates a sampler over a schedule.</summary>
    /// <param name="schedule">The model's noise schedule.</param>
    protected DiffusionSampler(NoiseSchedule schedule)
    {
        Schedule = schedule;
        AlphasCumulative = schedule.AlphasCumulative();
    }

    /// <summary>The model's noise schedule.</summary>
    protected NoiseSchedule Schedule { get; }

    /// <summary>ᾱ per training timestep.</summary>
    protected float[] AlphasCumulative { get; }

    /// <summary>The model timestep of every denoiser evaluation, in order.</summary>
    public abstract IReadOnlyList<float> Timesteps { get; }

    /// <summary>Factor applied to unit Gaussian noise to make the initial sample of text-to-image generation.</summary>
    public abstract float InitialNoiseScale { get; }

    /// <summary>Factor applied to the sample to make the denoiser input of evaluation <paramref name="index"/>.</summary>
    /// <param name="index">Index into <see cref="Timesteps"/>.</param>
    public abstract float ModelInputScale(int index);

    /// <summary>Advances the sample past evaluation <paramref name="index"/>.</summary>
    /// <param name="index">Index into <see cref="Timesteps"/>.</param>
    /// <param name="prediction">The (guided) denoiser output.</param>
    /// <param name="sample">The sample, updated in place.</param>
    /// <param name="noise">Fresh unit Gaussian noise of the sample's size, drawn only by stochastic samplers.</param>
    public abstract void Step(int index, ReadOnlySpan<float> prediction, Span<float> sample, Func<float[]> noise);

    /// <summary>The clean-sample estimate of the last <see cref="Step"/> (for previews), or <see langword="null"/>.</summary>
    public float[]? PredictedOriginal { get; protected set; }

    /// <summary>
    /// Noises a clean sample to the level of evaluation <paramref name="index"/> (image-to-image: the sampler then continues from that
    /// evaluation).
    /// </summary>
    /// <param name="original">The clean latent.</param>
    /// <param name="noise">Unit Gaussian noise of the same size.</param>
    /// <param name="index">Index into <see cref="Timesteps"/>.</param>
    /// <param name="sample">Receives the noised sample.</param>
    public abstract void AddNoise(ReadOnlySpan<float> original, ReadOnlySpan<float> noise, int index, Span<float> sample);

    /// <summary>
    /// Index of the first evaluation of image-to-image generation at <paramref name="strength"/> (the fraction of the schedule that
    /// runs), as the reference pipelines compute it.
    /// </summary>
    /// <param name="steps">Requested denoising steps.</param>
    /// <param name="strength">Strength in [0, 1].</param>
    public virtual int StartIndex(int steps, float strength)
    {
        int init = Math.Min((int)(steps * strength), steps);
        return Math.Max(steps - init, 0);
    }

    /// <summary>The clean-sample estimate from a prediction at noise level σ, for samplers that work on σ-scaled samples.</summary>
    protected void Denoised(ReadOnlySpan<float> prediction, ReadOnlySpan<float> sample, float sigma, Span<float> original)
    {
        switch (Schedule.PredictionType)
        {
            case "epsilon":
                for (int i = 0; i < sample.Length; i++)
                {
                    original[i] = sample[i] - sigma * prediction[i];
                }

                break;
            case "v_prediction":
                float outScale = -sigma / MathF.Sqrt(sigma * sigma + 1);
                float skip = 1f / (sigma * sigma + 1);
                for (int i = 0; i < sample.Length; i++)
                {
                    original[i] = prediction[i] * outScale + sample[i] * skip;
                }

                break;
            case "sample":
                prediction.CopyTo(original);
                break;
            default:
                throw new NotSupportedException($"Prediction type '{Schedule.PredictionType}' is not supported.");
        }
    }

    /// <summary>The noise prediction from a prediction at a VP timestep with cumulative alpha <paramref name="alpha"/>.</summary>
    protected void Epsilon(ReadOnlySpan<float> prediction, ReadOnlySpan<float> sample, float alpha, Span<float> epsilon)
    {
        float signal = MathF.Sqrt(alpha);
        float noise = MathF.Sqrt(1 - alpha);
        switch (Schedule.PredictionType)
        {
            case "epsilon":
                prediction.CopyTo(epsilon);
                break;
            case "v_prediction":
                for (int i = 0; i < sample.Length; i++)
                {
                    epsilon[i] = signal * prediction[i] + noise * sample[i];
                }

                break;
            case "sample":
                for (int i = 0; i < sample.Length; i++)
                {
                    epsilon[i] = (sample[i] - signal * prediction[i]) / noise;
                }

                break;
            default:
                throw new NotSupportedException($"Prediction type '{Schedule.PredictionType}' is not supported.");
        }
    }

    /// <summary>Inference timesteps of the schedule's spacing as integers, descending (the DDIM / PNDM convention).</summary>
    /// <param name="steps">Denoising steps.</param>
    /// <param name="defaultSpacing">The sampler's spacing when the schedule names none.</param>
    protected int[] IntegerTimesteps(int steps, string defaultSpacing)
    {
        int train = Schedule.TrainSteps;
        var timesteps = new int[steps];
        switch (Schedule.TimestepSpacing ?? defaultSpacing)
        {
            case "linspace":
                for (int i = 0; i < steps; i++)
                {
                    double value = steps == 1 ? 0 : (train - 1) * (double)i / (steps - 1);
                    timesteps[steps - 1 - i] = (int)Math.Round(value, MidpointRounding.ToEven);
                }

                break;
            case "trailing":
                double ratio = train / (double)steps;
                for (int i = 0; i < steps; i++)
                {
                    timesteps[i] = (int)Math.Round(train - i * ratio, MidpointRounding.ToEven) - 1;
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
