namespace Sharplight.ImageGeneration;

/// <summary>
/// DPM-Solver++(2M) (Lu et al. 2022), the multistep second-order solver in data prediction, as diffusers'
/// <c>DPMSolverMultistepScheduler</c> with <c>algorithm_type = "dpmsolver++"</c>, <c>solver_order = 2</c> and a final noise level of 0:
/// integer timesteps of the schedule's spacing (default <c>linspace</c> over steps + 1 points), optionally the Karras noise levels over
/// the whole training range; the first and the last step are first order.
/// </summary>
internal sealed class DpmSolverSampler : DiffusionSampler
{
    private readonly float[] _timesteps;
    private readonly float[] _sigmas;
    private float[]? _previousOriginal;

    /// <summary>Prepares a run.</summary>
    /// <param name="schedule">The model's noise schedule.</param>
    /// <param name="steps">Denoising steps.</param>
    /// <param name="karras">Whether to use the Karras noise levels.</param>
    public DpmSolverSampler(NoiseSchedule schedule, int steps, bool karras)
        : base(schedule)
    {
        var training = schedule.Sigmas();
        var timesteps = Spaced(steps);
        float[] sigmas;
        if (karras)
        {
            sigmas = KarrasSigmas.Schedule(training[0], training[^1], steps);
            timesteps = [.. sigmas.Select(s => MathF.Round(KarrasSigmas.Timestep(s, training), MidpointRounding.ToEven))];
        }
        else
        {
            sigmas = [.. timesteps.Select(t => KarrasSigmas.Interpolate(t, training))];
        }

        _timesteps = timesteps;
        _sigmas = [.. sigmas, 0f];
    }

    /// <inheritdoc/>
    public override IReadOnlyList<float> Timesteps => _timesteps;

    /// <inheritdoc/>
    public override float InitialNoiseScale => 1f;

    /// <inheritdoc/>
    public override float ModelInputScale(int index) => 1f;

    /// <inheritdoc/>
    public override void Step(int index, ReadOnlySpan<float> prediction, Span<float> sample, Func<float[]> noise)
    {
        var (alphaS, sigmaS) = AlphaSigma(_sigmas[index]);
        var (alphaT, sigmaT) = AlphaSigma(_sigmas[index + 1]);
        var original = new float[sample.Length];
        DataPrediction(prediction, sample, alphaS, sigmaS, original);
        float lambdaS = Lambda(alphaS, sigmaS);
        float lambdaT = Lambda(alphaT, sigmaT);
        float h = lambdaT - lambdaS;
        float decay = MathF.Exp(-h) - 1f;
        float ratio = sigmaT / sigmaS;
        bool last = index == _timesteps.Length - 1;
        if (_previousOriginal is null || last)
        {
            for (int i = 0; i < sample.Length; i++)
            {
                sample[i] = ratio * sample[i] - alphaT * decay * original[i];
            }
        }
        else
        {
            var (alphaPrevious, sigmaPrevious) = AlphaSigma(_sigmas[index - 1]);
            float h0 = lambdaS - Lambda(alphaPrevious, sigmaPrevious);
            float r0 = h0 / h;
            for (int i = 0; i < sample.Length; i++)
            {
                float d1 = (original[i] - _previousOriginal[i]) / r0;
                sample[i] = ratio * sample[i] - alphaT * decay * original[i] - 0.5f * alphaT * decay * d1;
            }
        }

        _previousOriginal = original;
        PredictedOriginal = original;
    }

    /// <inheritdoc/>
    public override void AddNoise(ReadOnlySpan<float> original, ReadOnlySpan<float> noise, int index, Span<float> sample)
    {
        var (alpha, sigma) = AlphaSigma(_sigmas[index]);
        for (int i = 0; i < sample.Length; i++)
        {
            sample[i] = alpha * original[i] + sigma * noise[i];
        }
    }

    private static (float Alpha, float Sigma) AlphaSigma(float sigma)
    {
        float alpha = 1f / MathF.Sqrt(sigma * sigma + 1);
        return (alpha, sigma * alpha);
    }

    private static float Lambda(float alpha, float sigma) => MathF.Log(alpha) - MathF.Log(sigma);

    private void DataPrediction(ReadOnlySpan<float> prediction, ReadOnlySpan<float> sample, float alpha, float sigma, Span<float> original)
    {
        switch (Schedule.PredictionType)
        {
            case "epsilon":
                for (int i = 0; i < sample.Length; i++)
                {
                    original[i] = (sample[i] - sigma * prediction[i]) / alpha;
                }

                break;
            case "v_prediction":
                for (int i = 0; i < sample.Length; i++)
                {
                    original[i] = alpha * sample[i] - sigma * prediction[i];
                }

                break;
            case "sample":
                prediction.CopyTo(original);
                break;
            default:
                throw new NotSupportedException($"Prediction type '{Schedule.PredictionType}' is not supported.");
        }
    }

    /// <summary>The solver's timesteps: <c>steps + 1</c> points of the spacing without the last (the reference's convention).</summary>
    private float[] Spaced(int steps)
    {
        int train = Schedule.TrainSteps;
        var timesteps = new float[steps];
        switch (Schedule.TimestepSpacing ?? "linspace")
        {
            case "linspace":
                for (int i = 0; i < steps; i++)
                {
                    double value = (train - 1) * (double)(steps - i) / steps;
                    timesteps[i] = (float)Math.Round(value, MidpointRounding.ToEven);
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
                int stepRatio = train / (steps + 1);
                for (int i = 0; i < steps; i++)
                {
                    timesteps[i] = (steps - i) * stepRatio + Schedule.StepsOffset;
                }

                break;
        }

        return timesteps;
    }
}
