namespace Sharplight.Audio;

/// <summary>
/// k-diffusion's DPM-Solver++(2M) SDE sampler for a v-objective network with the preconditioning of Stable Audio (diffusers' <c>CosineDPMSolverMultistepScheduler</c>
/// with the midpoint solver): an exponential sigma schedule from <c>σ_max</c> down to <c>σ_min</c> followed by 0, inputs scaled by <c>c_in = 1 / √(σ² + σ_d²)</c>, the
/// network's time <c>atan(σ) · 2 / π</c>, denoised estimates <c>c_skip · x + c_out · v</c>, a first-order first and last step and second-order (midpoint) steps between.
/// </summary>
internal sealed class DpmSolverSdeSampler
{
    private readonly double[] _sigmas;
    private readonly double _sigmaData;
    private float[]? _previous;

    /// <summary>Creates the schedule of <paramref name="steps"/> steps.</summary>
    /// <param name="steps">The number of steps (at least 1).</param>
    /// <param name="sigmaMin">The smallest nonzero sigma.</param>
    /// <param name="sigmaMax">The largest sigma.</param>
    /// <param name="sigmaData">The sigma of the data.</param>
    public DpmSolverSdeSampler(int steps, double sigmaMin, double sigmaMax, double sigmaData)
    {
        _sigmaData = sigmaData;
        _sigmas = new double[steps + 1];
        double low = Math.Log(sigmaMin);
        double high = Math.Log(sigmaMax);
        for (int i = 0; i < steps; i++)
        {
            // linspace(log σ_min, log σ_max, steps), exponentiated and reversed; computed in f32 as the reference does.
            float position = steps == 1 ? (float)low : (float)(low + ((high - low) * (steps - 1 - i) / (steps - 1)));
            _sigmas[i] = MathF.Exp(position);
        }

        Steps = steps;
        InitialSigma = Math.Sqrt((sigmaMax * sigmaMax) + 1);
    }

    /// <summary>The number of steps.</summary>
    public int Steps { get; }

    /// <summary>The scale of the initial noise (<c>√(σ_max² + 1)</c>).</summary>
    public double InitialSigma { get; }

    /// <summary>The sigma of step <paramref name="step"/> (the sigma after the last step is 0).</summary>
    /// <param name="step">The step, 0 to <see cref="Steps"/>.</param>
    public double Sigma(int step) => _sigmas[step];

    /// <summary>The factor the latents are multiplied with before the network sees them at step <paramref name="step"/>.</summary>
    /// <param name="step">The step.</param>
    public float InputScale(int step) => (float)(1.0 / Math.Sqrt((_sigmas[step] * _sigmas[step]) + (_sigmaData * _sigmaData)));

    /// <summary>The network's time at step <paramref name="step"/>.</summary>
    /// <param name="step">The step.</param>
    public float NetworkTime(int step) => (float)(Math.Atan(_sigmas[step]) * 2 / Math.PI);

    /// <summary>Advances the latents by one step.</summary>
    /// <param name="step">The step (called in order from 0).</param>
    /// <param name="sample">The latents, replaced by the latents at the next sigma.</param>
    /// <param name="prediction">The network's (guided) v prediction for <paramref name="sample"/>.</param>
    /// <param name="noise">Standard normal noise of the step.</param>
    public void Step(int step, Span<float> sample, ReadOnlySpan<float> prediction, ReadOnlySpan<float> noise)
    {
        double sigma = _sigmas[step];
        double next = _sigmas[step + 1];
        double variance = (sigma * sigma) + (_sigmaData * _sigmaData);
        float skip = (float)(_sigmaData * _sigmaData / variance);
        float output = (float)(-sigma * _sigmaData / Math.Sqrt(variance));
        var denoised = new float[sample.Length];
        for (int i = 0; i < denoised.Length; i++)
        {
            denoised[i] = (skip * sample[i]) + (output * prediction[i]);
        }

        // h = λ_t − λ_s with λ = −log σ; at the final sigma 0, h is infinite and the step returns the denoised estimate.
        double decay = next == 0 ? 0 : next / sigma * (next / sigma);
        double blend = 1 - decay;
        float keep = (float)decay;
        float toward = (float)blend;
        float spread = (float)(next * Math.Sqrt(blend));
        bool secondOrder = _previous is not null && step + 1 < Steps;
        if (secondOrder)
        {
            // D1 = (m0 − m1) / r0 with r0 = h_0 / h, h_0 = log(σ_{s-1} / σ_s), h = log(σ_s / σ_t); the midpoint solver adds half of it.
            double h = Math.Log(sigma / next);
            double h0 = Math.Log(_sigmas[step - 1] / sigma);
            float correction = (float)(0.5 * blend * h / h0);
            var previous = _previous!;
            for (int i = 0; i < sample.Length; i++)
            {
                sample[i] = (keep * sample[i]) + (toward * denoised[i]) + (correction * (denoised[i] - previous[i])) + (spread * noise[i]);
            }
        }
        else
        {
            for (int i = 0; i < sample.Length; i++)
            {
                sample[i] = (keep * sample[i]) + (toward * denoised[i]) + (spread * noise[i]);
            }
        }

        _previous = denoised;
    }
}
