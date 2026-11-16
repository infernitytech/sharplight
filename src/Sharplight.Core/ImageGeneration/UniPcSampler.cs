namespace Sharplight.ImageGeneration;

/// <summary>
/// UniPC (unified predictor-corrector, second order, B(h) = eᴴ − 1, data prediction) on rectified-flow schedules, as diffusers'
/// <c>UniPCMultistepScheduler</c> with <c>use_flow_sigmas</c> and <c>prediction_type="flow_prediction"</c> (the Wan pipelines'
/// scheduler): noise levels linspace(1, 1 / train steps, steps + 1) without the last, shifted statically by
/// <see cref="NoiseSchedule.Shift"/> (the scheduler's <c>flow_shift</c>), then 0; integer model timesteps ⌊σ · train steps⌋. Each step
/// converts the velocity into the clean-sample estimate x₀ = x − σ · v, corrects the previous step with it (UniC) and predicts the next
/// sample (UniP) from the last two estimates; the first step and the last one are first order.
/// </summary>
internal sealed class UniPcSampler : DiffusionSampler
{
    private const int SolverOrder = 2;
    private readonly float[] _sigmas;
    private readonly float[] _timesteps;
    private readonly List<float[]> _estimates = [];
    private float[]? _lastSample;
    private int _lastOrder;

    /// <summary>Prepares a run.</summary>
    /// <param name="schedule">The model's schedule (prediction type <c>flow</c>).</param>
    /// <param name="steps">Denoising steps.</param>
    public UniPcSampler(NoiseSchedule schedule, int steps)
        : base(schedule)
    {
        int train = schedule.TrainSteps;
        _sigmas = new float[steps + 1];
        _timesteps = new float[steps];
        double shift = schedule.Shift;
        for (int i = 0; i < steps; i++)
        {
            // np.linspace(1, 1 / train, steps + 1)[:-1], shifted, in double; the first level is nudged below 1 (log(0) otherwise).
            double linear = 1.0 + (1.0 / train - 1.0) * i / steps;
            double sigma = shift * linear / (1 + (shift - 1) * linear);
            if (i == 0 && Math.Abs(sigma - 1) < 1e-6)
            {
                sigma -= 1e-6;
            }

            _sigmas[i] = (float)sigma;
            _timesteps[i] = (long)(sigma * train);
        }
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
        float sigma = _sigmas[index];
        var estimate = new float[sample.Length];
        for (int i = 0; i < sample.Length; i++)
        {
            estimate[i] = sample[i] - sigma * prediction[i];
        }

        if (index > 0 && _lastSample is not null)
        {
            Correct(index, estimate, sample);
        }

        _estimates.Add(estimate);
        if (_estimates.Count > SolverOrder)
        {
            _estimates.RemoveAt(0);
        }

        int order = Math.Min(Math.Min(SolverOrder, _timesteps.Length - index), _estimates.Count);
        _lastOrder = order;
        _lastSample = sample.ToArray();
        Predict(index, order, sample);
        PredictedOriginal = estimate;
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

    /// <summary>λ = log(α) − log(σ) of a flow noise level (α = 1 − σ).</summary>
    private static double Lambda(float sigma) => Math.Log(1.0 - sigma) - Math.Log(sigma);

    /// <summary>The coefficients b_k = h φ_{k}(h) · k! / B(h) for k = 1..order (with hh = −h, data prediction).</summary>
    private static (double PhiOne, double Bh, double[] B) Coefficients(double h, int order)
    {
        double hh = -h;
        double phiOne = Math.Exp(hh) - 1;
        double phiK = phiOne / hh - 1;
        double bh = Math.Exp(hh) - 1;
        double factorial = 1;
        var b = new double[order];
        for (int i = 1; i <= order; i++)
        {
            b[i - 1] = phiK * factorial / bh;
            factorial *= i + 1;
            phiK = phiK / hh - 1 / factorial;
        }

        return (phiOne, bh, b);
    }

    /// <summary>UniP: the sample at the next noise level from the current one and the last <paramref name="order"/> estimates.</summary>
    private void Predict(int index, int order, Span<float> sample)
    {
        float sigmaT = _sigmas[index + 1];
        float sigmaS = _sigmas[index];
        double alphaT = 1.0 - sigmaT;
        double h = Lambda(sigmaT) - Lambda(sigmaS);
        var (phiOne, bh, _) = Coefficients(h, order);
        var m0 = _estimates[^1];
        double ratio = sigmaT / (double)sigmaS;
        float[]? previous = null;
        double weight = 0;
        if (order == 2)
        {
            // The second-order predictor uses ρ = 0.5 on D1 = (m1 − m0) / r1.
            double r1 = (Lambda(_sigmas[index - 1]) - Lambda(sigmaS)) / h;
            previous = _estimates[^2];
            weight = 0.5 / r1;
        }

        for (int i = 0; i < sample.Length; i++)
        {
            double x = ratio * sample[i] - alphaT * phiOne * m0[i];
            if (previous is not null)
            {
                x -= alphaT * bh * weight * (previous[i] - m0[i]);
            }

            sample[i] = (float)x;
        }
    }

    /// <summary>UniC: corrects the sample just predicted with the estimate at its noise level.</summary>
    private void Correct(int index, float[] estimate, Span<float> sample)
    {
        int order = _lastOrder;
        float sigmaT = _sigmas[index];
        float sigmaS = _sigmas[index - 1];
        double alphaT = 1.0 - sigmaT;
        double h = Lambda(sigmaT) - Lambda(sigmaS);
        var (phiOne, bh, b) = Coefficients(h, order);
        var m0 = _estimates[^1];
        var last = _lastSample!;
        double ratio = sigmaT / (double)sigmaS;
        float[]? previous = null;
        double rhoPrevious = 0;
        double rhoThis;
        if (order == 1)
        {
            rhoThis = 0.5;
        }
        else
        {
            // Solve [[1, 1], [r1, 1]] · ρ = b for the history term ρ₀ and the new term ρ₁.
            double r1 = (Lambda(_sigmas[index - 2]) - Lambda(sigmaS)) / h;
            double rho0 = (b[0] - b[1]) / (1 - r1);
            rhoThis = b[0] - rho0;
            rhoPrevious = rho0 / r1;
            previous = _estimates[^2];
        }

        for (int i = 0; i < sample.Length; i++)
        {
            double correction = rhoThis * (estimate[i] - m0[i]);
            if (previous is not null)
            {
                correction += rhoPrevious * (previous[i] - m0[i]);
            }

            sample[i] = (float)(ratio * last[i] - alphaT * phiOne * m0[i] - alphaT * bh * correction);
        }
    }
}
