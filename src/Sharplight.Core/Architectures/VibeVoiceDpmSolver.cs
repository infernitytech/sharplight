namespace Sharplight.Architectures;

/// <summary>
/// The sampler of VibeVoice's diffusion head, a port of diffusers' <c>DPMSolverMultistepScheduler</c> as the model's generation config sets it (DPM-Solver++ of order 2 with the midpoint
/// rule, a cosine cumulative-product schedule over 1000 training steps, <c>v_prediction</c>, linspace timesteps, the last step of first order, a final sigma of zero): <see cref="Begin"/>
/// a run of <c>n</c> steps, then for each step feed the model's output for <see cref="Timestep"/> to <see cref="Step"/>.
/// </summary>
internal sealed class VibeVoiceDpmSolver
{
    private readonly float[] _alphasCumprod;
    private readonly bool _velocity;
    private int[] _timesteps = [];
    private float[] _sigmas = [];
    private float[]? _previous;
    private float[]? _latest;
    private int _step;
    private int _lowerOrderSteps;

    /// <summary>Creates the sampler for a training schedule.</summary>
    /// <param name="trainSteps">The training steps of the schedule.</param>
    /// <param name="betaSchedule"><c>cosine</c> or <c>squaredcos_cap_v2</c> (the Glide cosine schedule), or <c>linear</c>.</param>
    /// <param name="predictionType"><c>v_prediction</c> or <c>epsilon</c>.</param>
    public VibeVoiceDpmSolver(int trainSteps, string betaSchedule, string predictionType)
    {
        _velocity = predictionType == "v_prediction";
        if (!_velocity && predictionType != "epsilon")
        {
            throw new NotSupportedException($"The diffusion prediction type '{predictionType}' is not supported.");
        }

        _alphasCumprod = new float[trainSteps];
        float product = 1f;
        for (int i = 0; i < trainSteps; i++)
        {
            float beta = betaSchedule switch
            {
                "cosine" or "squaredcos_cap_v2" => CosineBeta(i, trainSteps),
                "linear" => 0.0001f + ((0.02f - 0.0001f) * i / (trainSteps - 1)),
                _ => throw new NotSupportedException($"The diffusion beta schedule '{betaSchedule}' is not supported."),
            };
            product *= 1f - beta;
            _alphasCumprod[i] = product;
        }
    }

    /// <summary>The training timestep of the step <paramref name="index"/> of the current run.</summary>
    /// <param name="index">The step.</param>
    public int Timestep(int index) => _timesteps[index];

    /// <summary>The number of steps of the current run.</summary>
    public int Steps => _timesteps.Length;

    /// <summary>Starts a run of <paramref name="steps"/> steps.</summary>
    /// <param name="steps">The number of steps.</param>
    public void Begin(int steps)
    {
        int last = _alphasCumprod.Length;
        _timesteps = new int[steps];
        _sigmas = new float[steps + 1];
        for (int i = 0; i < steps; i++)
        {
            // numpy: linspace(0, last - 1, steps + 1).round()[::-1][:-1]
            double position = (last - 1) * (double)(steps - i) / steps;
            int t = (int)Math.Round(position, MidpointRounding.ToEven);
            _timesteps[i] = t;
            float ac = _alphasCumprod[t];
            _sigmas[i] = MathF.Sqrt((1f - ac) / ac);
        }

        _sigmas[steps] = 0f;
        _previous = null;
        _latest = null;
        _step = 0;
        _lowerOrderSteps = 0;
    }

    /// <summary>One step: the sample at the next timestep from the model's output for the current one.</summary>
    /// <param name="modelOutput">The model's output.</param>
    /// <param name="sample">The current sample.</param>
    public float[] Step(float[] modelOutput, float[] sample)
    {
        int n = sample.Length;
        var (alpha, sigma) = AlphaSigma(_sigmas[_step]);
        var data = new float[n];
        for (int i = 0; i < n; i++)
        {
            data[i] = _velocity ? (float)((alpha * sample[i]) - (sigma * modelOutput[i])) : (float)((sample[i] - (sigma * modelOutput[i])) / alpha);
        }

        _previous = _latest;
        _latest = data;
        int count = _timesteps.Length;
        bool lowerFinal = _step == count - 1;
        var next = new float[n];
        var (alphaT, sigmaT) = AlphaSigma(_sigmas[_step + 1]);
        var (alphaS, sigmaS) = AlphaSigma(_sigmas[_step]);
        double lambdaT = Math.Log(alphaT) - Math.Log(sigmaT);
        double lambdaS = Math.Log(alphaS) - Math.Log(sigmaS);
        double h = lambdaT - lambdaS;
        double decay = Math.Exp(-h) - 1.0;
        if (_lowerOrderSteps < 1 || lowerFinal || _previous is null)
        {
            for (int i = 0; i < n; i++)
            {
                next[i] = (float)(((sigmaT / sigmaS) * sample[i]) - (alphaT * decay * data[i]));
            }
        }
        else
        {
            var (alphaP, sigmaP) = AlphaSigma(_sigmas[_step - 1]);
            double lambdaP = Math.Log(alphaP) - Math.Log(sigmaP);
            double r0 = (lambdaS - lambdaP) / h;
            for (int i = 0; i < n; i++)
            {
                double d1 = (1.0 / r0) * (data[i] - _previous[i]);
                next[i] = (float)(((sigmaT / sigmaS) * sample[i]) - (alphaT * decay * data[i]) - (0.5 * alphaT * decay * d1));
            }
        }

        if (_lowerOrderSteps < 2)
        {
            _lowerOrderSteps++;
        }

        _step++;
        return next;
    }

    private static (double Alpha, double Sigma) AlphaSigma(float sigma)
    {
        double alpha = 1.0 / Math.Sqrt(((double)sigma * sigma) + 1.0);
        return (alpha, sigma * alpha);
    }

    private static float CosineBeta(int i, int steps)
    {
        static double AlphaBar(double t) => Math.Pow(Math.Cos((t + 0.008) / 1.008 * Math.PI / 2), 2);
        double first = (double)i / steps;
        double second = (double)(i + 1) / steps;
        return (float)Math.Min(1 - (AlphaBar(second) / AlphaBar(first)), 0.999);
    }
}
