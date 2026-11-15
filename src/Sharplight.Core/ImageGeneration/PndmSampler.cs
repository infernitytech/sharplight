namespace Sharplight.ImageGeneration;

/// <summary>
/// Pseudo linear multistep sampling (PLMS, the PNDM paper's linear multistep part with the Runge-Kutta warm-up skipped), as diffusers'
/// <c>PNDMScheduler</c> with <c>skip_prk_steps</c>: the first step is a Heun-like pair of evaluations at the same timestep, later steps
/// combine up to four past noise predictions. Takes one more evaluation than steps.
/// </summary>
internal sealed class PndmSampler : DiffusionSampler
{
    private readonly int[] _timesteps;
    private readonly float[] _timestepValues;
    private readonly int _stride;
    private readonly List<float[]> _history = [];
    private float[]? _firstSample;
    private int _counter;

    /// <summary>Prepares a run.</summary>
    /// <param name="schedule">The model's noise schedule.</param>
    /// <param name="steps">Denoising steps.</param>
    public PndmSampler(NoiseSchedule schedule, int steps)
        : base(schedule)
    {
        var ascending = IntegerTimesteps(steps, "leading").Reverse().ToArray();
        // The PLMS order: every timestep once, the second highest twice, descending.
        var order = ascending.Take(ascending.Length - 1).Append(ascending.Length >= 2 ? ascending[^2] : ascending[0]).Append(ascending[^1]);
        _timesteps = steps == 1 ? ascending : [.. order.Reverse()];
        _timestepValues = [.. _timesteps.Select(t => (float)t)];
        _stride = schedule.TrainSteps / steps;
        FinalAlpha = (schedule.SetAlphaToOne ?? false) ? 1f : AlphasCumulative[0];
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
        int timestep = _timesteps[index];
        int previous = timestep - _stride;
        var output = prediction.ToArray();
        if (_counter != 1)
        {
            if (_history.Count > 3)
            {
                _history.RemoveAt(0);
            }

            _history.Add(output);
        }
        else
        {
            previous = timestep;
            timestep += _stride;
        }

        float[] combined;
        if (_history.Count == 1 && _counter == 0)
        {
            combined = output;
            _firstSample = sample.ToArray();
        }
        else if (_history.Count == 1 && _counter == 1)
        {
            combined = Combine((output, 0.5f), (_history[^1], 0.5f));
            _firstSample!.CopyTo(sample);
            _firstSample = null;
        }
        else if (_history.Count == 2)
        {
            combined = Combine((_history[^1], 1.5f), (_history[^2], -0.5f));
        }
        else if (_history.Count == 3)
        {
            combined = Combine((_history[^1], 23f / 12), (_history[^2], -16f / 12), (_history[^3], 5f / 12));
        }
        else
        {
            combined = Combine((_history[^1], 55f / 24), (_history[^2], -59f / 24), (_history[^3], 37f / 24), (_history[^4], -9f / 24));
        }

        PreviousSample(sample, timestep, previous, combined);
        _counter++;
    }

    /// <inheritdoc/>
    public override void AddNoise(ReadOnlySpan<float> original, ReadOnlySpan<float> noise, int index, Span<float> sample)
        => VariancePreserving.AddNoise(AlphasCumulative[_timesteps[index]], original, noise, sample);

    /// <summary>The PNDM transfer from <paramref name="timestep"/> to <paramref name="previous"/> with a combined model output.</summary>
    private void PreviousSample(Span<float> sample, int timestep, int previous, float[] output)
    {
        float alpha = AlphasCumulative[timestep];
        float alphaPrevious = previous >= 0 ? AlphasCumulative[previous] : FinalAlpha;
        var epsilon = new float[sample.Length];
        Epsilon(output, sample, alpha, epsilon);
        float beta = 1 - alpha;
        float betaPrevious = 1 - alphaPrevious;
        float sampleCoefficient = MathF.Sqrt(alphaPrevious / alpha);
        float denominator = alpha * MathF.Sqrt(betaPrevious) + MathF.Sqrt(alpha * beta * alphaPrevious);
        var original = new float[sample.Length];
        for (int i = 0; i < sample.Length; i++)
        {
            original[i] = (sample[i] - MathF.Sqrt(beta) * epsilon[i]) / MathF.Sqrt(alpha);
            sample[i] = sampleCoefficient * sample[i] - (alphaPrevious - alpha) * epsilon[i] / denominator;
        }

        PredictedOriginal = original;
    }

    private static float[] Combine(params (float[] Values, float Weight)[] terms)
    {
        var result = new float[terms[0].Values.Length];
        foreach (var (values, weight) in terms)
        {
            for (int i = 0; i < result.Length; i++)
            {
                result[i] += weight * values[i];
            }
        }

        return result;
    }
}
