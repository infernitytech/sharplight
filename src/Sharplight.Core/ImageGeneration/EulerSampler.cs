namespace Sharplight.ImageGeneration;

/// <summary>
/// Euler's method on the probability-flow ODE in σ (Karras et al. 2022, Algorithm 2 without churn), as diffusers'
/// <c>EulerDiscreteScheduler</c>; deterministic. With <c>karras</c> the noise levels follow the Karras schedule.
/// </summary>
internal sealed class EulerSampler : SigmaSampler
{
    /// <summary>Prepares a run.</summary>
    /// <param name="schedule">The model's noise schedule.</param>
    /// <param name="steps">Denoising steps.</param>
    /// <param name="karras">Whether to use the Karras noise levels.</param>
    public EulerSampler(NoiseSchedule schedule, int steps, bool karras)
        : base(schedule, steps, karras)
    {
    }

    /// <inheritdoc/>
    public override void Step(int index, ReadOnlySpan<float> prediction, Span<float> sample, Func<float[]> noise)
    {
        float sigma = Sigmas[index];
        float next = Sigmas[index + 1];
        var original = new float[sample.Length];
        Denoised(prediction, sample, sigma, original);
        float dt = next - sigma;
        for (int i = 0; i < sample.Length; i++)
        {
            float derivative = (sample[i] - original[i]) / sigma;
            sample[i] += derivative * dt;
        }

        PredictedOriginal = original;
    }
}
