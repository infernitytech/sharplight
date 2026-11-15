namespace Sharplight.ImageGeneration;

/// <summary>
/// Ancestral Euler sampling (η = 1), as diffusers' <c>EulerAncestralDiscreteScheduler</c>: each step moves deterministically to a
/// lower noise level σ↓ and adds fresh noise of level σ↑ so the sample lands at the next level.
/// </summary>
internal sealed class EulerAncestralSampler : SigmaSampler
{
    /// <summary>Prepares a run.</summary>
    /// <param name="schedule">The model's noise schedule.</param>
    /// <param name="steps">Denoising steps.</param>
    public EulerAncestralSampler(NoiseSchedule schedule, int steps)
        : base(schedule, steps, karras: false)
    {
    }

    /// <inheritdoc/>
    public override void Step(int index, ReadOnlySpan<float> prediction, Span<float> sample, Func<float[]> noise)
    {
        float sigma = Sigmas[index];
        float next = Sigmas[index + 1];
        var original = new float[sample.Length];
        Denoised(prediction, sample, sigma, original);
        float up = MathF.Sqrt(next * next * (sigma * sigma - next * next) / (sigma * sigma));
        float down = MathF.Sqrt(next * next - up * up);
        float dt = down - sigma;
        for (int i = 0; i < sample.Length; i++)
        {
            float derivative = (sample[i] - original[i]) / sigma;
            sample[i] += derivative * dt;
        }

        if (up > 0)
        {
            var fresh = noise();
            for (int i = 0; i < sample.Length; i++)
            {
                sample[i] += fresh[i] * up;
            }
        }

        PredictedOriginal = original;
    }
}
