namespace Sharplight.Audio;

/// <summary>
/// Adaptive projected guidance for rectified-flow models (Stable Audio 3's base models): the guidance direction (conditional minus unconditional denoised latents) keeps
/// only its part orthogonal to the conditional denoised latents, measured over the valid frames, which avoids the over-saturation of plain classifier-free guidance.
/// </summary>
internal static class ProjectedGuidance
{
    /// <summary>Replaces the conditional velocity by the guided one.</summary>
    /// <param name="x">The latents at time <paramref name="t"/>, frame-major.</param>
    /// <param name="velocity">The conditional velocity; receives the guided velocity.</param>
    /// <param name="unconditional">The unconditional velocity.</param>
    /// <param name="t">The time (&gt; 0).</param>
    /// <param name="scale">The guidance scale.</param>
    /// <param name="valid">1 for the valid frames, 0 for the padding frames.</param>
    /// <param name="channels">The values per frame.</param>
    public static void Apply(ReadOnlySpan<float> x, Span<float> velocity, ReadOnlySpan<float> unconditional, float t, float scale, ReadOnlySpan<float> valid, int channels)
    {
        int count = x.Length;
        var conditional = new float[count];
        var difference = new float[count];
        double squares = 0;
        for (int i = 0; i < count; i++)
        {
            conditional[i] = x[i] - (velocity[i] * t);
            difference[i] = conditional[i] - (x[i] - (unconditional[i] * t));
            if (valid[i / channels] != 0f)
            {
                squares += (double)conditional[i] * conditional[i];
            }
        }

        float norm = (float)Math.Max(Math.Sqrt(squares), 1e-8);
        double dot = 0;
        for (int i = 0; i < count; i++)
        {
            if (valid[i / channels] != 0f)
            {
                dot += (double)difference[i] * (conditional[i] / norm);
            }
        }

        float projection = (float)dot;
        for (int i = 0; i < count; i++)
        {
            float mask = valid[i / channels];
            float orthogonal = (difference[i] - (projection * (conditional[i] * mask / norm))) * mask;
            float guided = conditional[i] + ((scale - 1f) * orthogonal);
            velocity[i] = (x[i] - guided) / t;
        }
    }
}
