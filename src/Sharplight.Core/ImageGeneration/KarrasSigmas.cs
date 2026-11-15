namespace Sharplight.ImageGeneration;

/// <summary>
/// Noise-level helpers of the Karras et al. (2022) formulation shared by the samplers: the ρ = 7 schedule between two noise levels
/// and the (fractional) training timestep of a noise level by interpolation in log σ.
/// </summary>
internal static class KarrasSigmas
{
    private const double Rho = 7.0;

    /// <summary>
    /// <paramref name="count"/> noise levels from <paramref name="sigmaMax"/> down to <paramref name="sigmaMin"/>, evenly spaced in
    /// σ^(1/ρ).
    /// </summary>
    public static float[] Schedule(float sigmaMin, float sigmaMax, int count)
    {
        double minInverse = Math.Pow(sigmaMin, 1 / Rho);
        double maxInverse = Math.Pow(sigmaMax, 1 / Rho);
        var sigmas = new float[count];
        for (int i = 0; i < count; i++)
        {
            double ramp = count == 1 ? 0 : (double)i / (count - 1);
            sigmas[i] = (float)Math.Pow(maxInverse + ramp * (minInverse - maxInverse), Rho);
        }

        return sigmas;
    }

    /// <summary>The fractional training timestep whose noise level is <paramref name="sigma"/> (linear in log σ between neighbours).</summary>
    /// <param name="sigma">The noise level.</param>
    /// <param name="trainingSigmas">σ of every training timestep, ascending.</param>
    public static float Timestep(float sigma, float[] trainingSigmas)
    {
        double logSigma = Math.Log(Math.Max(sigma, 1e-10f));
        int low = 0;
        for (int t = 0; t < trainingSigmas.Length; t++)
        {
            if (logSigma - Math.Log(trainingSigmas[t]) >= 0)
            {
                low = t;
            }
        }

        low = Math.Min(low, trainingSigmas.Length - 2);
        double lowLog = Math.Log(trainingSigmas[low]);
        double highLog = Math.Log(trainingSigmas[low + 1]);
        double weight = Math.Clamp((lowLog - logSigma) / (lowLog - highLog), 0, 1);
        return (float)((1 - weight) * low + weight * (low + 1));
    }

    /// <summary>Linear interpolation of the training σ at a fractional timestep.</summary>
    public static float Interpolate(float timestep, float[] trainingSigmas)
    {
        if (timestep <= 0)
        {
            return trainingSigmas[0];
        }

        if (timestep >= trainingSigmas.Length - 1)
        {
            return trainingSigmas[^1];
        }

        int low = (int)MathF.Floor(timestep);
        float fraction = timestep - low;
        return trainingSigmas[low] + fraction * (trainingSigmas[low + 1] - trainingSigmas[low]);
    }
}
