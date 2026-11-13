namespace Sharplight.Models;

/// <summary>
/// Format-neutral RoPE frequency scaling math shared by importers: turns a scaling description (llama3, yarn, linear)
/// into the canonical per-frequency divisors <c>rope.freq_factors</c> (θᵢ = position · base^(-2i/dim) / factorᵢ, the
/// convention of the <c>Rope</c> op's third input) plus an attention scale that folds yarn's cos/sin magnitude scaling.
/// </summary>
public static class RopeScaling
{
    /// <summary>Result of a scaling computation: <see cref="Factors"/> has dim/2 entries.</summary>
    public sealed record Result(float[] Factors, float AttentionScale);

    /// <summary>Linear scaling: positions are divided by <paramref name="factor"/> for every frequency.</summary>
    public static Result Linear(int dim, float factor) => new(Enumerable.Repeat(factor, dim / 2).ToArray(), 1f);

    /// <summary>
    /// "Proportional" partial rotary (Gemma 4 global layers): the whole head (<paramref name="dim"/> dims) is the frequency basis
    /// (θᵢ = base^(-2i/dim)) but only the first <paramref name="rotatedAngles"/> frequencies rotate; the rest get a huge divisor (no rotation).
    /// </summary>
    public static Result Proportional(int dim, int rotatedAngles, float factor = 1f)
    {
        var f = new float[dim / 2];
        for (int i = 0; i < f.Length; i++) f[i] = i < rotatedAngles ? factor : 1e30f;
        return new Result(f, 1f);
    }

    /// <summary>Llama 3.x scaling (long wavelengths divided by <paramref name="factor"/>, smooth blend in between).</summary>
    public static Result Llama3(int dim, float @base, float factor, float lowFreqFactor, float highFreqFactor, int originalContext)
    {
        var f = new float[dim / 2];
        double lowWave = originalContext / (double)lowFreqFactor, highWave = originalContext / (double)highFreqFactor;
        for (int i = 0; i < f.Length; i++)
        {
            double wavelen = 2 * Math.PI * Math.Pow(@base, 2.0 * i / dim);
            if (wavelen < highWave) f[i] = 1f;
            else if (wavelen > lowWave) f[i] = factor;
            else
            {
                double smooth = (originalContext / wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor);
                f[i] = (float)(1.0 / ((1 - smooth) / factor + smooth));
            }
        }
        return new Result(f, 1f);
    }

    /// <summary>YaRN scaling (NTK-by-parts blend between interpolation and extrapolation, plus magnitude scale).</summary>
    /// <param name="attentionFactor">cos/sin multiplier; null = 0.1·ln(factor)+1.</param>
    public static Result Yarn(int dim, float @base, float factor, int originalContext, float betaFast = 32f, float betaSlow = 1f,
                              float? attentionFactor = null, bool truncate = true)
    {
        double CorrDim(double rot) => dim * Math.Log(originalContext / (rot * 2 * Math.PI)) / (2 * Math.Log(@base));
        double low = CorrDim(betaFast), high = CorrDim(betaSlow);
        if (truncate) { low = Math.Floor(low); high = Math.Ceiling(high); }
        low = Math.Max(low, 0); high = Math.Min(high, dim - 1);
        if (low == high) high += 0.001;
        var f = new float[dim / 2];
        for (int i = 0; i < f.Length; i++)
        {
            double ramp = Math.Clamp((i - low) / (high - low), 0, 1);
            double extrap = 1 - ramp;                                  // weight of the unscaled frequency
            f[i] = (float)(1.0 / ((1 - extrap) / factor + extrap));    // inv_freq_final = inv/factor·(1-extrap) + inv·extrap
        }
        float mscale = attentionFactor ?? (factor <= 1 ? 1f : (float)(0.1 * Math.Log(factor) + 1));
        return new Result(f, mscale);
    }
}
