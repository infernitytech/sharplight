namespace Sharplight.Models;

/// <summary>
/// The standard normal distribution's CDF and quantile, for importers and graph builders that turn a target fraction into a cutoff in standard
/// deviations (Gemma 3n's activation sparsity) or back.
/// </summary>
public static class NormalDistribution
{
    // Acklam's rational approximation of the quantile (relative error below 1.2e-9).
    private static readonly double[] s_a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    private static readonly double[] s_b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    private static readonly double[] s_c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    private static readonly double[] s_d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];

    /// <summary>P(Z ≤ <paramref name="z"/>) for a standard normal Z; 0 for −∞ and 1 for +∞.</summary>
    public static double Cdf(double z) => 0.5 * Erfc(-z / Math.Sqrt(2.0));

    /// <summary>The z with P(Z ≤ z) = <paramref name="p"/>; −∞ for p ≤ 0 and +∞ for p ≥ 1.</summary>
    public static double Quantile(double p)
    {
        if (p <= 0)
        {
            return double.NegativeInfinity;
        }

        if (p >= 1)
        {
            return double.PositiveInfinity;
        }

        const double low = 0.02425;
        if (p < low)
        {
            double q = Math.Sqrt(-2 * Math.Log(p));
            return Tail(q);
        }

        if (p > 1 - low)
        {
            double q = Math.Sqrt(-2 * Math.Log(1 - p));
            return -Tail(q);
        }

        double r = p - 0.5;
        double s = r * r;
        return (((((s_a[0] * s + s_a[1]) * s + s_a[2]) * s + s_a[3]) * s + s_a[4]) * s + s_a[5]) * r
            / (((((s_b[0] * s + s_b[1]) * s + s_b[2]) * s + s_b[3]) * s + s_b[4]) * s + 1);
    }

    private static double Tail(double q)
        => (((((s_c[0] * q + s_c[1]) * q + s_c[2]) * q + s_c[3]) * q + s_c[4]) * q + s_c[5])
        / ((((s_d[0] * q + s_d[1]) * q + s_d[2]) * q + s_d[3]) * q + 1);

    // Complementary error function (Numerical Recipes' Chebyshev fit, fractional error below 1.2e-7).
    private static double Erfc(double x)
    {
        double z = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.5 * z);
        double r = t * Math.Exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806
            + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? r : 2.0 - r;
    }
}
