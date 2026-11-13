namespace Sharplight.Onnx;

/// <summary>
/// One axis of an ONNX Resize as a linear map: <c>out[o] = Σ_i Weights[o, i] · in[i] + Bias[o]</c>. ONNX Resize is separable, so an N-D
/// resize is a sequence of these maps, one per resized axis (the order of the ONNX reference implementation). Covers every mode
/// (nearest with all rounding modes, linear, cubic), coordinate transformation mode (including tf_crop_and_resize with its
/// extrapolation value, which lands in <see cref="Bias"/>), antialiasing and exclude_outside.
/// </summary>
internal sealed class ResizeAxis
{
    private ResizeAxis(double[] weights, double[]? bias)
    {
        Weights = weights;
        Bias = bias;
    }

    /// <summary>Row-major [outLen, inLen] interpolation weights.</summary>
    public double[] Weights { get; }

    /// <summary>Per-output constant (tf_crop_and_resize extrapolation), or null when every output is a pure interpolation.</summary>
    public double[]? Bias { get; }

    /// <summary>Builds the map of one axis.</summary>
    /// <param name="inLen">Input extent.</param>
    /// <param name="outLen">Output extent.</param>
    /// <param name="scale">Scale factor of the axis (the given scale, or outLen / inLen when sizes were given).</param>
    /// <param name="parameters">Mode, coordinate transformation and the other Resize attributes.</param>
    /// <param name="roiStart">tf_crop_and_resize region start (normalized).</param>
    /// <param name="roiEnd">tf_crop_and_resize region end (normalized).</param>
    public static ResizeAxis Build(int inLen, int outLen, double scale, ResizeParameters parameters, double roiStart, double roiEnd)
    {
        var weights = new double[(long)outLen * inLen];
        double[]? bias = null;
        double outWidth = scale * inLen;
        for (int o = 0; o < outLen; o++)
        {
            double x = SourceCoordinate(o, inLen, outLen, outWidth, scale, parameters.CoordinateMode, roiStart, roiEnd);
            if (parameters.CoordinateMode == "tf_crop_and_resize" && (x < 0 || x > inLen - 1))
            {
                bias ??= new double[outLen];
                bias[o] = parameters.ExtrapolationValue;
                continue;
            }

            double floor = Math.Floor(x);
            double ratio = x == floor ? 1.0 : x - floor;
            double[] coeffs = Coefficients(ratio, scale, parameters);
            int n = coeffs.Length;
            int pad = (n + 1) / 2;
            double padded = x + pad;
            double p = Math.Floor(padded);
            double frac = padded - p;
            long offset = n % 2 == 0
                ? (frac == 0 ? -(n / 2) : -(n / 2) + 1)
                : (frac <= 0.5 ? -((n - 1) / 2) : -((n - 1) / 2) + 1);
            long start = (long)p + offset - pad;
            if (parameters.ExcludeOutside)
            {
                double sum = 0;
                for (int k = 0; k < n; k++)
                {
                    if (start + k < 0 || start + k >= inLen)
                    {
                        coeffs[k] = 0;
                    }

                    sum += coeffs[k];
                }

                for (int k = 0; k < n && sum != 0; k++)
                {
                    coeffs[k] /= sum;
                }
            }

            for (int k = 0; k < n; k++)
            {
                long i = Math.Clamp(start + k, 0, inLen - 1);
                weights[(long)o * inLen + i] += coeffs[k];
            }
        }

        return new ResizeAxis(weights, bias);
    }

    private static double SourceCoordinate(int o, int inLen, int outLen, double outWidth, double scale, string mode, double roiStart, double roiEnd) => mode switch
    {
        "align_corners" => outWidth == 1 ? 0 : o * (inLen - 1) / (outWidth - 1),
        "asymmetric" => o / scale,
        "tf_crop_and_resize" => (outWidth == 1 ? (roiEnd - roiStart) * (inLen - 1) / 2 : o * (roiEnd - roiStart) * (inLen - 1) / (outWidth - 1)) + roiStart * (inLen - 1),
        "pytorch_half_pixel" => outWidth == 1 ? -0.5 : (o + 0.5) / scale - 0.5,
        "half_pixel" => (o + 0.5) / scale - 0.5,
        "half_pixel_symmetric" => inLen / 2.0 * (1 - outLen / outWidth) + (o + 0.5) / scale - 0.5,
        _ => throw new NotSupportedException($"Resize coordinate_transformation_mode '{mode}'"),
    };

    private static double[] Coefficients(double ratio, double scale, ResizeParameters parameters)
    {
        switch (parameters.Mode)
        {
            case "nearest":
                if (ratio == Math.Floor(ratio))
                {
                    return [0, 1];
                }

                return parameters.NearestMode switch
                {
                    "round_prefer_floor" => [ratio <= 0.5 ? 1 : 0, ratio > 0.5 ? 1 : 0],
                    "round_prefer_ceil" => [ratio < 0.5 ? 1 : 0, ratio >= 0.5 ? 1 : 0],
                    "floor" => [1, 0],
                    "ceil" => [0, 1],
                    _ => throw new NotSupportedException($"Resize nearest_mode '{parameters.NearestMode}'"),
                };
            case "linear" when parameters.Antialias && scale < 1:
                return LinearAntialias(ratio, scale);
            case "linear":
                return [1 - ratio, ratio];
            case "cubic" when parameters.Antialias && scale < 1:
                return CubicAntialias(ratio, scale, parameters.CubicCoeffA);
            case "cubic":
                return Cubic(ratio, parameters.CubicCoeffA);
            default:
                throw new NotSupportedException($"Resize mode '{parameters.Mode}'");
        }
    }

    private static double[] Cubic(double r, double a) =>
    [
        ((a * (r + 1) - 5 * a) * (r + 1) + 8 * a) * (r + 1) - 4 * a,
        ((a + 2) * r - (a + 3)) * r * r + 1,
        ((a + 2) * (1 - r) - (a + 3)) * (1 - r) * (1 - r) + 1,
        ((a * (2 - r) - 5 * a) * (2 - r) + 8 * a) * (2 - r) - 4 * a,
    ];

    private static double[] LinearAntialias(double ratio, double scale)
    {
        int start = (int)Math.Floor(-1 / scale) + 1;
        int footprint = 2 - 2 * start;
        var coeffs = new double[footprint];
        for (int k = 0; k < footprint; k++)
        {
            coeffs[k] = Math.Clamp(1 - Math.Abs((start + k - ratio) * scale), 0, 1);
        }

        return Normalized(coeffs);
    }

    private static double[] CubicAntialias(double ratio, double scale, double a)
    {
        int start = (int)Math.Floor(-2 / scale) + 1;
        int end = 2 - start;
        var coeffs = new double[end - start];
        for (int i = start; i < end; i++)
        {
            double x = Math.Abs(scale * (i - ratio));
            coeffs[i - start] = x <= 1 ? (a + 2) * x * x * x - (a + 3) * x * x + 1
                : x < 2 ? a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a
                : 0;
        }

        return Normalized(coeffs);
    }

    private static double[] Normalized(double[] coeffs)
    {
        double sum = coeffs.Sum();
        for (int k = 0; k < coeffs.Length; k++)
        {
            coeffs[k] /= sum;
        }

        return coeffs;
    }
}
