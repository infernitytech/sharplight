namespace Sharplight.Onnx;

/// <summary>
/// Constant tables of the ONNX signal operators: window functions, the mel filter bank and dense DFT basis matrices (a DFT of moderate
/// length is a MatMul against such a matrix, which runs on every provider through the existing matmul kernels).
/// </summary>
internal static class SignalMath
{
    /// <summary>Hann, Hamming or Blackman window of <paramref name="size"/> samples (periodic: denominator N, symmetric: N - 1).</summary>
    public static double[] Window(string op, long size, bool periodic)
    {
        if (size < 0)
        {
            throw new InvalidDataException($"{op}: negative window size {size}");
        }

        var w = new double[size];
        double denominator = periodic ? size : size - 1;
        for (long n = 0; n < size; n++)
        {
            double x = denominator == 0 ? 0 : 2 * Math.PI * n / denominator;
            w[n] = op switch
            {
                "HannWindow" => 0.5 - (0.5 * Math.Cos(x)),
                "HammingWindow" => (25.0 / 46.0) - (21.0 / 46.0 * Math.Cos(x)),
                "BlackmanWindow" => 0.42 - (0.5 * Math.Cos(x)) + (0.08 * Math.Cos(2 * x)),
                _ => throw new ArgumentException($"unknown window {op}", nameof(op)),
            };
        }

        return w;
    }

    /// <summary>
    /// ONNX MelWeightMatrix: <c>[dft_length / 2 + 1, num_mel_bins]</c> triangular filters between the mel-spaced bin edges (the bin
    /// positions are rounded down as in the ONNX reference implementation).
    /// </summary>
    public static double[] MelWeights(long melBins, long dftLength, long sampleRate, double lowerHz, double upperHz)
    {
        long spectrogramBins = (dftLength / 2) + 1;
        double lowMel = 2595 * Math.Log10(1 + (lowerHz / 700));
        double highMel = 2595 * Math.Log10(1 + (upperHz / 700));
        double step = (highMel - lowMel) / (melBins + 2);
        var edges = new long[melBins + 2];
        for (long i = 0; i < edges.Length; i++)
        {
            double hz = 700 * (Math.Pow(10, ((i * step) + lowMel) / 2595) - 1);
            edges[i] = (long)Math.Floor((dftLength + 1) * hz / sampleRate);
        }

        var output = new double[spectrogramBins * melBins];
        for (long i = 0; i < melBins; i++)
        {
            long low = edges[i];
            long center = edges[i + 1];
            long high = edges[i + 2];
            if (center == low)
            {
                Set(output, center, i);
            }
            else
            {
                for (long j = low; j <= center; j++)
                {
                    Set(output, j, i, (double)(j - low) / (center - low));
                }
            }

            for (long j = center; j < high; j++)
            {
                Set(output, j, i, (double)(high - j) / (high - center));
            }
        }

        return output;

        void Set(double[] o, long row, long col, double value = 1)
        {
            if (row < 0 || row >= spectrogramBins)
            {
                throw new InvalidDataException($"MelWeightMatrix: filter edge {row} lies outside the {spectrogramBins} spectrogram bins");
            }

            o[(row * melBins) + col] = value;
        }
    }

    /// <summary>
    /// Dense DFT basis as an IR MatMul weight <c>[outBins · outComponents, inLength · inComponents]</c> (row = output bin and component,
    /// column = input sample and component, components interleaved as in the ONNX <c>[..., 2]</c> layout).
    /// </summary>
    /// <param name="spec">The transform.</param>
    /// <param name="window">Per-input-sample weights folded into the basis (STFT window), or null.</param>
    public static float[] Basis(DftSpec spec, double[]? window)
    {
        int rows = spec.OutBins * spec.OutComponents;
        int cols = spec.InLength * spec.InComponents;
        var w = new float[(long)rows * cols];
        int used = Math.Min(spec.InLength, spec.InverseOneSided ? (spec.Length / 2) + 1 : spec.Length);
        for (int k = 0; k < spec.OutBins; k++)
        {
            for (int n = 0; n < used; n++)
            {
                // Forward: e^{-2πi kn/L}; inverse: e^{+2πi kn/L} / L. The angle is reduced modulo L in integers for accuracy.
                double angle = 2 * Math.PI * ((long)k * n % spec.Length) / spec.Length;
                double cos = Math.Cos(angle);
                double sin = spec.Inverse ? Math.Sin(angle) : -Math.Sin(angle);
                double scale = (spec.Inverse ? 1.0 / spec.Length : 1.0) * (window?[n] ?? 1.0);
                if (spec.InverseOneSided)
                {
                    // Real output of a Hermitian spectrum given by its first L/2 + 1 bins (numpy irfft): bins other than 0 and the
                    // Nyquist bin stand for themselves and their conjugate mirror; the imaginary parts of those two are ignored.
                    bool selfMirrored = n == 0 || (spec.Length % 2 == 0 && n == spec.Length / 2);
                    double weight = selfMirrored ? 1 : 2;
                    w[(long)k * cols + (n * spec.InComponents)] = (float)(scale * weight * cos);
                    if (spec.InComponents == 2 && !selfMirrored)
                    {
                        w[(long)k * cols + (n * 2) + 1] = (float)(-scale * weight * sin);
                    }

                    continue;
                }

                long re = (long)(k * spec.OutComponents) * cols;
                long im = re + cols;
                w[re + (n * spec.InComponents)] = (float)(scale * cos);
                w[im + (n * spec.InComponents)] = (float)(scale * sin);
                if (spec.InComponents == 2)
                {
                    w[re + (n * 2) + 1] = (float)(-scale * sin);
                    w[im + (n * 2) + 1] = (float)(scale * cos);
                }
            }
        }

        return w;
    }
}
