namespace Sharplight.Audio;

/// <summary>
/// Discrete Fourier transform of any length by mixed-radix decimation in time (split by the smallest prime factor, down to single
/// points), in double precision with a precomputed twiddle table. Immutable after construction and thread-safe; every call brings its own
/// scratch space. Used by the host-side audio feature extractors.
/// </summary>
internal sealed class FourierTransform
{
    private readonly double[] _cos;
    private readonly double[] _sin;

    /// <summary>Plans a transform of <paramref name="length"/> points.</summary>
    /// <param name="length">The transform length (positive).</param>
    public FourierTransform(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        Length = length;
        _cos = new double[length];
        _sin = new double[length];
        for (int k = 0; k < length; k++)
        {
            double angle = 2 * Math.PI * k / length;
            _cos[k] = Math.Cos(angle);
            _sin[k] = Math.Sin(angle);
        }
    }

    /// <summary>The transform length.</summary>
    public int Length { get; }

    /// <summary>
    /// The forward transform <c>X[k] = Σ x[j] · e^(−2πi·jk/N)</c> of a real signal.
    /// </summary>
    /// <param name="signal">N real input samples.</param>
    /// <param name="real">Receives the N real parts.</param>
    /// <param name="imaginary">Receives the N imaginary parts.</param>
    /// <param name="scratch">Work space of at least 2·N values.</param>
    public void Forward(ReadOnlySpan<double> signal, Span<double> real, Span<double> imaginary, Span<double> scratch)
    {
        if (signal.Length != Length || real.Length < Length || imaginary.Length < Length || scratch.Length < 2 * Length)
        {
            throw new ArgumentException($"A {Length}-point transform needs {Length} samples, {Length} outputs per part and {2 * Length} scratch values.");
        }

        Transform(signal, 0, 1, Length, real, imaginary, 0, scratch[..Length], scratch.Slice(Length, Length));
    }

    /// <summary>
    /// Transforms the <paramref name="count"/> points <c>signal[offset + j · stride]</c> into <c>real/imaginary[output .. output + count)</c>.
    /// </summary>
    private void Transform(ReadOnlySpan<double> signal, int offset, int stride, int count, Span<double> real, Span<double> imaginary, int output,
        Span<double> scratchReal, Span<double> scratchImaginary)
    {
        if (count == 1)
        {
            real[output] = signal[offset];
            imaginary[output] = 0;
            return;
        }

        int radix = SmallestFactor(count);
        int part = count / radix;
        for (int r = 0; r < radix; r++)
        {
            Transform(signal, offset + (r * stride), stride * radix, part, real, imaginary, output + (r * part), scratchReal, scratchImaginary);
        }

        // X[k] = Σ_r W_count^(r·k) · Y_r[k mod part], with W_count^q = W_N^(q · N / count).
        int twiddleStep = Length / count;
        for (int k = 0; k < count; k++)
        {
            double sumReal = 0;
            double sumImaginary = 0;
            int inner = k % part;
            for (int r = 0; r < radix; r++)
            {
                int index = (int)((long)r * k % count * twiddleStep);
                double c = _cos[index];
                double s = -_sin[index];
                double yr = real[output + (r * part) + inner];
                double yi = imaginary[output + (r * part) + inner];
                sumReal += (yr * c) - (yi * s);
                sumImaginary += (yr * s) + (yi * c);
            }

            scratchReal[output + k] = sumReal;
            scratchImaginary[output + k] = sumImaginary;
        }

        scratchReal.Slice(output, count).CopyTo(real[output..]);
        scratchImaginary.Slice(output, count).CopyTo(imaginary[output..]);
    }

    private static int SmallestFactor(int value)
    {
        if (value % 2 == 0)
        {
            return 2;
        }

        for (int factor = 3; factor * factor <= value; factor += 2)
        {
            if (value % factor == 0)
            {
                return factor;
            }
        }

        return value;
    }
}
