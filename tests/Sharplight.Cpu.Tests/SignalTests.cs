using System.Numerics;
using Sharplight.Graph;

namespace Sharplight.Cpu.Tests;

/// <summary>The Dft op (FFT kernel: radix-2 and Bluestein lengths) against a direct double-precision DFT.</summary>
public class SignalTests
{
    [Theory]
    [InlineData(8, 8, 2, false)]
    [InlineData(12, 12, 1, false)]
    [InlineData(1000, 1000, 2, true)]
    [InlineData(1024, 700, 1, false)]
    [InlineData(37, 50, 2, true)]
    public void Dft_MatchesDirectTransform(int length, int inLength, int components, bool inverse)
    {
        const int rows = 3;
        var x = Kit.Rand(rows * inLength * components, 7);
        var y = Run(x, rows, inLength, components, length, length, inverse, realOutput: false);
        var expected = new float[rows * length * 2];
        for (int r = 0; r < rows; r++)
        {
            var signal = Signal(x, r, inLength, components, length);
            for (int k = 0; k < length; k++)
            {
                var sum = Complex.Zero;
                for (int n = 0; n < length; n++)
                {
                    sum += signal[n] * Complex.FromPolarCoordinates(1, (inverse ? 2 : -2) * Math.PI * ((long)k * n % length) / length);
                }

                sum /= inverse ? length : 1;
                expected[(((r * length) + k) * 2) + 0] = (float)sum.Real;
                expected[(((r * length) + k) * 2) + 1] = (float)sum.Imaginary;
            }
        }

        Kit.AssertClose(expected, y, 1e-3, 1e-4);
    }

    [Theory]
    [InlineData(10, 6)]
    [InlineData(9, 5)]
    [InlineData(2048, 1025)]
    public void Dft_RealOutputIsTheInverseOfAOneSidedSpectrum(int length, int bins)
    {
        const int rows = 2;
        var x = Kit.Rand(rows * bins * 2, 11);
        var y = Run(x, rows, bins, 2, length, length, inverse: true, realOutput: true);
        var expected = new float[rows * length];
        for (int r = 0; r < rows; r++)
        {
            for (int n = 0; n < length; n++)
            {
                double sum = 0;
                for (int k = 0; k < bins; k++)
                {
                    bool selfMirrored = k == 0 || (length % 2 == 0 && k == length / 2);
                    double angle = 2 * Math.PI * ((long)k * n % length) / length;
                    double re = x[(((r * bins) + k) * 2) + 0];
                    double im = selfMirrored ? 0 : x[(((r * bins) + k) * 2) + 1];
                    sum += (selfMirrored ? 1 : 2) * ((re * Math.Cos(angle)) - (im * Math.Sin(angle)));
                }

                expected[(r * length) + n] = (float)(sum / length);
            }
        }

        Kit.AssertClose(expected, y, 1e-4, 1e-4);
    }

    private static Complex[] Signal(float[] x, int row, int inLength, int components, int length)
    {
        var s = new Complex[length];
        for (int n = 0; n < Math.Min(inLength, length); n++)
        {
            int at = ((row * inLength) + n) * components;
            s[n] = new Complex(x[at], components == 2 ? x[at + 1] : 0);
        }

        return s;
    }

    private static float[] Run(float[] x, int rows, int inLength, int components, int length, int outBins, bool inverse, bool realOutput)
    {
        var tg = new TestGraph();
        var input = tg.InF("x", rows, inLength, components);
        var attributes = Attributes.From(("length", (long)length), ("out_bins", (long)outBins), ("inverse", inverse), ("real_output", realOutput));
        tg.B.Output("y", tg.B.Emit1("Dft", [input], attributes));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, input, x);
        var p = Kit.P(1);
        m.Run(p);
        return Kit.Out(m, "y", p);
    }
}
