using System.Numerics;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// DFT along the second-to-last axis of a dense f32 <c>[..., N, C]</c> (see <c>StandardOps.Dft</c>), one FFT per row in double precision.
/// </summary>
internal sealed unsafe class DftOp(TRef x, TRef y, int length, int outBins, bool inverse, bool realOutput) : CpuOp
{
    private readonly FftPlan _plan = new(length);
    private readonly int _inLength = (int)x.Dims[^2];
    private readonly int _inComponents = (int)x.Dims[^1];

    public override string Name => "Dft";

    public override void Run(RunParameters p)
    {
        long rows = x.Elements(p) / (_inLength * _inComponents);
        var buffer = new Complex[length];
        var scratch = new Complex[_plan.ScratchLength];
        float* input = (float*)x.P;
        float* output = (float*)y.P;
        int outComponents = realOutput ? 1 : 2;
        for (long r = 0; r < rows; r++)
        {
            Load(input + (r * _inLength * _inComponents), buffer);
            _plan.Transform(buffer, inverse, scratch);
            float* o = output + (r * outBins * outComponents);
            double scale = inverse ? 1.0 / length : 1.0;
            for (int k = 0; k < outBins; k++)
            {
                o[k * outComponents] = (float)(buffer[k].Real * scale);
                if (!realOutput)
                {
                    o[(k * 2) + 1] = (float)(buffer[k].Imaginary * scale);
                }
            }
        }
    }

    /// <summary>The row as a length-L complex signal: truncated or zero-padded, or the Hermitian extension of a one-sided spectrum.</summary>
    private void Load(float* row, Span<Complex> buffer)
    {
        buffer.Clear();
        int used = Math.Min(_inLength, realOutput ? (length / 2) + 1 : length);
        for (int n = 0; n < used; n++)
        {
            double re = row[n * _inComponents];
            double im = _inComponents == 2 ? row[(n * 2) + 1] : 0;
            buffer[n] = new Complex(re, im);
        }

        if (!realOutput)
        {
            return;
        }

        // numpy irfft: the imaginary parts of the DC and Nyquist bins are ignored; bin k mirrors to L - k as its conjugate.
        buffer[0] = new Complex(buffer[0].Real, 0);
        if (length % 2 == 0 && length / 2 < used)
        {
            buffer[length / 2] = new Complex(buffer[length / 2].Real, 0);
        }

        for (int k = 1; k < used; k++)
        {
            if (length - k > k)
            {
                buffer[length - k] = Complex.Conjugate(buffer[k]);
            }
        }
    }
}
