namespace Sharplight.Audio;

/// <summary>
/// The short-time Fourier transform and its inverse as PyTorch computes them (<c>torch.stft</c> and <c>torch.istft</c> with a periodic Hann window, <c>center</c> and
/// <c>normalized</c>): frames are windowed, transformed and scaled by <c>1 / sqrt(fftSize)</c>; the inverse overlap-adds the windowed inverse transforms and divides by the summed squared
/// window. A radix-2 transform in double precision; the FFT size is a power of two. Not thread-safe (it owns work buffers).
/// </summary>
internal sealed class SpectralTransform
{
    private readonly int _size;
    private readonly int _hop;
    private readonly double[] _window;
    private readonly double[] _cos;
    private readonly double[] _sin;
    private readonly int[] _reverse;
    private readonly double[] _real;
    private readonly double[] _imaginary;

    /// <summary>Plans the transform.</summary>
    /// <param name="fftSize">The FFT size and window length (a power of two).</param>
    /// <param name="hop">The hop between frames.</param>
    public SpectralTransform(int fftSize, int hop)
    {
        if (fftSize < 2 || (fftSize & (fftSize - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(fftSize), "The FFT size must be a power of two.");
        }

        _size = fftSize;
        _hop = hop;
        _window = new double[fftSize];
        for (int i = 0; i < fftSize; i++)
        {
            _window[i] = 0.5 - (0.5 * Math.Cos(2 * Math.PI * i / fftSize));
        }

        _cos = new double[fftSize / 2];
        _sin = new double[fftSize / 2];
        for (int i = 0; i < fftSize / 2; i++)
        {
            _cos[i] = Math.Cos(2 * Math.PI * i / fftSize);
            _sin[i] = -Math.Sin(2 * Math.PI * i / fftSize);
        }

        _reverse = new int[fftSize];
        int bits = (int)Math.Log2(fftSize);
        for (int i = 0; i < fftSize; i++)
        {
            int r = 0;
            for (int b = 0; b < bits; b++)
            {
                r |= ((i >> b) & 1) << (bits - 1 - b);
            }

            _reverse[i] = r;
        }

        _real = new double[fftSize];
        _imaginary = new double[fftSize];
    }

    /// <summary>The frequency bins of a transform (<c>fftSize / 2 + 1</c>).</summary>
    public int Bins => (_size / 2) + 1;

    /// <summary>The number of frames <see cref="Forward"/> produces for <paramref name="length"/> samples.</summary>
    /// <param name="length">The signal length.</param>
    public int Frames(int length) => 1 + (length / _hop);

    /// <summary>The signal padded by reflection (the edge sample is not repeated), as <c>torch.nn.functional.pad(mode="reflect")</c>.</summary>
    /// <param name="signal">The signal (longer than either pad).</param>
    /// <param name="left">Samples added before.</param>
    /// <param name="right">Samples added after.</param>
    public static float[] ReflectPad(ReadOnlySpan<float> signal, int left, int right)
    {
        int n = signal.Length;
        var result = new float[left + n + right];
        for (int i = 0; i < result.Length; i++)
        {
            int j = i - left;
            int period = 2 * (n - 1);
            j = ((j % period) + period) % period;
            result[i] = signal[j < n ? j : period - j];
        }

        return result;
    }

    /// <summary>The transform of a signal (reflect-padded by half the FFT size on both sides, as <c>center=True</c>): real and imaginary parts, bin-major <c>[bins, frames]</c>.</summary>
    /// <param name="signal">The signal.</param>
    /// <param name="real">Receives the real parts.</param>
    /// <param name="imaginary">Receives the imaginary parts.</param>
    public void Forward(ReadOnlySpan<float> signal, float[] real, float[] imaginary)
    {
        var padded = ReflectPad(signal, _size / 2, _size / 2);
        int frames = Frames(signal.Length);
        double scale = 1.0 / Math.Sqrt(_size);
        for (int f = 0; f < frames; f++)
        {
            for (int i = 0; i < _size; i++)
            {
                _real[i] = padded[(f * _hop) + i] * _window[i];
                _imaginary[i] = 0;
            }

            Transform(_real, _imaginary);
            for (int k = 0; k < Bins; k++)
            {
                real[(k * frames) + f] = (float)(_real[k] * scale);
                imaginary[(k * frames) + f] = (float)(_imaginary[k] * scale);
            }
        }
    }

    /// <summary>The inverse: frames to a signal of <paramref name="length"/> samples (the samples after the first <c>fftSize / 2</c>).</summary>
    /// <param name="real">The real parts, bin-major <c>[bins, frames]</c>.</param>
    /// <param name="imaginary">The imaginary parts.</param>
    /// <param name="frames">The number of frames.</param>
    /// <param name="length">The output length.</param>
    public float[] Inverse(ReadOnlySpan<float> real, ReadOnlySpan<float> imaginary, int frames, int length)
    {
        int total = _size + (_hop * (frames - 1));
        var signal = new double[total];
        var envelope = new double[total];
        double scale = Math.Sqrt(_size);
        for (int f = 0; f < frames; f++)
        {
            // The conjugate trick: the inverse transform of a Hermitian spectrum is the conjugate of the forward transform of its conjugate, divided by N.
            for (int k = 0; k < _size; k++)
            {
                int bin = k <= _size / 2 ? k : _size - k;
                double re = real[(bin * frames) + f] * scale;
                double im = imaginary[(bin * frames) + f] * scale;
                if (bin == 0 || bin == _size / 2)
                {
                    im = 0;
                }

                _real[k] = re;
                _imaginary[k] = k <= _size / 2 ? -im : im;
            }

            Transform(_real, _imaginary);
            for (int i = 0; i < _size; i++)
            {
                signal[(f * _hop) + i] += _real[i] / _size * _window[i];
                envelope[(f * _hop) + i] += _window[i] * _window[i];
            }
        }

        var result = new float[length];
        for (int i = 0; i < length; i++)
        {
            int n = (_size / 2) + i;
            result[i] = (float)(signal[n] / envelope[n]);
        }

        return result;
    }

    // In-place radix-2 transform of (_real, _imaginary) arrays given as arguments.
    private void Transform(double[] re, double[] im)
    {
        int n = _size;
        for (int i = 0; i < n; i++)
        {
            int j = _reverse[i];
            if (j > i)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }

        for (int len = 2; len <= n; len <<= 1)
        {
            int half = len / 2;
            int step = n / len;
            for (int start = 0; start < n; start += len)
            {
                for (int k = 0; k < half; k++)
                {
                    double wr = _cos[k * step];
                    double wi = _sin[k * step];
                    int a = start + k;
                    int b = a + half;
                    double tr = (re[b] * wr) - (im[b] * wi);
                    double ti = (re[b] * wi) + (im[b] * wr);
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
        }
    }
}
