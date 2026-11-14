using System.Numerics;

namespace Sharplight.Cpu;

/// <summary>
/// Complex FFT of one fixed length in double precision: iterative radix-2 for powers of two, Bluestein's chirp-z algorithm (three
/// power-of-two FFTs) for every other length. Immutable after construction; <see cref="Transform"/> is thread safe when every thread
/// passes its own scratch.
/// </summary>
internal sealed class FftPlan
{
    private readonly int _length;
    private readonly Complex[] _twiddles;       // radix-2: e^{-2πi k/n} for k < n/2 (n = _length or the Bluestein size)
    private readonly int _size;                 // power-of-two transform size
    private readonly Complex[]? _chirp;         // Bluestein: e^{-πi k²/L}
    private readonly Complex[]? _chirpSpectrum; // FFT of the conjugate chirp, wrapped to _size

    public FftPlan(int length)
    {
        _length = length;
        _size = BitOperations.IsPow2(length) ? length : (int)BitOperations.RoundUpToPowerOf2((uint)((2 * length) - 1));
        _twiddles = new Complex[Math.Max(1, _size / 2)];
        for (int k = 0; k < _twiddles.Length; k++)
        {
            _twiddles[k] = Complex.FromPolarCoordinates(1, -2 * Math.PI * k / _size);
        }

        if (_size == length)
        {
            return;
        }

        _chirp = new Complex[length];
        var b = new Complex[_size];
        for (int k = 0; k < length; k++)
        {
            // k² mod 2L keeps the angle exact for long transforms.
            long k2 = (long)k * k % (2L * length);
            _chirp[k] = Complex.FromPolarCoordinates(1, -Math.PI * k2 / length);
            b[k] = Complex.Conjugate(_chirp[k]);
            if (k > 0)
            {
                b[_size - k] = b[k];
            }
        }

        Radix2(b, inverse: false);
        _chirpSpectrum = b;
    }

    /// <summary>Scratch elements <see cref="Transform"/> needs.</summary>
    public int ScratchLength => _chirp is null ? 0 : _size;

    /// <summary>In-place unnormalized transform of <paramref name="data"/> (length L): sign -1 forward, +1 with <paramref name="inverse"/>.</summary>
    public void Transform(Span<Complex> data, bool inverse, Span<Complex> scratch)
    {
        if (_chirp is null)
        {
            Radix2(data, inverse);
            return;
        }

        // An inverse transform is the conjugate of the forward transform of the conjugate.
        var a = scratch[.._size];
        a.Clear();
        for (int k = 0; k < _length; k++)
        {
            var v = inverse ? Complex.Conjugate(data[k]) : data[k];
            a[k] = v * _chirp[k];
        }

        Radix2(a, inverse: false);
        for (int k = 0; k < _size; k++)
        {
            a[k] *= _chirpSpectrum![k];
        }

        Radix2(a, inverse: true);
        for (int k = 0; k < _length; k++)
        {
            var v = a[k] / _size * _chirp[k];
            data[k] = inverse ? Complex.Conjugate(v) : v;
        }
    }

    private void Radix2(Span<Complex> a, bool inverse)
    {
        int n = a.Length;
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1)
            {
                j ^= bit;
            }

            j ^= bit;
            if (i < j)
            {
                (a[i], a[j]) = (a[j], a[i]);
            }
        }

        for (int len = 2; len <= n; len <<= 1)
        {
            int half = len / 2;
            int stride = _size / len;
            for (int start = 0; start < n; start += len)
            {
                for (int k = 0; k < half; k++)
                {
                    var w = _twiddles[k * stride];
                    if (inverse)
                    {
                        w = Complex.Conjugate(w);
                    }

                    var u = a[start + k];
                    var v = a[start + k + half] * w;
                    a[start + k] = u + v;
                    a[start + k + half] = u - v;
                }
            }
        }
    }
}
