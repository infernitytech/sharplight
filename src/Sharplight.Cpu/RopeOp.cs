using System.Runtime.CompilerServices;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Rotary embedding (plain, neox, gptj, multi-axis, interleaved multi-axis and multi-axis with per-element sections).</summary>
internal sealed unsafe class RopeOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _pos;
    private readonly TRef _y;
    private readonly TRef? _ff;
    private readonly int _dims;
    private readonly int _d;
    private readonly int _h;
    private readonly int _axes;
    private readonly bool _neox;
    private readonly float _freqScale;
    private readonly float _attnFactor;
    private readonly double[] _invFreq;   // base^(-2i/dims) * freq_scale
    private readonly int[] _axisOfFirst;    // per pair: the position axis of its first element
    private readonly int[] _axisOfSecond;   // per pair: the position axis of its second element (differs only with per-element sections)
    private readonly Action<int> _body;
    private long _t;
    private long _per;
    public override string Name => "Rope";

    public RopeOp(TRef x, TRef pos, TRef? ff, TRef y, int dims, float @base, string style, int[] sections, bool interleaved, bool perElement, float freqScale,
        float attnFactor)
    {
        _x = x;
        _pos = pos;
        _ff = ff;
        _y = y;
        _d = (int)x.Dims[2];
        _h = (int)x.Dims[1];
        _dims = dims <= 0 ? _d : dims;
        if (_dims > _d || (_dims & 1) != 0)
        {
            throw new ArgumentException($"Rope dims {_dims} invalid for head size {_d}.");
        }

        _neox = !string.Equals(style, "gptj", StringComparison.OrdinalIgnoreCase) && !string.Equals(style, "normal", StringComparison.OrdinalIgnoreCase);
        _freqScale = freqScale;
        _attnFactor = attnFactor;
        _axes = pos.Rank >= 2 ? (int)pos.Dims[1] : 1;
        int half = _dims / 2;
        _invFreq = new double[half];
        for (int i = 0; i < half; i++)
        {
            _invFreq[i] = Math.Pow(@base, -2.0 * i / _dims) * freqScale;
        }

        // Sections index the rotation pairs, or with per-element sections the rotated elements two by two (element e takes the axis of
        // pair section e / 2), so that the two elements of a neox pair (i, i + dims/2) can follow different axes (XD-RoPE).
        _axisOfFirst = new int[half];
        _axisOfSecond = new int[half];
        for (int i = 0; i < half; i++)
        {
            int first = _neox ? i : 2 * i;
            int second = _neox ? i + half : 2 * i + 1;
            _axisOfFirst[i] = AxisOf(perElement ? first / 2 : i, sections, interleaved);
            _axisOfSecond[i] = AxisOf(perElement ? second / 2 : i, sections, interleaved);
        }

        _body = Body;
    }

    /// <summary>The position axis of pair section index <paramref name="i"/> (llama.cpp's M-RoPE sector rules).</summary>
    private static int AxisOf(int i, int[] sections, bool interleaved)
    {
        bool multi = sections is { Length: 4 } && (sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
        if (!multi)
        {
            return 0;
        }

        int sectDims = sections[0] + sections[1] + sections[2] + sections[3];
        int sector = i % sectDims;
        int secW = sections[0] + sections[1];
        if (interleaved)
        {
            if (sector % 3 == 1 && sector < 3 * sections[1])
            {
                return 1;
            }

            if (sector % 3 == 2 && sector < 3 * sections[2])
            {
                return 2;
            }

            return sector % 3 == 0 && sector < 3 * sections[0] ? 0 : 3;
        }

        if (sector < sections[0])
        {
            return 0;
        }

        if (sector < secW)
        {
            return 1;
        }

        return sector < secW + sections[2] ? 2 : 3;
    }

    public override void Run(RunParameters p)
    {
        _t = _x.Dim(0, p);
        if (_t == 0)
        {
            return;
        }

        int chunks = (int)Math.Min(_t, CpuParallel.Chunks(_t * _h * _d, 16384));
        _per = (_t + chunks - 1) / chunks;
        CpuParallel.For(chunks, _body);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Body(int c)
    {
        long s = c * _per;
        long e = Math.Min(_t, s + _per);
        int half = _dims / 2;
        float* cs = stackalloc float[half * 4];   // per pair: cos, sin of its first and of its second element
        double* pp = stackalloc double[4];
        for (long t = s; t < e; t++)
        {
            int* pos = (int*)_pos.P + t * _pos.Strides[0];
            long ps1 = _pos.Rank >= 2 ? _pos.Strides[1] : 1;
            for (int a = 0; a < 4; a++)
            {
                pp[a] = a < _axes ? pos[a * ps1] : (_axes == 1 ? pos[0] : 0);
            }

            for (int i = 0; i < half; i++)
            {
                double theta = pp[_axisOfFirst[i]] * _invFreq[i];
                double theta2 = pp[_axisOfSecond[i]] * _invFreq[i];
                if (_ff != null)
                {
                    theta /= ((float*)_ff.P)[i];
                    theta2 /= ((float*)_ff.P)[i];
                }

                cs[4 * i] = (float)Math.Cos(theta) * _attnFactor;
                cs[4 * i + 1] = (float)Math.Sin(theta) * _attnFactor;
                cs[4 * i + 2] = (float)Math.Cos(theta2) * _attnFactor;
                cs[4 * i + 3] = (float)Math.Sin(theta2) * _attnFactor;
            }
            for (int h = 0; h < _h; h++)
            {
                float* x = _x.F(t * _h + h);
                float* y = (float*)_y.P + (t * _h + h) * _d;
                if (_neox)
                {
                    for (int i = 0; i < half; i++)
                    {
                        float x0 = x[i];
                        float x1 = x[i + half];
                        y[i] = x0 * cs[4 * i] - x1 * cs[4 * i + 1];
                        y[i + half] = x0 * cs[4 * i + 3] + x1 * cs[4 * i + 2];
                    }
                }
                else
                {
                    for (int i = 0; i < half; i++)
                    {
                        float x0 = x[2 * i];
                        float x1 = x[2 * i + 1];
                        y[2 * i] = x0 * cs[4 * i] - x1 * cs[4 * i + 1];
                        y[2 * i + 1] = x0 * cs[4 * i + 3] + x1 * cs[4 * i + 2];
                    }
                }
                for (int i = _dims; i < _d; i++)
                {
                    y[i] = x[i];
                }
            }
        }
    }
}
