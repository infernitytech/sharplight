using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Rotary embedding (plain, neox, gptj, multi-axis and interleaved multi-axis).</summary>
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
    private readonly bool _interleaved;
    private readonly int[] _sections;
    private readonly float _freqScale;
    private readonly float _attnFactor;
    private readonly double[] _invFreq;   // base^(-2i/dims) * freq_scale
    private readonly int[] _axisOfPair;
    private readonly Action<int> _body;
    private long _t;
    private long _per;
    public override string Name => "Rope";

    public RopeOp(TRef x, TRef pos, TRef? ff, TRef y, int dims, float @base, string style, int[] sections, bool interleaved, float freqScale, float attnFactor)
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
        _interleaved = interleaved;
        _sections = sections;
        _freqScale = freqScale;
        _attnFactor = attnFactor;
        _axes = pos.Rank >= 2 ? (int)pos.Dims[1] : 1;
        int half = _dims / 2;
        _invFreq = new double[half];
        for (int i = 0; i < half; i++)
        {
            _invFreq[i] = Math.Pow(@base, -2.0 * i / _dims) * freqScale;
        }

        _axisOfPair = new int[half];
        bool multi = sections is { Length: 4 } && (sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
        for (int i = 0; i < half; i++)
        {
            int axis = 0;
            if (multi)
            {
                int sectDims = sections[0] + sections[1] + sections[2] + sections[3];
                int sector = i % sectDims;
                int secW = sections[0] + sections[1];
                if (interleaved)
                {
                    if (sector % 3 == 1 && sector < 3 * sections[1])
                    {
                        axis = 1;
                    }
                    else if (sector % 3 == 2 && sector < 3 * sections[2])
                    {
                        axis = 2;
                    }
                    else if (sector % 3 == 0 && sector < 3 * sections[0])
                    {
                        axis = 0;
                    }
                    else
                    {
                        axis = 3;
                    }
                }
                else
                {
                    if (sector >= sections[0] && sector < secW)
                    {
                        axis = 1;
                    }
                    else if (sector >= secW && sector < secW + sections[2])
                    {
                        axis = 2;
                    }
                    else if (sector >= secW + sections[2])
                    {
                        axis = 3;
                    }
                }
            }
            _axisOfPair[i] = axis;
        }
        _body = Body;
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

    private void Body(int c)
    {
        long s = c * _per;
        long e = Math.Min(_t, s + _per);
        int half = _dims / 2;
        float* cs = stackalloc float[half * 2];
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
                double theta = pp[_axisOfPair[i]] * _invFreq[i];
                if (_ff != null)
                {
                    theta /= ((float*)_ff.P)[i];
                }

                cs[2 * i] = (float)Math.Cos(theta) * _attnFactor;
                cs[2 * i + 1] = (float)Math.Sin(theta) * _attnFactor;
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
                        y[i] = x0 * cs[2 * i] - x1 * cs[2 * i + 1];
                        y[i + half] = x0 * cs[2 * i + 1] + x1 * cs[2 * i];
                    }
                }
                else
                {
                    for (int i = 0; i < half; i++)
                    {
                        float x0 = x[2 * i];
                        float x1 = x[2 * i + 1];
                        y[2 * i] = x0 * cs[2 * i] - x1 * cs[2 * i + 1];
                        y[2 * i + 1] = x0 * cs[2 * i + 1] + x1 * cs[2 * i];
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
