using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Constant / reflect / edge padding of an arbitrary-rank (strided) f32 tensor; negative pads crop.</summary>
internal sealed unsafe class PadOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly long[] _pads;
    private readonly int _mode;
    private readonly float _value; // 0 constant 1 reflect 2 edge
    private readonly Action<int> _body;
    private long[] _od = [];
    private long[] _id = [];
    private long _rows;
    private long _per;
    public override string Name => "Pad";

    public PadOp(TRef x, TRef y, long[] pads, string mode, float value)
    {
        _x = x;
        _y = y;
        _pads = pads;
        _value = value;
        _mode = mode switch { "reflect" => 1, "edge" => 2, _ => 0 };
        _body = Body;
    }

    /// <summary>Source index for output index <paramref name="o"/> of an axis (padded by <paramref name="pb"/>), or -1 for constant fill.</summary>
    private static long Map(long o, long pb, long n, int mode)
    {
        long i = o - pb;
        if (i >= 0 && i < n)
        {
            return i;
        }

        switch (mode)
        {
            case 0:
                return -1;
            case 2:
                return i < 0 ? 0 : n - 1;
            default:
                if (n == 1)
                {
                    return 0;
                }

                long period = 2 * (n - 1);
                i %= period;
                if (i < 0)
                {
                    i += period;
                }

                return i >= n ? period - i : i;
        }
    }

    public override void Run(RunParameters p)
    {
        int r = _x.Rank;
        _id = new long[r];
        _od = new long[r];
        for (int d = 0; d < r; d++)
        {
            _id[d] = _x.Dim(d, p);
            _od[d] = _id[d] + _pads[d] + _pads[d + r];
        }
        long total = 1;
        foreach (var d in _od)
        {
            total *= d;
        }

        if (total == 0)
        {
            return;
        }

        _rows = total / _od[r - 1];
        int chunks = (int)Math.Min(_rows, CpuParallel.Chunks(total, 16384));
        _per = (_rows + chunks - 1) / chunks;
        CpuParallel.For(chunks, _body);
    }

    private void Body(int chunk)
    {
        int r = _x.Rank;
        long s = chunk * _per;
        long e = Math.Min(_rows, s + _per);
        long lastN = _id[r - 1];
        long lastO = _od[r - 1];
        long pbl = _pads[r - 1];
        long sl = _x.Strides[r - 1];
        for (long row = s; row < e; row++)
        {
            long rem = row;
            long so = 0;
            bool fill = false;
            for (int d = r - 2; d >= 0; d--)
            {
                long ix = rem % _od[d];
                rem /= _od[d];
                long m = Map(ix, _pads[d], _id[d], _mode);
                if (m < 0)
                {
                    fill = true;
                    break;
                }
                so += m * _x.Strides[d];
            }
            float* y = (float*)_y.P + row * lastO;
            if (fill)
            {
                new Span<float>(y, (int)lastO).Fill(_value);
                continue;
            }
            float* x = (float*)_x.P + so;
            for (long j = 0; j < lastO; j++)
            {
                long m = Map(j, pbl, lastN, _mode);
                y[j] = m < 0 ? _value : x[m * sl];
            }
        }
    }
}
