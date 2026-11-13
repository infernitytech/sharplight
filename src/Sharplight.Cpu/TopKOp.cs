using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Top-k values and indices along one axis.</summary>
internal sealed unsafe class TopKOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _v;
    private readonly TRef _i;
    private readonly int _axis;
    private readonly int _k;
    private readonly bool _largest;
    private readonly Action<int> _body;
    private long _outer;
    private long _a;
    private long _inner;
    private long _per;
    public override string Name => "TopK";
    public TopKOp(TRef x, TRef vals, TRef idx, int axis, int k, bool largest)
    {
        _x = x;
        _v = vals;
        _i = idx;
        _axis = axis;
        _k = k;
        _largest = largest;
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        var d = OpHelpers.Dims(_x, p);
        _outer = OpHelpers.Prod(d, 0, _axis);
        _a = d[_axis];
        _inner = OpHelpers.Prod(d, _axis + 1, d.Length);
        long items = _outer * _inner;
        if (items == 0 || _k == 0)
        {
            return;
        }

        var (ch, per) = OpHelpers.Split(items, _a * 4);
        _per = per;
        CpuParallel.For(ch, _body);
    }

    private void Body(int chunk)
    {
        long s = chunk * _per;
        long e = Math.Min(_outer * _inner, s + _per);
        float* x = (float*)_x.P;
        float* ov = (float*)_v.P;
        int* oi = (int*)_i.P;
        var ids = new int[_a];
        for (long it = s; it < e; it++)
        {
            long o = it / _inner;
            long i = it % _inner;
            float* b = x + o * _a * _inner + i;
            for (int a = 0; a < _a; a++)
            {
                ids[a] = a;
            }

            bool largest = _largest;
            // total order: value (desc for largest, asc for smallest; NaN counts as the extreme), then lower index first
            int Cmp(int p, int q)
            {
                float u = b[p * _inner];
                float v = b[q * _inner];
                int c;
                if (u == v || (float.IsNaN(u) && float.IsNaN(v)))
                {
                    c = 0;
                }
                else if (float.IsNaN(u))
                {
                    c = -1;
                }
                else if (float.IsNaN(v))
                {
                    c = 1;
                }
                else
                {
                    c = largest ? (u > v ? -1 : 1) : (u < v ? -1 : 1);
                }

                if (!largest && c != 0 && (float.IsNaN(u) || float.IsNaN(v)))
                {
                    c = -c;
                }

                return c != 0 ? c : p.CompareTo(q);
            }
            if (_k >= _a)
            {
                Array.Sort(ids, Cmp);
            }
            else
            {
                // partial selection: keep the k best in an insertion-sorted prefix
                int k = _k;
                var top = new int[k];
                int cnt = 0;
                for (int a = 0; a < _a; a++)
                {
                    if (cnt == k && Cmp(a, top[k - 1]) >= 0)
                    {
                        continue;
                    }

                    int pos = cnt < k ? cnt++ : k - 1;
                    while (pos > 0 && Cmp(a, top[pos - 1]) < 0)
                    {
                        top[pos] = top[pos - 1];
                        pos--;
                    }
                    top[pos] = a;
                }
                for (int j = 0; j < k; j++)
                {
                    ids[j] = top[j];
                }
            }
            long ob = o * _k * _inner + i;
            for (int j = 0; j < _k; j++)
            {
                ov[ob + j * _inner] = b[ids[j] * _inner];
                oi[ob + j * _inner] = ids[j];
            }
        }
    }
}
