using System.Numerics;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>ScatterElements / ScatterND on f32, i32 or i64 data (sequential, so duplicate indices resolve in order).</summary>
internal sealed unsafe class ScatterOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _idx;
    private readonly TRef _upd;
    private readonly TRef _y;
    private readonly int _axis;
    private readonly bool _nd;
    private readonly int _red; // 0 none 1 add 2 mul 3 max 4 min
    public override string Name => _nd ? "ScatterND" : "ScatterElements";

    public ScatterOp(TRef x, TRef idx, TRef upd, TRef y, int axis, bool nd, string reduction)
    {
        _x = x;
        _idx = idx;
        _upd = upd;
        _y = y;
        _axis = axis;
        _nd = nd;
        _red = reduction switch { "add" => 1, "mul" => 2, "max" => 3, "min" => 4, _ => 0 };
    }

    public override void Run(RunParameters p)
    {
        long n = _x.Elements(p);
        Buffer.MemoryCopy(_x.P, _y.P, n * _x.ElemSize, n * _x.ElemSize);
        if (_x.Type == DataType.F32)
        {
            Go<float>(p);
        }
        else if (_x.Type == DataType.I32)
        {
            Go<int>(p);
        }
        else
        {
            Go<long>(p);
        }
    }

    private static T Red<T>(int red, T a, T b) where T : unmanaged, INumber<T> => red switch
    {
        1 => a + b, 2 => a * b, 3 => T.Max(a, b), 4 => T.Min(a, b), _ => b,
    };

    private void Go<T>(RunParameters p) where T : unmanaged, INumber<T>
    {
        var yd = OpHelpers.Dims(_x, p);
        var ys = OpHelpers.Strides(yd);
        T* y = (T*)_y.P;
        T* u = (T*)_upd.P;
        var id = OpHelpers.Dims(_idx, p);
        if (!_nd)
        {
            long total = OpHelpers.Prod(id, 0, id.Length);
            int r = id.Length;
            for (long l = 0; l < total; l++)
            {
                long rem = l;
                long off = 0;
                for (int d = r - 1; d >= 0; d--)
                {
                    long ix = rem % id[d];
                    rem /= id[d];
                    if (d == _axis)
                    {
                        ix = OpHelpers.Wrap(OpHelpers.Idx(_idx, l), yd[d], "ScatterElements");
                    }

                    off += ix * ys[d];
                }
                y[off] = Red(_red, y[off], u[l]);
            }
        }
        else
        {
            int q = (int)id[^1];
            long tuples = OpHelpers.Prod(id, 0, id.Length - 1);
            long slice = OpHelpers.Prod(yd, q, yd.Length);
            for (long t = 0; t < tuples; t++)
            {
                long off = 0;
                for (int j = 0; j < q; j++)
                {
                    off += OpHelpers.Wrap(OpHelpers.Idx(_idx, t * q + j), yd[j], "ScatterND") * ys[j];
                }

                for (long i = 0; i < slice; i++)
                {
                    y[off + i] = Red(_red, y[off + i], u[t * slice + i]);
                }
            }
        }
    }
}
