using System.Runtime.Intrinsics;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Broadcasting binary elementwise op over f32 (comparisons and logical ops produce 0/1 values).</summary>
internal sealed unsafe class BinaryOp : CpuOp
{
    private readonly BinaryKind _kind;
    private readonly bool _fmod;
    private readonly TRef _a;
    private readonly TRef _b;
    private readonly TRef _y;
    private readonly long[] _sa, _sb; // broadcast strides aligned to output rank
    private readonly Action<int> _body;
    private RunParameters _p = null!;
    private long _outer;
    private long _perChunk;
    private long _l;
    public override string Name => "Binary." + _kind;

    public BinaryOp(BinaryKind kind, TRef a, TRef b, TRef y, bool fmod = false)
    {
        _fmod = fmod;
        _kind = kind;
        _a = a;
        _b = b;
        _y = y;
        _body = Body;
        _sa = Align(a, y);
        _sb = Align(b, y);
    }

    private static long[] Align(TRef t, TRef y)
    {
        int r = y.Rank;
        var s = new long[r];
        for (int k = 0; k < r; k++)
        {
            int ik = k - (r - t.Rank);
            if (ik < 0)
            {
                s[k] = 0;
            }
            else if (t.Dims[ik] == 1 && !(ik == 0 && t.Sym is not null))
            {
                s[k] = 0;
            }
            else
            {
                s[k] = t.Strides[ik];
            }
        }
        return s;
    }

    public override void Run(RunParameters p)
    {
        _p = p;
        long total = _y.Elements(p);
        if (total == 0)
        {
            return;
        }

        _l = _y.Cols(p);
        _outer = total / _l;
        int chunks = CpuParallel.Chunks(total, 32768);
        chunks = (int)Math.Min(chunks, _outer);
        _perChunk = (_outer + chunks - 1) / chunks;
        CpuParallel.For(chunks, _body);
    }

    private void Body(int c)
    {
        long s = c * _perChunk;
        long e = Math.Min(_outer, s + _perChunk);
        int rank = _y.Rank;
        long l = _l;
        long sa = _sa[rank - 1];
        long sb = _sb[rank - 1];
        for (long r = s; r < e; r++)
        {
            long ao = 0;
            long bo = 0;
            long rr = r;
            for (int k = rank - 2; k >= 1; k--)
            {
                long d = _y.Dims[k];
                long q = rr / d;
                long ix = rr - q * d;
                rr = q;
                ao += ix * _sa[k];
                bo += ix * _sb[k];
            }
            if (rank >= 2)
            {
                ao += rr * _sa[0];
                bo += rr * _sb[0];
            }
            float* pa = (float*)_a.P + ao;
            float* pb = (float*)_b.P + bo;
            float* py = (float*)_y.P + r * l;
            Apply(_kind, pa, sa, pb, sb, py, l, _fmod);
        }
    }

    private static void Apply(BinaryKind k, float* a, long sa, float* b, long sb, float* y, long n, bool fmod)
    {
        long i = 0;
        if (k >= BinaryKind.Mod)
        {
            for (; i < n; i++)
            {
                y[i] = OpS2(k, a[i * sa], b[i * sb], fmod);
            }

            return;
        }
        if (k == BinaryKind.Pow)
        {
            if (sb == 0)
            {
                float e = *b;
                if (e == 2f)
                {
                    for (; i < n; i++)
                    {
                        float v = a[i * sa];
                        y[i] = v * v;
                    }
                }
                else if (e == 1f) { for (; i < n; i++)
                    {
                        y[i] = a[i * sa];
                    }
                }
                else if (e == 0.5f) { for (; i < n; i++)
                    {
                        y[i] = MathF.Sqrt(a[i * sa]);
                    }
                }
                else if (e == 3f)
                {
                    for (; i < n; i++)
                    {
                        float v = a[i * sa];
                        y[i] = v * v * v;
                    }
                }
                else
                {
                    for (; i < n; i++)
                    {
                        y[i] = MathF.Pow(a[i * sa], e);
                    }
                }
            }
            else
            {
                for (; i < n; i++)
                {
                    y[i] = MathF.Pow(a[i * sa], b[i * sb]);
                }
            }

            return;
        }
        if (Vector256.IsHardwareAccelerated && (sa == 1 || sa == 0) && (sb == 1 || sb == 0))
        {
            if (sa == 1 && sb == 1)
            {
                for (; i + 8 <= n; i += 8)
                {
                    Op(k, Vector256.Load(a + i), Vector256.Load(b + i)).Store(y + i);
                }
            }
            else if (sa == 1)
            {
                var bv = Vector256.Create(*b);
                for (; i + 8 <= n; i += 8)
                {
                    Op(k, Vector256.Load(a + i), bv).Store(y + i);
                }
            }
            else if (sb == 1)
            {
                var av = Vector256.Create(*a);
                for (; i + 8 <= n; i += 8)
                {
                    Op(k, av, Vector256.Load(b + i)).Store(y + i);
                }
            }
            else
            {
                float v = OpS(k, *a, *b);
                for (; i < n; i++)
                {
                    y[i] = v;
                }
            }
        }
        for (; i < n; i++)
        {
            y[i] = OpS(k, a[i * sa], b[i * sb]);
        }
    }

    private static Vector256<float> Op(BinaryKind k, Vector256<float> a, Vector256<float> b) => k switch
    {
        BinaryKind.Add => a + b,
        BinaryKind.Sub => a - b,
        BinaryKind.Mul => a * b,
        BinaryKind.Div => a / b,
        BinaryKind.Max => Vector256.Max(a, b),
        _ => Vector256.Min(a, b),
    };

    private static float OpS2(BinaryKind k, float a, float b, bool fmod)
    {
        switch (k)
        {
            case BinaryKind.Mod:
            {
                float r = a % b; // C fmod (sign of dividend)
                if (!fmod && r != 0 && (r < 0) != (b < 0))
                    {
                        r += b;
                    }

                    return r;
            }
            case BinaryKind.PRelu:
                return a < 0 ? a * b : a;
            case BinaryKind.Equal:
                return a == b ? 1f : 0f;
            case BinaryKind.NotEqual:
                return a != b ? 1f : 0f;
            case BinaryKind.Less:
                return a < b ? 1f : 0f;
            case BinaryKind.LessEqual:
                return a <= b ? 1f : 0f;
            case BinaryKind.Greater:
                return a > b ? 1f : 0f;
            case BinaryKind.GreaterEqual:
                return a >= b ? 1f : 0f;
            case BinaryKind.And:
                return a != 0 && b != 0 ? 1f : 0f;
            case BinaryKind.Or:
                return a != 0 || b != 0 ? 1f : 0f;
            default:
                return (a != 0) != (b != 0) ? 1f : 0f;
        }
    }

    private static float OpS(BinaryKind k, float a, float b) => k switch
    {
        BinaryKind.Add => a + b,
        BinaryKind.Sub => a - b,
        BinaryKind.Mul => a * b,
        BinaryKind.Div => a / b,
        BinaryKind.Max => MathF.Max(a, b),
        BinaryKind.Pow => MathF.Pow(a, b),
        _ => MathF.Min(a, b),
    };
}
