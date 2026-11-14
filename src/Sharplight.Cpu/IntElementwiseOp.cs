using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>
/// Broadcasting unary or binary elementwise op over i32 inputs with exact integer semantics (see StandardOps.Numeric): arithmetic and
/// bitwise ops return i32 (wrapping), comparisons return f32 0/1 masks.
/// </summary>
internal sealed unsafe class IntElementwiseOp : CpuOp
{
    private readonly IntOpKind _op;
    private readonly TRef _a;
    private readonly TRef? _b;
    private readonly TRef _y;
    private readonly bool _fmod;
    private readonly bool _unsigned;
    private readonly long[] _sa;
    private readonly long[] _sb;
    private readonly Action<int> _body;
    private long _outer;
    private long _per;
    private long _l;

    public override string Name => "Int." + _op;

    /// <param name="op">The operation.</param>
    /// <param name="a">First (or only) i32 input.</param>
    /// <param name="b">Second i32 input of a binary op.</param>
    /// <param name="y">Dense output: i32, or f32 for comparisons.</param>
    /// <param name="fmod">Mod: C-style remainder (sign of the dividend) instead of the sign of the divisor.</param>
    /// <param name="unsigned">ShiftRight: logical instead of arithmetic.</param>
    public IntElementwiseOp(IntOpKind op, TRef a, TRef? b, TRef y, bool fmod, bool unsigned)
    {
        _op = op;
        _a = a;
        _b = b;
        _y = y;
        _fmod = fmod;
        _unsigned = unsigned;
        _body = Body;
        _sa = OpHelpers.Align(a, y.Rank);
        _sb = b is null ? new long[y.Rank] : OpHelpers.Align(b, y.Rank);
    }

    public override void Run(RunParameters p)
    {
        long total = _y.Elements(p);
        if (total == 0)
        {
            return;
        }

        _l = _y.Cols(p);
        _outer = total / _l;
        var (chunks, per) = OpHelpers.Split(_outer, _l);
        _per = per;
        CpuParallel.For(chunks, _body);
    }

    private void Body(int chunk)
    {
        int rank = _y.Rank;
        bool mask = _y.Type == DataType.F32;
        int* a = (int*)_a.P;
        int* b = _b is null ? null : (int*)_b.P;
        long start = chunk * _per;
        long end = Math.Min(_outer, start + _per);
        for (long row = start; row < end; row++)
        {
            long oa = 0;
            long ob = 0;
            long rest = row;
            for (int k = rank - 2; k >= 0; k--)
            {
                long d = _y.Dims[k];
                long ix = rest % d;
                rest /= d;
                oa += ix * _sa[k];
                ob += ix * _sb[k];
            }

            long sa = rank == 0 ? 0 : _sa[rank - 1];
            long sb = rank == 0 ? 0 : _sb[rank - 1];
            byte* y = _y.P + row * _l * 4;
            for (long i = 0; i < _l; i++)
            {
                int x = a[oa + i * sa];
                int z = b is null ? 0 : b[ob + i * sb];
                if (mask)
                {
                    ((float*)y)[i] = Compare(_op, x, z) ? 1f : 0f;
                }
                else
                {
                    ((int*)y)[i] = Apply(_op, x, z, _fmod, _unsigned);
                }
            }
        }
    }

    private static bool Compare(IntOpKind op, int a, int b) => op switch
    {
        IntOpKind.Equal => a == b,
        IntOpKind.NotEqual => a != b,
        IntOpKind.Less => a < b,
        IntOpKind.LessEqual => a <= b,
        IntOpKind.Greater => a > b,
        IntOpKind.GreaterEqual => a >= b,
        _ => throw new NotSupportedException($"integer comparison {op}"),
    };

    /// <summary>One element of an integer op; division by zero yields 0 and int.MinValue / -1 wraps.</summary>
    internal static int Apply(IntOpKind op, int a, int b, bool fmod, bool unsigned) => op switch
    {
        IntOpKind.Add => unchecked(a + b),
        IntOpKind.Sub => unchecked(a - b),
        IntOpKind.Mul => unchecked(a * b),
        IntOpKind.Div => b == 0 ? 0 : b == -1 ? unchecked(-a) : a / b,
        IntOpKind.Mod => Mod(a, b, fmod),
        IntOpKind.Max => Math.Max(a, b),
        IntOpKind.Min => Math.Min(a, b),
        IntOpKind.Pow => Pow(a, b),
        IntOpKind.BitAnd => a & b,
        IntOpKind.BitOr => a | b,
        IntOpKind.BitXor => a ^ b,
        IntOpKind.ShiftLeft => (uint)b >= 32 ? 0 : a << b,
        IntOpKind.ShiftRight => (uint)b >= 32 ? (unsigned || a >= 0 ? 0 : -1) : unsigned ? (int)((uint)a >> b) : a >> b,
        IntOpKind.Neg => unchecked(-a),
        IntOpKind.Abs => a < 0 ? unchecked(-a) : a,
        IntOpKind.Sign => Math.Sign(a),
        IntOpKind.BitNot => ~a,
        _ => throw new NotSupportedException($"integer op {op}"),
    };

    private static int Mod(int a, int b, bool fmod)
    {
        if (b == 0 || b == -1)
        {
            return 0;
        }

        int r = a % b;
        return !fmod && r != 0 && (r < 0) != (b < 0) ? r + b : r;
    }

    private static int Pow(int a, int e)
    {
        if (e < 0)
        {
            return a == 1 ? 1 : a == -1 ? ((e & 1) == 0 ? 1 : -1) : 0;
        }

        int result = 1;
        int power = a;
        while (e > 0)
        {
            if ((e & 1) != 0)
            {
                result = unchecked(result * power);
            }

            power = unchecked(power * power);
            e >>= 1;
        }

        return result;
    }
}
