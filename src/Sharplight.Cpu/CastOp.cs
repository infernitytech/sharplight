using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>Element type conversion between the plain numeric types (f32, f16, bf16, integers, bool).</summary>
internal sealed unsafe class CastOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly Action<int> _body;
    private long _rows;
    private long _cols;
    private long _per;
    public override string Name => $"Cast.{_x.Type}->{_y.Type}";
    public CastOp(TRef x, TRef y)
    {
        _x = x;
        _y = y;
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        _rows = _x.Rows(p);
        _cols = _x.Cols(p);
        if (_rows * _cols == 0)
        {
            return;
        }

        int chunks = (int)Math.Min(_rows, CpuParallel.Chunks(_rows * _cols, 32768));
        _per = (_rows + chunks - 1) / chunks;
        CpuParallel.For(chunks, _body);
    }

    private static float Load(DataType t, byte* p, long i)
    {
        if (t == DataType.F32)
        {
            return ((float*)p)[i];
        }

        if (t == DataType.F16)
        {
            return Simd.HalfToFloat(((ushort*)p)[i]);
        }

        if (t == DataType.BF16)
        {
            return Simd.BFloatToFloat(((ushort*)p)[i]);
        }

        if (t == DataType.I32)
        {
            return ((int*)p)[i];
        }

        if (t == DataType.I64)
        {
            return ((long*)p)[i];
        }

        if (t == DataType.I8)
        {
            return ((sbyte*)p)[i];
        }

        if (t == DataType.U8)
        {
            return p[i];
        }

        if (t == DataType.Bool)
        {
            return p[i] != 0 ? 1 : 0;
        }

        throw new NotSupportedException($"Cast from {t}");
    }

    private static void Store(DataType t, byte* p, long i, float v)
    {
        if (t == DataType.F32)
        {
            ((float*)p)[i] = v;
        }
        else if (t == DataType.F16)
        {
            ((ushort*)p)[i] = Simd.FloatToHalf(v);
        }
        else if (t == DataType.BF16)
        {
            ((ushort*)p)[i] = Simd.FloatToBFloat(v);
        }
        else if (t == DataType.I32)
        {
            ((int*)p)[i] = (int)v;
        }
        else if (t == DataType.I64)
        {
            ((long*)p)[i] = (long)v;
        }
        else if (t == DataType.I8)
        {
            ((sbyte*)p)[i] = (sbyte)v;
        }
        else if (t == DataType.U8)
        {
            p[i] = (byte)v;
        }
        else if (t == DataType.Bool)
        {
            p[i] = (byte)(v != 0 ? 1 : 0);
        }
        else
        {
            throw new NotSupportedException($"Cast to {t}");
        }
    }

    private void Body(int c)
    {
        long s = c * _per;
        long e = Math.Min(_rows, s + _per);
        var st = _x.Type;
        var dt = _y.Type;
        for (long r = s; r < e; r++)
        {
            byte* src = _x.RowPtr(r);
            byte* dst = _y.P + r * _cols * dt.BlockBytes;
            if (st == DataType.F32 && dt == DataType.F32)
            {
                Buffer.MemoryCopy(src, dst, _cols * 4, _cols * 4);
            }
            else if (st == DataType.I32 && dt == DataType.I32)
            {
                Buffer.MemoryCopy(src, dst, _cols * 4, _cols * 4);
            }
            else if (st == DataType.F16 && dt == DataType.F32)
            {
                Simd.HalfToFloat((ushort*)src, (float*)dst, _cols);
            }
            else if (st == DataType.BF16 && dt == DataType.F32)
            {
                Simd.BFloatToFloat((ushort*)src, (float*)dst, _cols);
            }
            else if (st == DataType.I32 && dt == DataType.I64)
            {
                for (long i = 0; i < _cols; i++)
                {
                    ((long*)dst)[i] = ((int*)src)[i];
                }
            }
            else if (st == DataType.I64 && dt == DataType.I32)
            {
                for (long i = 0; i < _cols; i++)
                {
                    ((int*)dst)[i] = (int)((long*)src)[i];
                }
            }
            else
            {
                for (long i = 0; i < _cols; i++)
                {
                    Store(dt, dst, i, Load(st, src, i));
                }
            }
        }
    }
}
