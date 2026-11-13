using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>Rounds f32 values (or wraps/clamps i32 values) to the value set of a narrower numeric format (graph op Narrow).</summary>
internal sealed unsafe class NarrowOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly NumericFormat _format;
    private readonly bool _saturate;
    private readonly ExponentRounding _rounding;
    private readonly Action<int> _body;
    private long _rows;
    private long _cols;
    private long _per;

    public override string Name => $"Narrow.{NumericFormats.Name(_format)}";

    public NarrowOp(TRef x, TRef y, NumericFormat format, bool saturate, ExponentRounding rounding)
    {
        _x = x;
        _y = y;
        _format = format;
        _saturate = saturate;
        _rounding = rounding;
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

    private void Body(int chunk)
    {
        long start = chunk * _per;
        long end = Math.Min(_rows, start + _per);
        for (long r = start; r < end; r++)
        {
            byte* src = _x.RowPtr(r);
            byte* dst = _y.P + r * _cols * 4;
            if (_x.Type == DataType.I32)
            {
                NarrowIntegers((int*)src, (int*)dst);
            }
            else
            {
                NarrowFloats((float*)src, (float*)dst);
            }
        }
    }

    private void NarrowIntegers(int* x, int* y)
    {
        var (min, max) = NumericFormats.IntegerRange(_format);
        int bits = NumericFormats.Bits(_format);
        for (long i = 0; i < _cols; i++)
        {
            y[i] = (int)(_saturate ? Math.Clamp(x[i], min, max) : NumericFormats.Wrap(x[i], bits, min < 0));
        }
    }

    private void NarrowFloats(float* x, float* y)
    {
        switch (_format)
        {
            case NumericFormat.F16:
                for (long i = 0; i < _cols; i++)
                {
                    y[i] = (float)(Half)x[i];
                }

                break;
            case NumericFormat.BF16:
                for (long i = 0; i < _cols; i++)
                {
                    y[i] = Simd.BFloatToFloat(Simd.FloatToBFloat(x[i]));
                }

                break;
            default:
                for (long i = 0; i < _cols; i++)
                {
                    y[i] = (float)NumericFormats.Round(_format, x[i], _saturate, _rounding);
                }

                break;
        }
    }
}
