using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Concatenation along one axis into a dense output; along the outer axis the offsets follow the inputs' run-time extents.</summary>
internal sealed unsafe class ConcatOp : CpuOp
{
    private readonly TRef[] _xs;
    private readonly TRef _y;
    private readonly int _axis;
    private readonly long[] _dstStrides;
    private readonly long[] _offsets;
    public override string Name => "Concat";

    public ConcatOp(TRef[] xs, TRef y, int axis)
    {
        _xs = xs;
        _y = y;
        _axis = axis;
        _dstStrides = TRef.RowMajor(y.Dims);
        _offsets = new long[xs.Length];
        long o = 0;
        for (int i = 0; i < xs.Length; i++)
        {
            _offsets[i] = o;
            o += xs[i].Dims[axis];
        }
    }

    public override void Run(RunParameters p)
    {
        int es = _y.ElemSize;
        long outer = 0;
        for (int i = 0; i < _xs.Length; i++)
        {
            var x = _xs[i];
            long rows = x.Dim(0, p);
            long offset = _axis == 0 ? outer : _offsets[i];
            Mem.CopyStrided(x.P, x.Strides, _y.P + offset * _dstStrides[_axis] * es, _dstStrides, x.Dims, rows, es);
            outer += rows;
        }
    }
}
