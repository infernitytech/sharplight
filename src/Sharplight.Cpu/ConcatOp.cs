using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// Concatenation along one axis into the output's layout (dense, or channels-last for a channel concat of channels-last
/// views); along the outer axis the offsets follow the inputs' run-time extents. Large copies run in parallel, split along
/// the outer axis or, when it is short, along the largest other axis.
/// </summary>
internal sealed unsafe class ConcatOp : CpuOp
{
    /// <summary>Elements per parallel task aimed for.</summary>
    private const long ElementsPerTask = 32768;

    private readonly TRef[] _xs;
    private readonly TRef _y;
    private readonly int _axis;
    private readonly long[] _offsets;
    private readonly Action<int> _body;

    // per-run state
    private readonly long[] _rows;
    private readonly long[] _outer;
    private readonly int[] _taskStart;
    private int _split;
    private long _perTask;

    public ConcatOp(TRef[] xs, TRef y, int axis)
    {
        _xs = xs;
        _y = y;
        _axis = axis;
        _offsets = new long[xs.Length];
        _rows = new long[xs.Length];
        _outer = new long[xs.Length];
        _taskStart = new int[xs.Length + 1];
        _body = Body;
        long o = 0;
        for (int i = 0; i < xs.Length; i++)
        {
            _offsets[i] = o;
            o += xs[i].Dims[axis];
        }
    }

    public override string Name => "Concat";

    public override void Run(RunParameters p)
    {
        long total = _y.Elements(p);
        long outer = 0;
        for (int i = 0; i < _xs.Length; i++)
        {
            _rows[i] = _xs[i].Dim(0, p);
            _outer[i] = outer;
            outer += _rows[i];
        }

        int rank = _y.Rank;
        if (total < 2 * ElementsPerTask || rank < 2)
        {
            _split = -1;
            for (int i = 0; i < _xs.Length; i++)
            {
                Copy(i, 0, 1);
            }

            return;
        }

        // Split axis: the outer one unless it is short, then the largest non-concat axis before the last.
        _split = 0;
        long splitDim = _y.Dim(0, p);
        for (int k = 1; k < rank - 1 && splitDim < 2 * CpuParallel.Threads; k++)
        {
            if (k != _axis && _y.Dims[k] > splitDim)
            {
                _split = k;
                splitDim = _y.Dims[k];
            }
        }

        if (_split == _axis || (_split == 0 && _axis == 0))
        {
            _split = -1;
            for (int i = 0; i < _xs.Length; i++)
            {
                Copy(i, 0, 1);
            }

            return;
        }

        long per = Math.Max(1, ElementsPerTask * splitDim / Math.Max(1, total));
        _perTask = per;
        int tasks = 0;
        for (int i = 0; i < _xs.Length; i++)
        {
            _taskStart[i] = tasks;
            long d = _split == 0 ? _rows[i] : _xs[i].Dims[_split];
            tasks += (int)((d + per - 1) / per);
        }

        _taskStart[_xs.Length] = tasks;
        CpuParallel.For(tasks, _body);
    }

    private void Body(int task)
    {
        int i = 0;
        while (_taskStart[i + 1] <= task)
        {
            i++;
        }

        long d = _split == 0 ? _rows[i] : _xs[i].Dims[_split];
        long j0 = (task - _taskStart[i]) * _perTask;
        long j1 = Math.Min(d, j0 + _perTask);
        Copy(i, j0, j1);
    }

    /// <summary>Copies input <paramref name="i"/>, along the split axis only indices [j0, j1) (all of it without a split).</summary>
    private void Copy(int i, long j0, long j1)
    {
        var x = _xs[i];
        int es = _y.ElemSize;
        long offset = _axis == 0 ? _outer[i] : _offsets[i];
        byte* src = x.P;
        byte* dst = _y.P + offset * _y.Strides[_axis] * es;
        long rows = _rows[i];
        if (_split < 0)
        {
            Mem.CopyStrided(src, x.Strides, dst, _y.Strides, x.Dims, rows, es);
            return;
        }

        src += j0 * x.Strides[_split] * es;
        dst += j0 * _y.Strides[_split] * es;
        if (_split == 0)
        {
            Mem.CopyStrided(src, x.Strides, dst, _y.Strides, x.Dims, j1 - j0, es);
            return;
        }

        var dims = (long[])x.Dims.Clone();
        dims[_split] = j1 - j0;
        Mem.CopyStrided(src, x.Strides, dst, _y.Strides, dims, rows, es);
    }
}
