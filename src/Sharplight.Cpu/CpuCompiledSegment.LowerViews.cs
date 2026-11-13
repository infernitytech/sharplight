using Sharplight.Graph;

namespace Sharplight.Cpu;

// Layout ops: views that alias their input, and copies that realize a layout change.
internal sealed unsafe partial class CpuCompiledSegment
{
    private void LowerReshape(Node n)
    {
        _refs[n.Output] = ReshapeView(Ref(n.Inputs[0]), n.Output);
    }

    private void LowerSlice(Node n)
    {
        var a = n.Attributes;
        var source = Ref(n.Inputs[0]);
        int axis = a.Get<int>("axis");
        if (axis < 0)
        {
            axis += source.Rank;
        }

        long start = a.Get<long>("start");
        var dims = (long[])source.Dims.Clone();
        dims[axis] = a.Get<long>("length");
        _refs[n.Output] = View(source, dims, source.Strides, start * source.Strides[axis], source.Sym);
    }

    private void LowerPermute(Node n)
    {
        var source = Ref(n.Inputs[0]);
        var perm = n.Attributes.Get<int[]>("perm");
        var dims = perm.Select(p => source.Dims[p]).ToArray();
        var strides = perm.Select(p => source.Strides[p]).ToArray();
        _refs[n.Output] = View(source, dims, strides, 0, perm[0] == 0 ? source.Sym : null);
    }

    /// <summary>Broadcast copy: a stride-0 view of the input copied into a dense output.</summary>
    private void LowerExpand(Node n)
    {
        var x = Ref(n.Inputs[0]);
        var y = NewDense(n.Output);
        int rank = y.Rank;
        var strides = new long[rank];
        for (int k = 0; k < rank; k++)
        {
            int inputAxis = k - (rank - x.Rank);
            bool broadcast = inputAxis >= 0
                && x.Dims[inputAxis] == 1
                && !(inputAxis == 0 && x.Sym is not null)
                && y.Dims[k] != 1;
            strides[k] = inputAxis < 0 || broadcast ? 0 : x.Strides[inputAxis];
        }

        var view = View(x, (long[])y.Dims.Clone(), strides, 0, y.Sym);
        AddStep(new CopyOp(view, y), x, y);
        _refs[n.Output] = y;
    }

    /// <summary>Tile as a copy: each tiled axis becomes (repeat, size) with a stride-0 repeat axis in the input view.</summary>
    private void LowerTile(Node n)
    {
        var x = Ref(n.Inputs[0]);
        var y = NewDense(n.Output);
        var repeats = n.Attributes.Get<long[]>("repeats");
        var viewDims = new List<long>();
        var viewStrides = new List<long>();
        var outDims = new List<long>();
        for (int k = 0; k < x.Rank; k++)
        {
            if (k == 0 && x.Sym is not null)
            {
                viewDims.Add(x.Dims[0]);
                viewStrides.Add(x.Strides[0]);
                outDims.Add(y.Dims[0]);
                continue;
            }

            viewDims.Add(repeats[k]);
            viewStrides.Add(0);
            outDims.Add(repeats[k]);
            viewDims.Add(x.Dims[k]);
            viewStrides.Add(x.Strides[k]);
            outDims.Add(x.Dims[k]);
        }

        var view = View(x, viewDims.ToArray(), viewStrides.ToArray(), 0, x.Sym);
        var outView = View(y, outDims.ToArray(), TRef.RowMajor(outDims.ToArray()), 0, y.Sym);
        AddStep(new CopyOp(view, outView), x, y);
        _refs[n.Output] = y;
    }

    private void LowerConcat(Node n)
    {
        int axis = n.Attributes.Get<int>("axis");
        if (axis < 0)
        {
            axis += n.Output.Rank;
        }

        var xs = n.Inputs.Select(Ref).ToArray();
        var y = NewDense(n.Output);
        AddStep(new ConcatOp(xs, y, axis), [.. xs, y]);
        _refs[n.Output] = y;
    }

    private void LowerCast(Node n)
    {
        var x = RowContig(Ref(n.Inputs[0]));
        var y = NewDense(n.Output);
        AddStep(new CastOp(x, y), x, y);
        _refs[n.Output] = y;
    }

    /// <summary>Contiguous and Transfer: a plain copy into a dense buffer.</summary>
    private void LowerCopy(Node n)
    {
        var x = Ref(n.Inputs[0]);
        var y = NewDense(n.Output);
        AddStep(new CopyOp(x, y), x, y);
        _refs[n.Output] = y;
    }
}
