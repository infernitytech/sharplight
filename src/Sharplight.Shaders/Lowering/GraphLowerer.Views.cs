using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    // ---- views -------------------------------------------------------------------------------
    /// <summary>A reshape of strided data that cannot be expressed as a view (e.g. after a permute) is materialized with a copy.</summary>
    private void LowerReshape(Node n)
    {
        var src = Ref(n.Inputs[0]);
        try
        {
            _refs[n.Output] = ViewReshape(src, BoundShape(n.Output.Shape), n.Output.Shape.Symbol);
        }
        catch (NotSupportedException)
        {
            // The output becomes its own root (its memory is not tracked by the view-root liveness of the input; it is
            // simply never reused, which only costs arena space).
            var dst = Produce(n.Output);
            EmitCopy(src, dst with { Shape = src.Shape, Strides = TensorRef.DenseStrides(src.Shape), Symbol = src.Symbol }, "reshape-copy", [n]);
        }
    }

    private static TensorRef ViewSlice(TensorRef r, Attributes a)
    {
        int axis = a.Get<int>("axis");
        if (axis < 0)
        {
            axis += r.Rank;
        }

        long start = a.Get<long>("start");
        long len = a.Get<long>("length");
        var shape = (long[])r.Shape.Clone();
        shape[axis] = len;
        return r with { ByteOffset = r.ByteOffset + start * r.Strides[axis] * r.Type.ElementSize, Shape = shape };
    }

    private static TensorRef ViewPermute(TensorRef r, int[] perm)
    {
        if (r.Symbol is not null && perm[0] != 0)
        {
            throw new NotSupportedException("Permute must keep the symbolic axis outermost.");
        }

        return r with { Shape = [.. perm.Select(p => r.Shape[p])], Strides = [.. perm.Select(p => r.Strides[p])] };
    }

    /// <summary>Stride-compatible reshape (PyTorch view semantics).</summary>
    private static TensorRef ViewReshape(TensorRef r, long[] newShape, string? symbol)
    {
        var oldShape = r.Shape;
        var oldStride = r.Strides;
        var newStride = new long[newShape.Length];
        int viewD = newShape.Length - 1;
        long chunkBase = oldStride.Length > 0 ? oldStride[^1] : 1;
        long tensorNumel = 1;
        long viewNumel = 1;
        for (int td = oldShape.Length - 1; td >= 0; td--)
        {
            tensorNumel *= oldShape[td];
            if (td == 0 || (oldShape[td - 1] != 1 && oldStride[td - 1] != tensorNumel * chunkBase))
            {
                while (viewD >= 0 && (viewNumel < tensorNumel || newShape[viewD] == 1))
                {
                    newStride[viewD] = viewNumel * chunkBase;
                    viewNumel *= newShape[viewD];
                    viewD--;
                }
                if (viewNumel != tensorNumel)
                {
                    throw new NotSupportedException($"Reshape [{string.Join(",", oldShape)}] → [{string.Join(",", newShape)}] is not expressible as a view of strides [{string.Join(",", oldStride)}].");
                }

                if (td > 0)
                {
                    chunkBase = oldStride[td - 1];
                    tensorNumel = 1;
                    viewNumel = 1;
                }
            }
        }
        if (viewD != -1)
        {
            throw new NotSupportedException("Reshape: incompatible shapes.");
        }

        return r with { Shape = newShape, Strides = newStride, Symbol = symbol };
    }

    /// <summary>Rows view of a tensor: [rows = outer*mid, inner] with uniform row stride, inner contiguous.</summary>
    private static bool TryRows(TensorRef r, out long rowStride, out long rowsPerOuter)
    {
        rowStride = 0;
        rowsPerOuter = 1;
        if (r.Rank < 2 || r.Strides[^1] != 1)
        {
            return false;
        }

        for (int k = 1; k <= r.Rank - 2; k++)
        {
            if (r.Strides[k - 1] != r.Strides[k] * r.Shape[k])
            {
                return false;
            }

            rowsPerOuter *= r.Shape[k];
        }
        rowStride = r.Strides[r.Rank - 2];
        return true;
    }

    /// <summary>[T, R1, D] view with strides (S0, S1) for row kernels; middle dims must be collapsible.</summary>
    private static bool TryRows3(TensorRef r, out long r1, out long s0, out long s1)
    {
        r1 = 1;
        s0 = 0;
        s1 = 0;
        if (r.Strides[^1] != 1)
        {
            return false;
        }

        if (r.Rank == 1)
        {
            s0 = r.Shape[0];
            return true;
        }
        if (r.Rank == 2)
        {
            s0 = r.Strides[0];
            s1 = r.Shape[1];
            return true;
        }
        for (int k = 2; k <= r.Rank - 2; k++)
        {
            if (r.Shape[k] != 1 && r.Strides[k - 1] != r.Strides[k] * r.Shape[k])
            {
                return false;
            }
        }

        for (int k = 1; k <= r.Rank - 2; k++)
        {
            r1 *= r.Shape[k];
        }

        s0 = r.Strides[0];
        s1 = r.Strides[r.Rank - 2];
        return true;
    }

    /// <summary>
    /// Returns <paramref name="x"/> if <see cref="TryRows3"/> accepts it, otherwise a dense copy (named after
    /// <paramref name="why"/>) that it accepts.
    /// </summary>
    private TensorRef PrepareRows3(TensorRef x, string why, out long r1, out long s0, out long s1)
    {
        if (!TryRows3(x, out r1, out s0, out s1))
        {
            x = DenseTemp(x, why);
            TryRows3(x, out r1, out s0, out s1);
        }

        return x;
    }

    /// <summary>Returns <paramref name="r"/> if it is dense, otherwise a dense copy.</summary>
    private TensorRef EnsureDense(TensorRef r, string why) => r.IsDense ? r : DenseTemp(r, why);

    /// <summary>Returns <paramref name="r"/> if its stride along <paramref name="axis"/> is one, otherwise a dense copy.</summary>
    private TensorRef EnsureUnitInnerStride(TensorRef r, int axis, string why) => r.Strides[axis] == 1 ? r : DenseTemp(r, why);

    /// <summary>Scratch tensor in the arena, released after the current node is lowered.</summary>
    private TensorRef Temp(DataType type, long[] shape, string? symbol)
    {
        var key = new object();
        long bytes = type.ByteSize(ElementwiseCodegen.ElementCount(shape));
        long off = _alloc.Allocate(key, Math.Max(16, bytes));
        _temps.Add(key);
        return new TensorRef(_arena, off, type, shape, TensorRef.DenseStrides(shape), symbol);
    }

    /// <summary>Returns <paramref name="r"/> if it is f32 (strides preserved), else a dense f32 copy.</summary>
    private TensorRef AsF32(TensorRef r, string why)
    {
        if (r.Type == DataType.F32)
        {
            return r;
        }

        var dst = Temp(DataType.F32, r.Shape, r.Symbol);
        EmitCopy(r, dst, "to_f32:" + why);
        return dst;
    }

    /// <summary>Dense f32 tensor (copying/converting when needed).</summary>
    private TensorRef AsDenseF32(TensorRef r, string why)
    {
        if (r.Type == DataType.F32 && r.IsDense)
        {
            return r;
        }

        var dst = Temp(DataType.F32, r.Shape, r.Symbol);
        EmitCopy(r, dst, "dense_f32:" + why);
        return dst;
    }

    /// <summary>Copies a strided tensor into a fresh dense temporary in the arena.</summary>
    private TensorRef DenseTemp(TensorRef src, string why)
    {
        var dst = Temp(src.Type, src.Shape, src.Symbol);
        EmitCopy(src, dst, "densify:" + why);
        return dst;
    }

    // ---- concat ------------------------------------------------------------------------------
    /// <summary>
    /// Concatenation along a symbolic axis 0 (e.g. the detections of several NonMaxSuppression ops): each input is copied at the
    /// run-time sum of the extents before it, a push constant evaluated from the bindings.
    /// </summary>
    private void LowerConcatSymbolic(Node n, TensorRef y)
    {
        Dim offset = 0;
        foreach (var input in n.Inputs)
        {
            var src = Ref(input);
            string type = RawType(src.Type);
            var coords = Enumerable.Range(0, src.Rank).Select(Dn).ToArray();
            var scalars = new List<(string, KernelArg)>();
            string start = $"{offset.Fixed}u";
            if (offset.IsSymbolic)
            {
                scalars.Add(("offset", new SymbolArg(offset.Symbol!, 1)));
                start = "p.offset";
            }

            var target = coords.Select((c, k) => k == 0 ? $"{c} + {start}" : c).ToArray();
            EmitGen($"concat {n.Output.Name}", [n], src.Shape, src.Symbol, [src], [y], $"  p.out0[{Lin(y, target)}] = p.in0[{Lin(src, coords)}];",
                [type], [type], scalars: scalars);
            offset += input.Shape[0];
        }
    }

    private void LowerConcat(Node n)
    {
        var y = Produce(n.Output);
        int axis = n.Attributes.Get<int>("axis");
        if (axis < 0)
        {
            axis += y.Rank;
        }

        if (axis == 0 && n.Output.Shape[0].IsSymbolic && n.Inputs.Any(i => i.Shape[0] != n.Output.Shape[0]))
        {
            LowerConcatSymbolic(n, y);
            return;
        }

        long start = 0;
        foreach (var i in n.Inputs)
        {
            var src = Ref(i);
            var shape = (long[])y.Shape.Clone();
            shape[axis] = src.Shape[axis];
            var dst = y with { ByteOffset = y.ByteOffset + start * y.Strides[axis] * y.Type.ElementSize, Shape = shape };
            EmitCopy(src, dst, $"concat {n.Output.Name}", [n]);
            start += src.Shape[axis];
        }
    }
}
