using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// Shape manipulation: reshape, squeeze, concat, slice, split, expand and friends.
internal sealed partial class OnnxTranslator
{
    // ================================ shape manipulation ===================================================
    /// <summary>ONNX Reshape semantics (0 = copy, -1 = infer; <see cref="OnnxConst.SymBatch"/> = batch size) on top of Sharplight's.</summary>
    private Value ReshapeOnnx(Value x, long[] req, bool allowZero, string? name = null)
    {
        bool srcSym = Sym(x);
        var r = new long[req.Length];
        for (int k = 0; k < req.Length; k++)
        {
            long v = req[k];
            if (v == OnnxConst.SymBatch)
            {
                if (!srcSym || k != 0)
                {
                    throw new NotSupportedException("the target shape mentions the symbolic batch size somewhere other than the outermost dim of a batched tensor");
                }

                r[k] = 0;
                continue;
            }
            if (v == 0 && !allowZero)
            {
                if (k >= x.Rank)
                {
                    throw new InvalidDataException("Reshape: 0 beyond the input rank.");
                }

                if (k == 0 && srcSym)
                {
                    r[0] = 0;
                    continue;
                }
                v = x.Shape[k].Fixed;
            }
            r[k] = v;
        }
        if (srcSym)
        {
            if (r.Length == 0)
            {
                throw new NotSupportedException("reshaping a batched tensor to a scalar");
            }

            if (r[0] == -1)
            {
                long rest = 1;
                for (int k = 1; k < r.Length; k++)
                {
                    if (r[k] == -1)
                    {
                        throw new NotSupportedException("two inferred dims");
                    }

                    rest *= r[k];
                }

                if (rest != x.Shape.InnerElements)
                {
                    throw new NotSupportedException($"reshape of batched {x.Shape} to [{string.Join(",", req)}] would merge the batch dimension into another dim");
                }

                r[0] = 0;
            }
            else if (r[0] != 0)
            {
                throw new NotSupportedException($"reshape of batched {x.Shape} fixes the batch dimension to {r[0]}");
            }
        }
        else if (req.Any(v => v == OnnxConst.SymBatch))
        {
            throw new NotSupportedException("the target shape mentions the symbolic batch size but the input is not batched");
        }

        return Emit("Reshape", [x], Attributes.From(("shape", r)), name);
    }

    private void Flatten(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        int axis = NormalizeAxis(n.Int("axis", 1), x.Rank);
        bool sym = Sym(x);
        if (sym && axis != 1)
        {
            throw new NotSupportedException($"Flatten(axis={axis}) of a batched tensor would merge or hide the batch dimension");
        }

        long inner = 1;
        for (int i = axis; i < x.Rank; i++)
        {
            inner *= x.Shape[i].Fixed;
        }

        long outer = 1;
        if (!sym)
        {
            for (int i = 0; i < axis; i++)
            {
                outer *= x.Shape[i].Fixed;
            }
        }

        SetDyn(n, 0, Emit("Reshape", [x], Attributes.From(("shape", L(sym ? 0 : outer, inner))), Nm(n)));
    }

    private void SqueezeUnsqueeze(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        bool sym = Sym(x);
        var axesArr = n.Has("axes") ? n.Ints("axes") : OptConst(ins, 1, "axes")?.Ints();
        var dims = x.Shape.Dims.Select(d => d.IsSymbolic ? 0L : d.Fixed).ToArray();   // 0 marks the symbolic outer dim
        if (ins[0]!.Scalar && n.OpType == "Squeeze")
        {
            SetDyn(n, 0, x);
            _env[n.Outputs[0]].Scalar = true;
            return;
        }
        if (ins[0]!.Scalar)
        {
            dims = [];   // a logical scalar (stored as [1]): unsqueeze yields rank = number of axes, squeeze is a no-op
        }

        long[] outDims;
        if (n.OpType == "Unsqueeze")
        {
            if (axesArr is null)
            {
                throw new InvalidDataException("Unsqueeze without axes");
            }

            outDims = OnnxFold.UnsqueezeDims(dims, axesArr);
            if (sym && outDims[0] != 0)
            {
                throw new NotSupportedException("Unsqueeze at axis 0 of a batched tensor would move the batch dimension");
            }
        }
        else
        {
            if (sym && axesArr is not null && axesArr.Any(a => a == 0 || a == -x.Rank))
            {
                throw new NotSupportedException("Squeeze of the batch dimension");
            }

            outDims = OnnxFold.SqueezeDims(sym ? dims.Select((d, i) => i == 0 ? -7 : d).ToArray() : dims, axesArr);
            if (sym)
            {
                outDims = outDims.Select(d => d == -7 ? 0 : d).ToArray();
            }

            if (sym && axesArr is null && outDims[0] != 0)
            {
                throw new NotSupportedException("Squeeze moved the batch dimension");
            }
        }
        SetDyn(n, 0, Emit("Reshape", [x], Attributes.From(("shape", outDims)), Nm(n)));
    }

    private void Concat(OnnxNode n, TranslatedTensor?[] ins)
    {
        var vals = ins.Select((t, i) => X(ins, i, Nm(n) + $".in{i}")).ToList();
        int axis = NormalizeAxis(n.Int("axis", 0), vals[0].Rank);

        if (axis == 0 && Sym(vals[0]))
        {
            throw new NotSupportedException("Concat along the symbolic batch axis");
        }

        if (vals.Count == 1)
        {
            SetDyn(n, 0, vals[0]);
            return;
        }
        var cur = vals;
        while (cur.Count > 64)
        {
            cur = [_b.Concat(axis, cur.Take(64).ToArray()), .. cur.Skip(64)];
        }

        SetDyn(n, 0, Emit("Concat", [.. cur], Attributes.From(("axis", axis)), Nm(n)));
    }

    private Value SliceAxis(Value x, int axis, long start, long length)
    {
        if (start == 0 && length == x.Shape[axis].Fixed)
        {
            return x;
        }

        return _b.Slice(x, axis, start, length);
    }

    private void Slice(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        long[] starts;
        long[] ends;
        long[]? axes;
        long[]? steps = null;
        if (n.Has("starts"))
        {
            starts = n.Ints("starts")!;
            ends = n.Ints("ends")!;
            axes = n.Ints("axes");
        }
        else
        {
            starts = ConstIn(ins, 1, "Slice starts").Ints();
            ends = ConstIn(ins, 2, "Slice ends").Ints();
            axes = OptConst(ins, 3, "Slice axes")?.Ints();
            steps = OptConst(ins, 4, "Slice steps")?.Ints();
        }
        axes ??= Enumerable.Range(0, starts.Length).Select(i => (long)i).ToArray();
        var y = x;
        for (int i = 0; i < axes.Length; i++)
        {
            int ax = (int)(axes[i] < 0 ? axes[i] + x.Rank : axes[i]);
            long step = steps is null ? 1 : steps[i];
            if (step == 0)
            {
                throw new InvalidDataException("Slice with step 0");
            }

            if (step < 0)
            {
                if (ax == 0 && Sym(x))
                {
                    throw new NotSupportedException("slicing the symbolic batch axis");
                }

                y = ReverseSlice(y, ax, starts[i], ends[i], step);
                continue;
            }
            if (ax == 0 && Sym(x))
            {
                if (starts[i] == 0 && ends[i] >= int.MaxValue)
                {
                    continue;
                }

                throw new NotSupportedException("slicing the symbolic batch axis");
            }
            var (s, l) = OnnxFold.SliceRange(starts[i], ends[i], x.Shape[ax].Fixed);
            y = step == 1 ? SliceAxis(y, ax, s, l) : StridedSlice(y, ax, s, l, step);
        }
        SetDyn(n, 0, ReferenceEquals(y, x) ? Emit("Reshape", [x], Attributes.From(("shape", x.Shape.Dims.Select((d, i) => d.IsSymbolic ? 0L : d.Fixed).ToArray())), Nm(n)) : y);
    }

    /// <summary>Negative-step slice as a gather with constant indices.</summary>
    private Value ReverseSlice(Value x, int ax, long start, long end, long step)
    {
        long dim = x.Shape[ax].Fixed;
        if (start < 0)
        {
            start += dim;
        }

        start = Math.Clamp(start, 0, dim - 1);
        if (end < 0)
        {
            end += dim;
        }

        end = Math.Clamp(end, -1, dim - 1);
        var idx = new List<long>();
        for (long k = start; k > end; k += step)
        {
            idx.Add(k);
        }

        if (idx.Count == 0)
        {
            throw new NotSupportedException("empty Slice result");
        }

        return Emit("GatherAxis", [x, Idx(TranslatedTensor.Const(OnnxConst.FromInts(OnnxType.Int64, [idx.Count], idx.ToArray())))], Attributes.From(("axis", (long)ax)));
    }

    /// <summary>x[start : start+length : step] along <paramref name="ax"/> as Slice + Reshape([..., groups, step, ...]) + Slice(0) (+ one trailing element).</summary>
    private Value StridedSlice(Value x, int ax, long start, long length, long step)
    {
        long full = length / step;
        long rem = length % step;
        long[] dims = x.Shape.Dims.Select(d => d.IsSymbolic ? 0L : d.Fixed).ToArray();
        Value? main = null;
        if (full > 0)
        {
            var region = SliceAxis(x, ax, start, full * step);
            var split = dims.Take(ax).Append(full).Append(step).Concat(dims.Skip(ax + 1)).ToArray();
            var picked = _b.Slice(_b.Reshape(region, split), ax + 1, 0, 1);
            main = _b.Reshape(picked, dims.Take(ax).Append(full).Concat(dims.Skip(ax + 1)).ToArray());
        }
        if (rem == 0)
        {
            return main!;
        }

        var tail = _b.Slice(x, ax, start + full * step, 1);
        return main is null ? tail : _b.Concat(ax, main, tail);
    }

    private void Split(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        int axis = NormalizeAxis(n.Int("axis", 0), x.Rank);
        if (axis == 0 && Sym(x))
        {
            throw new NotSupportedException("Split along the symbolic batch axis");
        }

        var sizes = n.Has("split") ? n.Ints("split") : OptConst(ins, 1, "Split sizes")?.Ints();
        int outs = n.Outputs.Count;
        sizes ??= Enumerable.Repeat(x.Shape[axis].Fixed / outs, outs).ToArray();
        long start = 0;
        for (int i = 0; i < outs; i++)
        {
            var piece = SliceAxis(x, axis, start, sizes[i]);
            if (ReferenceEquals(piece, x))
            {
                piece = Emit("Reshape", [x], Attributes.From(("shape", x.Shape.Dims.Select(d => d.IsSymbolic ? 0L : d.Fixed).ToArray())));
            }

            if (n.Outputs[i] != "")
            {
                SetDyn(n, i, piece);
            }

            start += sizes[i];
        }
    }

    /// <summary>A zero tensor [N,1,...,1] with the dynamic batch size, used to materialize broadcasts.</summary>
    private Value BatchZeros(int rank)
    {
        if (_batchZero is null)
        {
            var inp = _firstInput ?? throw new NotSupportedException("broadcasting a constant to the batch size needs a floating point graph input");
            var v = inp;
            for (int a = 1; a < inp.Rank; a++)
            {
                v = _b.Slice(v, a, 0, 1);
            }

            _batchZero = _b.Scale(v, 0f, 0f);
        }
        var shape = new long[rank];
        for (int i = 1; i < rank; i++)
        {
            shape[i] = 1;
        }

        return Rs(_batchZero, shape);
    }

    private void Expand(OnnxNode n, TranslatedTensor?[] ins)
    {
        var xt = ins[0]!;
        var target = ConstIn(ins, 1, "Expand shape").Ints();
        var x = AsValue(xt);
        int rank = Math.Max(x.Rank, target.Length);
        var xd = Enumerable.Repeat(1L, rank - x.Rank).Concat(x.Shape.Dims.Select(d => d.IsSymbolic ? OnnxConst.SymBatch : d.Fixed)).ToArray();
        var td = Enumerable.Repeat(1L, rank - target.Length).Concat(target).ToArray();
        var res = new long[rank];
        bool changed = false;
        for (int i = 0; i < rank; i++)
        {
            if (td[i] == OnnxConst.SymBatch || xd[i] == OnnxConst.SymBatch)
            {
                res[i] = OnnxConst.SymBatch;
                if (xd[i] != OnnxConst.SymBatch)
                {
                    changed = true;
                }
            }
            else if (td[i] == xd[i] || td[i] == 1)
            {
                res[i] = xd[i];
            }
            else if (xd[i] == 1)
            {
                res[i] = td[i];
                changed = true;
            }
            else
            {
                throw new InvalidDataException("Expand: incompatible shapes.");
            }
        }
        if (res.Count(v => v == OnnxConst.SymBatch) > 1 || (res.Any(v => v == OnnxConst.SymBatch) && res[0] != OnnxConst.SymBatch))
        {
            throw new NotSupportedException("Expand of the batch dimension to a position other than the outermost");
        }

        if (!changed && x.Rank == rank)
        {
            _env[n.Outputs[0]] = xt;
            return;
        }
        // y = x + zeros(target): the zeros come from a batch-sized zero tensor and (if needed) a static zero constant.
        bool hasBatch = res[0] == OnnxConst.SymBatch;
        var zeroParts = new List<Value>();
        var innerDims = res.Skip(hasBatch ? 1 : 0).ToArray();
        bool needInner = false;
        for (int i = hasBatch ? 1 : 0; i < rank; i++)
        {
            if (xd[i] != res[i])
            {
                needInner = true;
            }
        }

        Value y = x;
        if (x.Rank < rank)
        {
            y = Emit("Reshape", [x], Attributes.From(("shape", xd.Select((d, i) => d == OnnxConst.SymBatch ? 0L : d).ToArray())));
        }

        if (hasBatch && xd[0] != OnnxConst.SymBatch)
        {
            y = Emit("Add", [y, BatchZeros(rank)]);
        }

        if (needInner)
        {
            var zd = (hasBatch ? new long[] { 1 } : []).Concat(innerDims).ToArray();
            var z = _b.Constant($"{Nm(n)}.zeros", DataType.F32, new Shape(zd.Select(d => (Dim)d).ToArray()), new byte[zd.Aggregate(1L, (a, b) => a * b) * 4]);
            y = Emit("Add", [y, z]);
        }
        SetDyn(n, 0, y);
    }

    /// <summary>Shape of a dynamic tensor as a compile-time constant (the symbolic batch dimension is encoded as <see cref="OnnxConst.SymBatch"/>).</summary>
    private void ShapeOf(OnnxNode n, TranslatedTensor?[] ins)
    {
        var dims = ins[0]!.V!.Shape.Dims.Select(d => d.IsSymbolic ? OnnxConst.SymBatch : d.Fixed).ToArray();
        long start = n.Int("start", 0);
        long end = n.Int("end", dims.Length);
        if (start < 0)
        {
            start += dims.Length;
        }

        if (end < 0)
        {
            end += dims.Length;
        }

        start = Math.Clamp(start, 0, dims.Length);
        end = Math.Clamp(end, start, dims.Length);
        _env[n.Outputs[0]] = TranslatedTensor.Const(OnnxConst.FromInts(OnnxType.Int64, [end - start], dims[(int)start..(int)end]));
    }

    private void SizeOf(OnnxNode n, TranslatedTensor?[] ins)
    {
        if (Sym(ins[0]!.V!))
        {
            throw new NotSupportedException("Size of a tensor with a symbolic batch dimension");
        }

        _env[n.Outputs[0]] = TranslatedTensor.Const(OnnxConst.ScalarInt(ins[0]!.V!.Shape.Elements()));
    }

    private void Reshape(OnnxNode n, TranslatedTensor?[] ins)
    {
        var target = ConstIn(ins, 1, "Reshape target shape").Ints();
        SetDyn(n, 0, ReshapeOnnx(X(ins, 0), target, n.Int("allowzero", 0) != 0, Nm(n)));
    }

    private void Transpose(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        var perm = n.Ints("perm")?.Select(p => (int)p).ToArray() ?? Enumerable.Range(0, x.Rank).Reverse().ToArray();
        SetDyn(n, 0, Permute(x, perm));
    }

    private void Tile(OnnxNode n, TranslatedTensor?[] ins)
    {
        var attributes = Attributes.From(("repeats", ConstIn(ins, 1, "Tile repeats").Ints()));
        SetDyn(n, 0, Emit("Tile", [X(ins, 0)], attributes, Nm(n)));
    }

    private void Trilu(OnnxNode n, TranslatedTensor?[] ins)
    {
        long k = ins.Length > 1 && ins[1] is not null ? ConstIn(ins, 1, "Trilu k").ScalarInt() : 0;
        var attributes = Attributes.From(("upper", n.Int("upper", 1) != 0), ("k", k));
        SetDyn(n, 0, Emit("Trilu", [X(ins, 0)], attributes, Nm(n)));
    }
}
