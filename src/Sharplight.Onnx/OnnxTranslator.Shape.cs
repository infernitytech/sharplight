using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// Shape manipulation: reshape, squeeze, concat, slice, split, expand and friends.
internal sealed partial class OnnxTranslator
{
    // ================================ shape manipulation ===================================================
    /// <summary>
    /// ONNX Reshape semantics (0 = copy, -1 = infer, <see cref="SymbolicInts"/> codes = symbolic extents such as <c>N</c> or <c>12*N</c>)
    /// on top of Sharplight's. Only the outermost target dim may be symbolic; it becomes the <c>outer_symbol</c> of the IR Reshape.
    /// </summary>
    private Value ReshapeOnnx(Value x, long[] req, bool allowZero, string? name = null)
    {
        bool srcSym = Sym(x);
        var dims = new SymbolicExpression?[req.Length];
        int infer = -1;
        for (int k = 0; k < req.Length; k++)
        {
            long v = req[k];
            if (v == -1)
            {
                if (infer >= 0)
                {
                    throw new InvalidDataException("Reshape: two inferred dims.");
                }

                infer = k;
                continue;
            }

            if (v == 0 && !allowZero)
            {
                if (k >= x.Rank)
                {
                    throw new InvalidDataException("Reshape: 0 beyond the input rank.");
                }

                dims[k] = x.Shape[k].IsSymbolic ? x.Shape[k].Expression : SymbolicExpression.Constant(x.Shape[k].Fixed);
                continue;
            }

            dims[k] = SymbolicInts.Decode(v);
        }

        if (!srcSym)
        {
            if (dims.Any(d => d is { IsConstant: false }))
            {
                throw new NotSupportedException("the target shape has a symbolic extent but the input has none");
            }

            var fixedDims = dims.Select(d => d?.ConstantValue ?? -1).ToArray();
            return Emit("Reshape", [x], Attributes.From(("shape", fixedDims)), name);
        }

        if (req.Length == 0)
        {
            throw new NotSupportedException("reshaping a batched tensor to a scalar");
        }

        var total = x.Shape[0].Expression * x.Shape.InnerElements;
        if (infer >= 0)
        {
            var known = dims.Where(d => d is not null).Aggregate(SymbolicExpression.Constant(1), (a, b) => a * b!);
            dims[infer] = DivideExactly(total, known)
                ?? throw new NotSupportedException($"reshape of {x.Shape} to [{string.Join(",", req.Select(SymbolicInts.Format))}]: cannot infer the -1 dim symbolically");
        }

        if (dims.Skip(1).Any(d => !d!.IsConstant))
        {
            throw new NotSupportedException($"reshape of {x.Shape} to [{string.Join(", ", dims.Select(d => d!.ToString()))}] puts a symbolic extent on an inner dim");
        }

        var outer = dims[0]!;
        if (outer.IsConstant)
        {
            throw new NotSupportedException($"reshape of batched {x.Shape} fixes the batch dimension to {outer}");
        }

        var shape = dims.Select(d => d!.IsConstant ? d.ConstantValue : 0L).ToArray();
        if (outer.Equals(x.Shape[0].Expression))
        {
            return Emit("Reshape", [x], Attributes.From(("shape", shape)), name);
        }

        return Emit("Reshape", [x], Attributes.From(("shape", shape), ("outer_symbol", outer.ToString())), name);
    }

    /// <summary><paramref name="total"/> / <paramref name="known"/> when the quotient is the same expression for every symbol value, else null.</summary>
    private static SymbolicExpression? DivideExactly(SymbolicExpression total, SymbolicExpression known)
    {
        if (known.IsConstant)
        {
            if (known.ConstantValue == 0)
            {
                return null;
            }

            var q = total.FloorDiv(known.ConstantValue);
            return (q * known.ConstantValue).Equals(total) ? q : null;
        }

        // Symbolic divisor: the quotient must be a constant (e.g. 12*N / N); check it at several symbol values.
        long? ratio = null;
        foreach (long probe in new long[] { 1, 2, 3, 7, 64 })
        {
            long t = total.Evaluate(_ => probe);
            long k = known.Evaluate(_ => probe);
            if (k == 0 || t % k != 0 || (ratio is { } r && r != t / k))
            {
                return null;
            }

            ratio = t / k;
        }

        return (known * ratio!.Value).Equals(total) ? SymbolicExpression.Constant(ratio.Value) : null;
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
        const long symbolic = -7;   // marks the symbolic outer dim while the ONNX dims are rearranged
        int lead = ins[0]!.LeadingUnits;
        var dims = Enumerable.Repeat(1L, lead).Concat(x.Shape.Dims.Select(d => d.IsSymbolic ? symbolic : d.Fixed)).ToArray();
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
            outDims = OnnxFold.UnsqueezeDims(dims, axesArr ?? throw new InvalidDataException("Unsqueeze without axes"));
        }
        else
        {
            if (sym && axesArr is not null && axesArr.Any(a => NormalizeAxis(a, dims.Length) == lead))
            {
                throw new NotSupportedException("Squeeze of the batch dimension");
            }

            outDims = OnnxFold.SqueezeDims(dims, axesArr);
        }

        // Unit dims in front of the symbolic dim are not part of the IR value (only its outermost dim may be symbolic).
        int outLead = sym ? Array.IndexOf(outDims, symbolic) : 0;
        if (outLead < 0 || outDims.Take(outLead).Any(d => d != 1))
        {
            throw new NotSupportedException($"{n.OpType} moved the symbolic dimension behind a non-unit dim");
        }

        var irDims = outDims.Skip(outLead).Select(d => d == symbolic ? 0 : d).ToArray();
        SetDyn(n, 0, Emit("Reshape", [x], Attributes.From(("shape", irDims)), Nm(n)));
        _env[n.Outputs[0]].LeadingUnits = outLead;
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
            starts = ConstIn(ins, 1, "Slice starts", symbolic: true).Ints();
            ends = ConstIn(ins, 2, "Slice ends", symbolic: true).Ints();
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
                if (starts[i] == 0 && (ends[i] >= int.MaxValue || ends[i] == SymbolicInts.Encode(x.Shape[0])))
                {
                    continue;
                }

                throw new NotSupportedException($"slicing the symbolic batch axis ({SymbolicInts.Format(starts[i])}:{SymbolicInts.Format(ends[i])})");
            }

            if (SymbolicInts.IsSymbolic(starts[i]) || SymbolicInts.IsSymbolic(ends[i]))
            {
                throw new NotSupportedException($"slicing a fixed axis at the symbolic position {SymbolicInts.Format(SymbolicInts.IsSymbolic(starts[i]) ? starts[i] : ends[i])}");
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

    private void Expand(OnnxNode n, TranslatedTensor?[] ins)
    {
        var xt = ins[0]!;
        var target = ConstIn(ins, 1, "Expand shape", symbolic: true).Ints();
        var x = AsValue(xt);
        int rank = Math.Max(x.Rank, target.Length);
        var xd = Enumerable.Repeat(1L, rank - x.Rank).Concat(x.Shape.Dims.Select(SymbolicInts.Encode)).ToArray();
        var td = Enumerable.Repeat(1L, rank - target.Length).Concat(target).ToArray();
        var res = new long[rank];
        for (int i = 0; i < rank; i++)
        {
            if (td[i] == xd[i] || td[i] == 1)
            {
                res[i] = xd[i];
            }
            else if (xd[i] == 1)
            {
                res[i] = td[i];
            }
            else
            {
                throw new NotSupportedException($"Expand of [{string.Join(", ", xd.Select(SymbolicInts.Format))}] to [{string.Join(", ", td.Select(SymbolicInts.Format))}]: incompatible or undecidable extents");
            }
        }

        if (res.Skip(1).Any(SymbolicInts.IsSymbolic))
        {
            throw new NotSupportedException("Expand to a symbolic extent at a position other than the outermost");
        }

        if (x.Rank == rank && res.SequenceEqual(xd))
        {
            _env[n.Outputs[0]] = xt;
            return;
        }

        // y = x + zeros(target): a Fill over the symbolic outer extent and (if needed) a static zero constant for the inner dims.
        bool symbolic = SymbolicInts.IsSymbolic(res[0]);
        Value y = x;
        if (x.Rank < rank)
        {
            y = Emit("Reshape", [x], Attributes.From(("shape", xd.Select(d => SymbolicInts.IsSymbolic(d) ? 0L : d).ToArray())));
        }

        if (symbolic && xd[0] != res[0])
        {
            y = Emit("Add", [y, FillOuter(SymbolicInts.Decode(res[0]), Enumerable.Repeat(1L, rank - 1).ToArray(), y.Type, 0, 0)]);
        }

        if (res.Skip(symbolic ? 1 : 0).Where((d, i) => d != xd[i + (symbolic ? 1 : 0)]).Any())
        {
            var zd = (symbolic ? new long[] { 1 } : []).Concat(res.Skip(symbolic ? 1 : 0)).ToArray();
            var z = _b.Constant($"{Nm(n)}.zeros", DataType.F32, new Shape(zd.Select(d => (Dim)d).ToArray()), new byte[zd.Aggregate(1L, (a, b) => a * b) * 4]);
            y = Emit("Add", [y, z]);
        }

        SetDyn(n, 0, y);
    }

    /// <summary>Shape of a dynamic tensor as a compile-time constant (symbolic dims are encoded as <see cref="SymbolicInts"/> codes).</summary>
    private void ShapeOf(OnnxNode n, TranslatedTensor?[] ins)
    {
        var dims = Enumerable.Repeat(1L, ins[0]!.LeadingUnits).Concat(ins[0]!.V!.Shape.Dims.Select(SymbolicInts.Encode)).ToArray();
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
        var shape = ins[0]!.V!.Shape;
        long size = Sym(ins[0]!.V!) ? SymbolicInts.Encode(shape[0].Expression * shape.InnerElements) : shape.Elements();
        _env[n.Outputs[0]] = TranslatedTensor.Const(OnnxConst.ScalarInt(size));
    }

    private void Reshape(OnnxNode n, TranslatedTensor?[] ins)
    {
        var target = ConstIn(ins, 1, "Reshape target shape", symbolic: true).Ints();
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
        var repeats = ConstIn(ins, 1, "Tile repeats", symbolic: true).Ints();
        var x = X(ins, 0);
        if (!SymbolicInts.AnySymbolic(repeats))
        {
            SetDyn(n, 0, Emit("Tile", [x], Attributes.From(("repeats", repeats)), Nm(n)));
            return;
        }

        // Repeating a unit outer dim a symbolic number of times is a broadcast over that extent.
        if (repeats.Skip(1).Any(SymbolicInts.IsSymbolic) || Sym(x) || x.Rank == 0 || x.Shape[0].Fixed != 1)
        {
            throw new NotSupportedException($"Tile of {x.Shape} by [{string.Join(", ", repeats.Select(SymbolicInts.Format))}]");
        }

        var inner = (long[])[1, .. repeats[1..]];
        var y = inner.All(r => r == 1) ? x : Emit("Tile", [x], Attributes.From(("repeats", inner)));
        var zeros = FillOuter(SymbolicInts.Decode(repeats[0]), Enumerable.Repeat(1L, x.Rank - 1).ToArray(), y.Type, 0, 0);
        SetDyn(n, 0, Emit("Add", [y, zeros], null, Nm(n)));
    }

    private void Trilu(OnnxNode n, TranslatedTensor?[] ins)
    {
        long k = ins.Length > 1 && ins[1] is not null ? ConstIn(ins, 1, "Trilu k").ScalarInt() : 0;
        var attributes = Attributes.From(("upper", n.Int("upper", 1) != 0), ("k", k));
        SetDyn(n, 0, Emit("Trilu", [X(ins, 0)], attributes, Nm(n)));
    }
}
