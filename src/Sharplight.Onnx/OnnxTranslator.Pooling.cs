using Sharplight.Graph;

namespace Sharplight.Onnx;

// Pooling: MaxPool / AveragePool / LpPool of any spatial rank, global pooling, MaxUnpool and MaxRoiPool. The IR Pool op is 2-D; pooling
// windows are boxes, so max, sum and the valid-tap counts are separable and N-D pooling runs as a 2-D pass over the last two spatial
// axes followed by one (k, 1) pass per remaining axis.
internal sealed partial class OnnxTranslator
{
    private void Pool(OnnxNode n, TranslatedTensor?[] ins)
    {
        bool max = n.OpType == "MaxPool";
        var x = X(ins, 0);
        var window = PoolWindow(n, x);
        bool ceil = n.Int("ceil_mode", 0) != 0;
        var y = PoolNd(x, max ? "max" : "avg", window, ceil, n.Int("count_include_pad", 0) != 0, Nm(n));
        SetDyn(n, 0, y);
        if (!max)
        {
            CheckUnusedOutputs(n, 1);
            return;
        }

        if (n.Outputs.Count > 1 && n.Outputs[1] != "")
        {
            if (x.Rank > 4)
            {
                throw new NotSupportedException("MaxPool indices for more than two spatial dims");
            }

            bool oneD = x.Rank == 3;
            var attributes = Attributes.From(("kernel", oneD ? [1, window.Kernel[0]] : window.Kernel), ("strides", oneD ? [1, window.Strides[0]] : window.Strides),
                ("pads", oneD ? [0, window.Pads[0], 0, window.Pads[1]] : window.Pads), ("dilations", oneD ? [1, window.Dilations[0]] : window.Dilations),
                ("ceil_mode", ceil), ("storage_order", n.Int("storage_order", 0)));
            var indices = Emit("PoolArgMax", [To4D(x)], attributes, Nm(n, 1));
            SetDyn(n, 1, oneD ? Rs(indices, 0, x.Shape[1].Fixed, indices.Shape[3].Fixed) : indices);
        }
    }

    private void LpPool(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        double p = n.Int("p", 2);
        var window = PoolWindow(n, x);
        var powered = LpPower(x, p);
        var mean = PoolNd(powered, "avg", window, n.Int("ceil_mode", 0) != 0, countIncludePad: true, Nm(n) + ".mean");
        var sum = _b.Mul(mean, ConstValue(PoolTapCounts(x, window, mean), Nm(n) + ".taps"));
        SetDyn(n, 0, LpRoot(sum, p, Nm(n)));
    }

    private void GlobalPool(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        var axes = Enumerable.Range(2, Math.Max(0, x.Rank - 2)).Select(i => (long)i).ToArray();
        if (n.OpType == "GlobalLpPool")
        {
            double p = n.Int("p", 2);
            var summed = Emit("Reduce", [LpPower(x, p)], Attributes.From(("mode", "sum"), ("axes", axes), ("keepdims", true)));
            SetDyn(n, 0, LpRoot(summed, p, Nm(n)));
            return;
        }

        string mode = n.OpType == "GlobalAveragePool" ? "mean" : "max";
        SetDyn(n, 0, Emit("Reduce", [x], Attributes.From(("mode", mode), ("axes", axes), ("keepdims", true)), Nm(n)));
    }

    /// <summary>
    /// MaxUnpool: scatters the pooled values to their recorded flat indices in a zero tensor of the inferred unpooled shape, then pads
    /// with zeros at the end of each axis up to output_shape when one is given.
    /// </summary>
    private void MaxUnpool(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        if (Sym(x))
        {
            throw new NotSupportedException("MaxUnpool with a symbolic batch");
        }

        int spatial = x.Rank - 2;
        var kernel = n.Ints("kernel_shape")!;
        var strides = n.Ints("strides") ?? Enumerable.Repeat(1L, spatial).ToArray();
        var pads = n.Ints("pads") ?? new long[2 * spatial];
        long[] inferred = [.. x.Shape.Dims.Take(2).Select(d => d.Fixed),
            .. Enumerable.Range(0, spatial).Select(k => (x.Shape[k + 2].Fixed - 1) * strides[k] + kernel[k] - pads[k] - pads[spatial + k])];
        long total = inferred.Aggregate(1L, (a, b) => a * b);
        long count = x.Shape.Dims.Aggregate(1L, (a, d) => a * d.Fixed);
        var zeros = Emit("Fill", [], Attributes.From(("shape", new[] { total }), ("start", 0f), ("delta", 0f)));
        var indices = Rs(Idx(ins[1]!), count);
        var scattered = Emit("ScatterElements", [zeros, indices, Rs(x, count)], Attributes.From(("axis", 0L), ("reduction", "none")));
        var y = Emit("Reshape", [scattered], Attributes.From(("shape", inferred)), Nm(n));
        if (OptConst(ins, 2, "MaxUnpool output_shape")?.Ints() is { } shape && !shape.SequenceEqual(inferred))
        {
            var growth = new long[2 * x.Rank];
            for (int k = 0; k < x.Rank; k++)
            {
                growth[x.Rank + k] = shape[k] - inferred[k];
            }

            y = Emit("Pad", [y], Attributes.From(("pads", growth), ("mode", "constant"), ("value", 0f)), Nm(n));
        }

        SetDyn(n, 0, y);
    }

    private void MaxRoiPool(OnnxNode n, TranslatedTensor?[] ins)
    {
        var attributes = Attributes.From(("pooled_shape", n.Ints("pooled_shape")!), ("spatial_scale", n.Float("spatial_scale", 1f)));
        SetDyn(n, 0, Emit("MaxRoiPool", [X(ins, 0), X(ins, 1)], attributes, Nm(n)));
    }

    /// <summary>Kernel, strides, dilations and explicit pads ([begins..., ends...]) of an N-D pooling node.</summary>
    private static PoolingWindow PoolWindow(OnnxNode n, Value x)
    {
        int spatial = x.Rank - 2;
        if (spatial < 1)
        {
            throw new NotSupportedException($"{n.OpType} needs at least one spatial dim (input rank {x.Rank})");
        }

        var kernel = n.Ints("kernel_shape") ?? throw new InvalidDataException($"{n.OpType} without kernel_shape");
        var strides = n.Ints("strides") ?? Enumerable.Repeat(1L, spatial).ToArray();
        var dilations = n.Ints("dilations") ?? Enumerable.Repeat(1L, spatial).ToArray();
        var inSize = x.Shape.Dims.Skip(2).Select(d => d.Fixed).ToArray();
        string autoPad = n.Str("auto_pad", "NOTSET");
        long[] pads = autoPad switch
        {
            "SAME_UPPER" or "SAME_LOWER" => AutoPad(autoPad, inSize, kernel, strides, dilations),
            "VALID" => new long[2 * spatial],
            _ => n.Ints("pads") ?? new long[2 * spatial],
        };
        return new PoolingWindow(kernel, strides, dilations, pads);
    }

    /// <summary>N-D pooling as separable 2-D Pool passes (see the file comment).</summary>
    private Value PoolNd(Value x, string mode, PoolingWindow w, bool ceil, bool countIncludePad, string name)
    {
        int s = x.Rank - 2;
        long c = x.Shape[1].Fixed;
        var inSize = x.Shape.Dims.Skip(2).Select(d => d.Fixed).ToArray();
        if (s == 1)
        {
            var flat = Pool2D(Rs(x, 0, c, 1, inSize[0]), mode, [1, w.Kernel[0]], [1, w.Strides[0]], [0, w.Pads[0], 0, w.Pads[1]], [1, w.Dilations[0]], ceil,
                countIncludePad, name + ".2d");
            return Rs(flat, 0, c, flat.Shape[3].Fixed);
        }

        if (s == 2)
        {
            return Pool2D(x, mode, w.Kernel, w.Strides, w.Pads, w.Dilations, ceil, countIncludePad, name);
        }

        long leading = c;
        for (int j = 0; j < s - 2; j++)
        {
            leading *= inSize[j];
        }

        var outSize = (long[])inSize.Clone();
        var y = Pool2D(Rs(x, 0, leading, inSize[s - 2], inSize[s - 1]), mode, [w.Kernel[s - 2], w.Kernel[s - 1]], [w.Strides[s - 2], w.Strides[s - 1]],
            [w.Pads[s - 2], w.Pads[s - 1], w.Pads[2 * s - 2], w.Pads[2 * s - 1]], [w.Dilations[s - 2], w.Dilations[s - 1]], ceil, countIncludePad, $"{name}.axes{s - 2}");
        outSize[s - 2] = y.Shape[2].Fixed;
        outSize[s - 1] = y.Shape[3].Fixed;
        for (int j = s - 3; j >= 0; j--)
        {
            if (w.Kernel[j] == 1 && w.Strides[j] == 1 && w.Pads[j] == 0 && w.Pads[s + j] == 0)
            {
                continue;
            }

            long before = c;
            for (int k = 0; k < j; k++)
            {
                before *= inSize[k];
            }

            long after = outSize.Skip(j + 1).Aggregate(1L, (a, b) => a * b);
            y = Pool2D(Rs(y, 0, before, inSize[j], after), mode, [w.Kernel[j], 1], [w.Strides[j], 1], [w.Pads[j], 0, w.Pads[s + j], 0], [w.Dilations[j], 1],
                ceil, countIncludePad, $"{name}.axis{j}");
            outSize[j] = y.Shape[2].Fixed;
        }

        return Emit("Reshape", [y], Attributes.From(("shape", (long[])[0, c, .. outSize])), name);
    }

    private Value Pool2D(Value x, string mode, long[] kernel, long[] strides, long[] pads, long[] dilations, bool ceil, bool countIncludePad, string name)
    {
        var attributes = Attributes.From(("mode", mode), ("kernel", kernel), ("strides", strides), ("pads", pads), ("dilations", dilations), ("ceil_mode", ceil),
            ("count_include_pad", countIncludePad));
        return Emit("Pool", [x], attributes, name);
    }

    /// <summary>Per output position the number of window taps inside the padded input (the count_include_pad divisor), as a constant
    /// broadcastable over the pooled tensor; multiplying an average by it gives the window sum.</summary>
    private static OnnxConst PoolTapCounts(Value x, PoolingWindow w, Value pooled)
    {
        int s = x.Rank - 2;
        var outSize = pooled.Shape.Dims.Skip(2).Select(d => d.Fixed).ToArray();
        var perAxis = new long[s][];
        for (int j = 0; j < s; j++)
        {
            long inLen = x.Shape[j + 2].Fixed;
            perAxis[j] = new long[outSize[j]];
            for (long o = 0; o < outSize[j]; o++)
            {
                long start = o * w.Strides[j] - w.Pads[j];
                perAxis[j][o] = Math.Min(w.Kernel[j], (inLen + w.Pads[s + j] - 1 - start) / w.Dilations[j] + 1);
            }
        }

        long total = outSize.Aggregate(1L, (a, b) => a * b);
        var counts = new double[total];
        for (long flat = 0; flat < total; flat++)
        {
            long rest = flat;
            double product = 1;
            for (int j = s - 1; j >= 0; j--)
            {
                product *= perAxis[j][rest % outSize[j]];
                rest /= outSize[j];
            }

            counts[flat] = product;
        }

        return OnnxConst.FromFloats(OnnxType.Float, outSize, counts);
    }

    /// <summary>|x|^p (p = 1 and 2 without Pow).</summary>
    private Value LpPower(Value x, double p) => p switch
    {
        1 => _b.Unary("Abs", x),
        2 => _b.Mul(x, x),
        _ => Emit("Pow", [_b.Unary("Abs", x), ConstValue(OnnxConst.Scalar((float)p), "lp.p")]),
    };

    /// <summary>s^(1/p) (p = 1 and 2 without Pow).</summary>
    private Value LpRoot(Value sum, double p, string name) => p switch
    {
        1 => sum,
        2 => Emit("Sqrt", [sum], null, name),
        _ => Emit("Pow", [sum, ConstValue(OnnxConst.Scalar((float)(1 / p)), "lp.inverse_p")], null, name),
    };
}
