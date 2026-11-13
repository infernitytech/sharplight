using Sharplight.Graph;

namespace Sharplight.Onnx;

// Convolution, pooling, padding, resizing and other spatial operators.
internal sealed partial class OnnxTranslator
{
    // ================================ spatial ============================================================
    private static long[] AutoPad(string mode, long[] inSize, long[] kernel, long[] strides, long[] dilations)
    {
        int r = inSize.Length;
        var pads = new long[2 * r];
        for (int i = 0; i < r; i++)
        {
            long outSize = (inSize[i] + strides[i] - 1) / strides[i];
            long total = Math.Max(0, (outSize - 1) * strides[i] + (kernel[i] - 1) * dilations[i] + 1 - inSize[i]);
            long small = total / 2;
            long big = total - small;
            bool upper = mode == "SAME_UPPER";
            pads[i] = upper ? small : big;
            pads[i + r] = upper ? big : small;
        }
        return pads;
    }

    /// <summary>Returns 4-D view of a 3-D tensor ([N,C,L] → [N,C,1,L]) or the tensor itself.</summary>
    private Value To4D(Value x)
    {
        if (x.Rank == 4)
        {
            return x;
        }

        if (x.Rank == 3)
        {
            return Rs(x, 0, x.Shape[1].Fixed, 1, x.Shape[2].Fixed);
        }

        throw new NotSupportedException($"only 1-D and 2-D spatial operators are supported (input rank {x.Rank})");
    }

    private OnnxNode? FusableBatchNorm(OnnxNode conv)
    {
        var o = conv.Outputs[0];
        if (_graphOutputs.Contains(o) || !_consumers.TryGetValue(o, out var cs) || cs.Count != 1)
        {
            return null;
        }

        var bn = cs[0];
        if (bn.OpType != "BatchNormalization" || bn.Inputs[0] != o || bn.Inputs.Count < 5)
        {
            return null;
        }

        if (bn.Outputs.Count > 1 && bn.Outputs.Skip(1).Any(x => x != "" && (_usedNames.Contains(x) || _graphOutputs.Contains(x))))
        {
            return null;
        }

        for (int i = 1; i < 5; i++)
        {
            if (!_env.TryGetValue(bn.Inputs[i], out var t) || !t.IsConst)
            {
                return null;
            }
        }

        return bn;
    }

    private (float[] S, float[] T) BatchNormParams(OnnxNode bn)
    {
        var scale = Get(bn.Inputs[1]).C!.F32();
        var bias = Get(bn.Inputs[2]).C!.F32();
        var mean = Get(bn.Inputs[3]).C!.F32();
        var var_ = Get(bn.Inputs[4]).C!.F32();
        float eps = bn.Float("epsilon", 1e-5f);
        var s = new float[scale.Length];
        var t = new float[scale.Length];
        for (int i = 0; i < s.Length; i++)
        {
            s[i] = (float)(scale[i] / Math.Sqrt((double)var_[i] + eps));
            t[i] = (float)(bias[i] - (double)mean[i] * s[i]);
        }
        return (s, t);
    }

    private void BatchNorm(OnnxNode n, TranslatedTensor?[] ins)
    {
        CheckUnusedOutputs(n, 1);
        for (int i = 1; i < 5; i++)
        {
            ConstIn(ins, i, "BatchNormalization parameter");
        }

        var x = X(ins, 0);
        var (s, t) = BatchNormParams(n);
        long[] dims = [s.Length, .. Enumerable.Repeat(1L, Math.Max(0, x.Rank - 2))];
        var sc = OnnxConst.FromFloats(OnnxType.Float, dims, s.Select(v => (double)v).ToArray(), Nm(n) + ".bn_scale");
        var sh = OnnxConst.FromFloats(OnnxType.Float, dims, t.Select(v => (double)v).ToArray(), Nm(n) + ".bn_shift");
        var y = _b.Mul(x, ConstValue(sc, null));
        SetDyn(n, 0, Emit("Add", [y, ConstValue(sh, null)], null, Nm(n)));
    }

    private void DepthSpace(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        long b = n.Int("blocksize", 1);
        long c = x.Shape[1].Fixed;
        long h = x.Shape[2].Fixed;
        long w = x.Shape[3].Fixed;
        Value y;
        if (n.OpType == "DepthToSpace")
        {
            bool crd = n.Str("mode", "DCR") == "CRD";
            long oc = c / (b * b);
            y = crd ? Rs(x, 0, oc, b, b, h, w) : Rs(x, 0, b, b, oc, h, w);
            y = _b.Permute(y, crd ? [0, 1, 4, 2, 5, 3] : [0, 3, 4, 1, 5, 2]);
            y = Emit("Reshape", [y], Attributes.From(("shape", L(0, oc, h * b, w * b))), Nm(n));
        }
        else
        {
            y = Rs(x, 0, c, h / b, b, w / b, b);
            y = _b.Permute(y, [0, 3, 5, 1, 2, 4]);
            y = Emit("Reshape", [y], Attributes.From(("shape", L(0, c * b * b, h / b, w / b))), Nm(n));
        }
        SetDyn(n, 0, y);
    }

    private void Pad(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        long[] pads = n.Has("pads") ? n.Ints("pads")! : ConstIn(ins, 1, "Pad pads").Ints();
        float value = n.Has("value") ? n.Float("value", 0) : OptConst(ins, 2, "Pad constant_value") is { } cv ? cv.ScalarF32() : 0f;
        if (OptConst(ins, 3, "Pad axes") is { } axesC)
        {
            var full = new long[2 * x.Rank];
            var ax = axesC.Ints();
            for (int i = 0; i < ax.Length; i++)
            {
                int axis = NormalizeAxis(ax[i], x.Rank);
                full[axis] = pads[i];
                full[axis + x.Rank] = pads[i + ax.Length];
            }

            pads = full;
        }

        string mode = n.Str("mode", "constant");
        if (mode == "wrap")
        {
            SetDyn(n, 0, WrapPad(x, pads, Nm(n)));
            return;
        }

        if (mode is not ("constant" or "reflect" or "edge"))
        {
            throw new NotSupportedException($"Pad mode '{mode}'");
        }

        if (pads.All(p => p == 0))
        {
            SetDyn(n, 0, x);
            return;
        }
        SetDyn(n, 0, Emit("Pad", [x], Attributes.From(("pads", pads), ("mode", mode), ("value", value)), Nm(n)));
    }

    /// <summary>Lifts a 1-D spatial attribute to the 2-D form used by the IR ([1, v] for 1-D operators, a default pair when absent).</summary>
    private static long[] ExpandSpatial(long[]? values, long defaultValue, bool oneD)
    {
        if (values is null)
        {
            return [defaultValue, defaultValue];
        }

        return oneD ? [1, values[0]] : values;
    }

    /// <summary>Explicit pads [top, left, bottom, right] from auto_pad / pads attributes of a convolution or pooling node.</summary>
    private static long[] ResolvePads(OnnxNode n, Value x4D, long[] kernel, long[] strides, long[] dilations, bool oneD)
    {
        string autoPad = n.Str("auto_pad", "NOTSET");
        if (autoPad is "SAME_UPPER" or "SAME_LOWER")
        {
            return AutoPad(autoPad, [x4D.Shape[2].Fixed, x4D.Shape[3].Fixed], kernel, strides, dilations);
        }

        if (autoPad == "VALID")
        {
            return [0, 0, 0, 0];
        }

        if (n.Ints("pads") is { } pads)
        {
            return oneD ? [0, pads[0], 0, pads[1]] : pads;
        }

        return [0, 0, 0, 0];
    }

    private void Conv(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x0 = X(ins, 0);
        if (x0.Rank == 5)
        {
            ConvVolume(n, ins, transposed: false);
            return;
        }

        var weightIn = ConstIn(ins, 1, "Conv weight");
        var biasIn = ins.Length > 2 && ins[2] is not null ? ConstIn(ins, 2, "Conv bias") : null;
        bool oneD = x0.Rank == 3;
        var x = To4D(x0);
        long[] weightDims = oneD ? [weightIn.Dims[0], weightIn.Dims[1], 1, weightIn.Dims[2]] : weightIn.Dims;
        if (weightDims.Length != 4)
        {
            throw new NotSupportedException("only 1-D and 2-D convolutions are supported");
        }

        long groups = n.Int("group", 1);
        var strides = ExpandSpatial(n.Ints("strides"), 1, oneD);
        var dilations = ExpandSpatial(n.Ints("dilations"), 1, oneD);
        long[] kernel = [weightDims[2], weightDims[3]];
        var pads = ResolvePads(n, x, kernel, strides, dilations, oneD);

        // Fold a directly following BatchNormalization into the weights.
        OnnxNode? batchNorm = FusableBatchNorm(n);
        var weight = weightIn.Reshaped(weightDims);
        var bias = biasIn;
        string outName = Nm(n);
        if (batchNorm is not null)
        {
            string baseName = weightIn.Name != "" ? weightIn.Name : outName;
            (weight, bias) = FoldBatchNorm(batchNorm, weight, biasIn, baseName, outName);
            _skip.Add(batchNorm);
            outName = batchNorm.Outputs[0];
        }

        var w = WeightValue(weight, keepHalf: false, outName + ".weight");
        var inputs = new List<Value> { x, w };
        if (bias is not null)
        {
            var flatBias = bias.Rank == 1 && bias.ElemType == OnnxType.Float ? bias : bias.Reshaped([bias.Count]);
            inputs.Add(ConstValue(flatBias, outName + ".bias"));
        }

        var attributes = Attributes.From(("strides", strides), ("pads", pads), ("dilations", dilations), ("groups", groups));
        var y = Emit("Conv", [.. inputs], attributes, oneD ? outName + ".2d" : outName);
        Heavy();
        if (oneD)
        {
            y = Rs(y, 0, y.Shape[1].Fixed, y.Shape[3].Fixed);
        }

        _env[batchNorm is not null ? batchNorm.Outputs[0] : n.Outputs[0]] = TranslatedTensor.Dyn(y);
    }

    /// <summary>Scales the convolution weights per output channel and merges the shift into the bias (lazily evaluated).</summary>
    private (OnnxConst Weight, OnnxConst Bias) FoldBatchNorm(OnnxNode batchNorm, OnnxConst weight, OnnxConst? bias, string baseName, string outName)
    {
        var (scale, shift) = BatchNormParams(batchNorm);
        long outChannels = weight.Dims[0];
        long perChannel = weight.Count / outChannels;
        var foldedWeight = OnnxConst.Lazy(OnnxType.Float, weight.Dims, () =>
        {
            var f = weight.F32();
            for (long o = 0; o < outChannels; o++)
            {
                for (long i = 0; i < perChannel; i++)
                {
                    f[o * perChannel + i] *= scale[o];
                }
            }

            return ToBytes(f);
        }, baseName + ".bnfold");
        var foldedBias = OnnxConst.Lazy(OnnxType.Float, [outChannels], () =>
        {
            var b = bias?.F32();
            var result = new float[outChannels];
            for (int o = 0; o < outChannels; o++)
            {
                result[o] = (b is null ? 0 : b[o] * scale[o]) + shift[o];
            }

            return ToBytes(result);
        }, outName + ".bnfold_bias");
        return (foldedWeight, foldedBias);
    }

    private void GridSample(OnnxNode n, TranslatedTensor?[] ins)
    {
        string mode = n.Str("mode", "bilinear");
        mode = mode == "bilinear" ? "linear" : mode == "bicubic" ? "cubic" : mode;
        var x = X(ins, 0);
        if (x.Rank is not (4 or 5) || (x.Rank == 5 && mode == "cubic"))
        {
            throw new NotSupportedException($"GridSample mode {mode} on a rank-{x.Rank} input (ONNX defines 2-D and volumetric non-cubic sampling)");
        }

        var attributes = Attributes.From(("mode", mode), ("padding", n.Str("padding_mode", "zeros")), ("align_corners", n.Int("align_corners", 0) != 0));
        SetDyn(n, 0, Emit("GridSample", [x, X(ins, 1)], attributes, Nm(n)));
    }

    private void RoiAlign(OnnxNode n, TranslatedTensor?[] ins)
    {
        // Opset 10 has no coordinate_transformation_mode; it behaves like "output_half_pixel".
        string transform = n.Str("coordinate_transformation_mode", Opset >= 16 ? "half_pixel" : "output_half_pixel");
        var attributes = Attributes.From(("output_height", n.Int("output_height", 1)), ("output_width", n.Int("output_width", 1)),
            ("sampling_ratio", n.Int("sampling_ratio", 0)), ("spatial_scale", n.Float("spatial_scale", 1f)), ("mode", n.Str("mode", "avg")),
            ("half_pixel", transform == "half_pixel"));
        SetDyn(n, 0, Emit("RoiAlign", [X(ins, 0), X(ins, 1), Idx(ins[2]!)], attributes, Nm(n)));
    }

    private void ConvTranspose(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x0 = X(ins, 0);
        if (x0.Rank == 5)
        {
            ConvVolume(n, ins, transposed: true);
            return;
        }

        var weightIn = ConstIn(ins, 1, "ConvTranspose weight");
        var bias = ins.Length > 2 && ins[2] is not null ? ConstIn(ins, 2, "ConvTranspose bias") : null;
        bool oneD = x0.Rank == 3;
        var x = To4D(x0);
        long[] weightDims = oneD ? [weightIn.Dims[0], weightIn.Dims[1], 1, weightIn.Dims[2]] : weightIn.Dims;
        var strides = ExpandSpatial(n.Ints("strides"), 1, oneD);
        var dilations = ExpandSpatial(n.Ints("dilations"), 1, oneD);
        var outputPadding = ExpandSpatial(n.Ints("output_padding"), 0, oneD);
        long groups = n.Int("group", 1);
        var pads = ConvTransposePads(n, x, weightDims, strides, dilations, outputPadding, oneD);
        var w = WeightValue(weightIn.Reshaped(weightDims), false, Nm(n) + ".weight");
        var inputs = new List<Value> { x, w };
        if (bias is not null)
        {
            inputs.Add(ConstValue(bias, Nm(n) + ".bias"));
        }

        var attributes = Attributes.From(("strides", strides), ("pads", pads), ("dilations", dilations), ("groups", groups), ("output_padding", outputPadding));
        var y = Emit("ConvTranspose", [.. inputs], attributes, oneD ? null : Nm(n));
        Heavy();
        if (oneD)
        {
            y = Emit("Reshape", [y], Attributes.From(("shape", L(0, y.Shape[1].Fixed, y.Shape[3].Fixed))), Nm(n));
        }

        SetDyn(n, 0, y);
    }

    /// <summary>
    /// Pads of a transposed convolution; derived from output_shape or auto_pad when given. An output_shape larger than the full
    /// transposed extent grows <paramref name="outputPadding"/> instead (negative pads are not representable).
    /// </summary>
    private static long[] ConvTransposePads(OnnxNode n, Value x4D, long[] weightDims, long[] strides, long[] dilations, long[] outputPadding, bool oneD)
    {
        string autoPad = n.Str("auto_pad", "NOTSET");
        long[]? outputShape = n.Ints("output_shape");
        if (outputShape is null && autoPad is not ("SAME_UPPER" or "SAME_LOWER"))
        {
            return n.Ints("pads") is { } pads ? (oneD ? [0, pads[0], 0, pads[1]] : pads) : [0, 0, 0, 0];
        }

        long[] want = outputShape is not null
            ? (oneD ? [1, outputShape[0]] : outputShape)
            : [x4D.Shape[2].Fixed * strides[0], x4D.Shape[3].Fixed * strides[1]];
        var result = new long[4];
        for (int i = 0; i < 2; i++)
        {
            long total = strides[i] * (x4D.Shape[2 + i].Fixed - 1) + outputPadding[i] + (weightDims[2 + i] - 1) * dilations[i] + 1 - want[i];
            if (total < 0)
            {
                // An output_shape beyond the full transposed extent appends zeros at the end, which is extra output padding.
                outputPadding[i] -= total;
                continue;
            }

            bool upper = autoPad == "SAME_UPPER" || (outputShape is not null && autoPad != "SAME_LOWER");
            long small = total / 2;
            long big = total - small;
            result[i] = upper ? small : big;
            result[i + 2] = upper ? big : small;
        }

        return result;
    }
}
