using Sharplight.Graph;

namespace Sharplight.Onnx;

// Volumetric (3-D) convolutions on the 2-D IR Conv / ConvTranspose. The depth axis is folded into the batch and the depth taps into the
// output channels, so one 2-D (transposed) convolution computes every (depth slice, depth tap) plane; a constant 0/1 matrix then sums the
// planes into the output depths. That matrix is [Kd·D, D_out] with at most one 1 per (tap, slice) row, so the extra work is small next to
// the convolution itself.
internal sealed partial class OnnxTranslator
{
    /// <summary>Conv / ConvTranspose on x [N, C, D, H, W] with constant weights.</summary>
    private void ConvVolume(OnnxNode n, TranslatedTensor?[] ins, bool transposed)
    {
        var x = X(ins, 0);
        var weight = ConstIn(ins, 1, $"{n.OpType} weight");
        var bias = ins.Length > 2 && ins[2] is not null ? ConstIn(ins, 2, $"{n.OpType} bias") : null;
        long[] inSize = [.. x.Shape.Dims.Skip(2).Select(d => d.Fixed)];
        long[] kernel = weight.Dims[2..];
        var strides = n.Ints("strides") ?? [1, 1, 1];
        var dilations = n.Ints("dilations") ?? [1, 1, 1];
        long groups = n.Int("group", 1);
        var outputPadding = transposed ? (long[])(n.Ints("output_padding") ?? [0, 0, 0]).Clone() : [0, 0, 0];
        string autoPad = n.Str("auto_pad", "NOTSET");
        long[] pads = transposed
            ? TransposedVolumePads(n, inSize, kernel, strides, dilations, outputPadding)
            : autoPad switch
            {
                "SAME_UPPER" or "SAME_LOWER" => AutoPad(autoPad, inSize, kernel, strides, dilations),
                "VALID" => new long[6],
                _ => n.Ints("pads") ?? new long[6],
            };
        var y = VolumeConv(x, weight, transposed, strides, pads, dilations, outputPadding, groups, Nm(n));
        if (bias is not null)
        {
            y = _b.Add(y, ConstValue(bias.Reshaped([1, bias.Count, 1, 1, 1]), Nm(n) + ".bias"), Nm(n));
        }

        Heavy();
        SetDyn(n, 0, y);
    }

    /// <summary>
    /// Pads of a 3-D transposed convolution: explicit, or derived from output_shape / auto_pad; an output_shape beyond the full transposed
    /// extent grows <paramref name="outputPadding"/>.
    /// </summary>
    private static long[] TransposedVolumePads(OnnxNode n, long[] inSize, long[] kernel, long[] strides, long[] dilations, long[] outputPadding)
    {
        string autoPad = n.Str("auto_pad", "NOTSET");
        long[]? outputShape = n.Ints("output_shape");
        if (outputShape is null && autoPad is not ("SAME_UPPER" or "SAME_LOWER"))
        {
            return n.Ints("pads") ?? new long[6];
        }

        var pads = new long[6];
        for (int i = 0; i < 3; i++)
        {
            long want = outputShape is not null ? outputShape[^(3 - i)] : inSize[i] * strides[i];
            long total = strides[i] * (inSize[i] - 1) + outputPadding[i] + (kernel[i] - 1) * dilations[i] + 1 - want;
            if (total < 0)
            {
                outputPadding[i] -= total;
                continue;
            }

            bool upper = autoPad != "SAME_LOWER";
            pads[i] = upper ? total / 2 : total - total / 2;
            pads[i + 3] = total - pads[i];
        }

        return pads;
    }

    /// <summary>
    /// 3-D (transposed) convolution of x [N, C, D, H, W] with a constant weight ([O, C/g, kd, kh, kw], or [C, O/g, kd, kh, kw] when
    /// transposed); pads are [begin d, h, w, end d, h, w]. Returns [N, O, D_out, H_out, W_out] without bias.
    /// </summary>
    private Value VolumeConv(Value x, OnnxConst weight, bool transposed, long[] strides, long[] pads, long[] dilations, long[] outputPadding, long groups,
        string name)
    {
        if (Sym(x))
        {
            throw new NotSupportedException("3-D convolutions with a symbolic batch; fix the batch size through OnnxImportOptions.InputShapes");
        }

        long batch = x.Shape[0].Fixed;
        long channels = x.Shape[1].Fixed;
        long depth = x.Shape[2].Fixed;
        long taps = weight.Dims[2];
        long outChannels = transposed ? weight.Dims[1] * groups : weight.Dims[0];

        // Depth taps folded into the output channels as (o, kd), which keeps every group's output channels contiguous.
        var folded = transposed
            ? weight.Reshaped([weight.Dims[0], weight.Dims[1] * taps, weight.Dims[3], weight.Dims[4]])
            : weight.Transposed([0, 2, 1, 3, 4]).Reshaped([weight.Dims[0] * taps, weight.Dims[1], weight.Dims[3], weight.Dims[4]]);
        var slices = Rs(Permute(x, [0, 2, 1, 3, 4]), batch * depth, channels, x.Shape[3].Fixed, x.Shape[4].Fixed);
        var attributes = Attributes.From(("strides", strides[1..]), ("pads", (long[])[pads[1], pads[2], pads[4], pads[5]]), ("dilations", dilations[1..]),
            ("groups", groups));
        if (transposed)
        {
            attributes = attributes.With("output_padding", outputPadding[1..]);
        }

        var planes = Emit(transposed ? "ConvTranspose" : "Conv", [slices, WeightValue(folded, false, name + ".weight")], attributes, name + ".planes");
        long ho = planes.Shape[2].Fixed;
        long wo = planes.Shape[3].Fixed;

        long depthOut = transposed
            ? strides[0] * (depth - 1) + outputPadding[0] + (taps - 1) * dilations[0] + 1 - pads[0] - pads[3]
            : (depth + pads[0] + pads[3] - ((taps - 1) * dilations[0] + 1)) / strides[0] + 1;
        var mix = new double[taps * depth * depthOut];
        for (long tap = 0; tap < taps; tap++)
        {
            for (long slice = 0; slice < depth; slice++)
            {
                for (long o = 0; o < depthOut; o++)
                {
                    bool hit = transposed
                        ? slice * strides[0] + tap * dilations[0] - pads[0] == o
                        : o * strides[0] - pads[0] + tap * dilations[0] == slice;
                    if (hit)
                    {
                        mix[(tap * depth + slice) * depthOut + o] = 1;
                    }
                }
            }
        }

        var grouped = Permute(Rs(planes, batch, depth, outChannels, taps, ho * wo), [0, 2, 4, 3, 1]);
        var mixed = MatMulWeightOrBatch(Rs(grouped, batch, outChannels * ho * wo, taps * depth), OnnxConst.FromFloats(OnnxType.Float, [taps * depth, depthOut], mix),
            name + ".depth");
        var y = Permute(Rs(mixed, batch, outChannels, ho, wo, depthOut), [0, 1, 4, 2, 3]);
        return Emit("Reshape", [y], Attributes.From(("shape", (long[])[batch, outChannels, depthOut, ho, wo])), name);
    }
}
