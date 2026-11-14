using Sharplight.Graph;

namespace Sharplight.Onnx;

// Image-layout operators expressed with existing IR ops: Col2Im (a grouped transposed convolution with one-hot weights), AffineGrid
// (a constant base grid times theta), CenterCropPad (slices and pads) and wrap padding (slices and concats).
internal sealed partial class OnnxTranslator
{
    /// <summary>
    /// Col2Im: input [N, C·∏block, L] → [N, C, *image]. Every column block is scattered back with overlapping entries summed, which is
    /// exactly a transposed convolution of the [N, C·K, *block grid] tensor with groups = C and a one-hot [C·K, 1, *block] weight
    /// (2-D or 3-D images).
    /// </summary>
    private void Col2Im(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        var image = ConstIn(ins, 1, "Col2Im image_shape").Ints();
        var block = ConstIn(ins, 2, "Col2Im block_shape").Ints();
        int s = image.Length;
        if (s is not (2 or 3))
        {
            throw new NotSupportedException($"Col2Im with {s} spatial dims (2-D and 3-D images)");
        }

        var strides = n.Ints("strides") ?? Enumerable.Repeat(1L, s).ToArray();
        var dilations = n.Ints("dilations") ?? Enumerable.Repeat(1L, s).ToArray();
        var pads = n.Ints("pads") ?? new long[2 * s];
        long k = block.Aggregate(1L, (a, b) => a * b);
        long channels = x.Shape[1].Fixed / k;
        var grid = new long[s];
        var outputPadding = new long[s];
        for (int i = 0; i < s; i++)
        {
            long effective = dilations[i] * (block[i] - 1) + 1;
            grid[i] = (image[i] + pads[i] + pads[i + s] - effective) / strides[i] + 1;
            outputPadding[i] = image[i] - (strides[i] * (grid[i] - 1) + effective - pads[i] - pads[i + s]);
        }

        if (grid.Aggregate(1L, (a, b) => a * b) != x.Shape[2].Fixed)
        {
            throw new InvalidDataException($"Col2Im: {x.Shape[2]} blocks do not match the {string.Join("x", grid)} block grid");
        }

        var oneHot = new double[channels * k * k];
        for (long row = 0; row < channels * k; row++)
        {
            oneHot[row * k + row % k] = 1;
        }

        var weight = OnnxConst.FromFloats(OnnxType.Float, [channels * k, 1, .. block], oneHot);
        var blocks = Rs(x, [0, channels * k, .. grid]);
        Value y;
        if (s == 3)
        {
            y = VolumeConv(blocks, weight, true, strides, pads, dilations, outputPadding, channels, Nm(n));
        }
        else
        {
            var attributes = Attributes.From(("strides", strides), ("pads", pads), ("dilations", dilations), ("groups", channels), ("output_padding", outputPadding));
            y = Emit("ConvTranspose", [blocks, WeightValue(weight, false, Nm(n) + ".blocks")], attributes, Nm(n));
        }

        Heavy();
        SetDyn(n, 0, y);
    }

    /// <summary>
    /// AffineGrid: theta [N, 2, 3] (or [N, 3, 4]) and a constant size [N, C, H, W] (or [N, C, D, H, W]) → sampling grid [N, H, W, 2]
    /// (or [N, D, H, W, 3]) = theta applied to the normalized base grid.
    /// </summary>
    private void AffineGrid(OnnxNode n, TranslatedTensor?[] ins)
    {
        var theta = X(ins, 0);
        var size = ConstIn(ins, 1, "AffineGrid size").Ints();
        bool alignCorners = n.Int("align_corners", 0) != 0;
        int spatial = size.Length - 2;
        long[] extents = size[2..];
        long points = extents.Aggregate(1L, (a, b) => a * b);
        // base^T [spatial + 1, points]: rows x (innermost axis), y, (z,) 1.
        var baseT = new double[(spatial + 1) * points];
        for (long p = 0; p < points; p++)
        {
            long rest = p;
            for (int axis = spatial - 1; axis >= 0; axis--)
            {
                long len = extents[axis];
                long i = rest % len;
                rest /= len;
                double coordinate = alignCorners ? (len == 1 ? 0 : -1 + 2.0 * i / (len - 1)) : (2.0 * i + 1) / len - 1;
                baseT[(spatial - 1 - axis) * points + p] = coordinate;
            }

            baseT[spatial * points + p] = 1;
        }

        var projected = MatMulWeightOrBatch(theta, OnnxConst.FromFloats(OnnxType.Float, [spatial + 1, points], baseT), Nm(n) + ".base");
        var grid = Permute(projected, [0, 2, 1]);
        SetDyn(n, 0, Emit("Reshape", [grid], Attributes.From(("shape", (long[])[0, .. extents, spatial])), Nm(n)));
    }

    /// <summary>CenterCropPad: per axis, a centered slice (target smaller) or a centered zero pad (target larger).</summary>
    private void CenterCropPad(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        var target = ConstIn(ins, 1, "CenterCropPad shape").Ints();
        int rank = x.Rank;
        var axes = (n.Ints("axes") ?? Enumerable.Range(0, rank).Select(a => (long)a).ToArray()).Select(a => NormalizeAxis(a, rank)).ToArray();
        var pads = new long[2 * rank];
        var y = x;
        for (int k = 0; k < axes.Length; k++)
        {
            int axis = axes[k];
            long length = x.Shape[axis].Fixed;
            if (target[k] < length)
            {
                y = _b.Slice(y, axis, (length - target[k]) / 2, target[k]);
            }
            else if (target[k] > length)
            {
                pads[axis] = (target[k] - length) / 2;
                pads[rank + axis] = target[k] - length - pads[axis];
            }
        }

        if (pads.Any(p => p != 0))
        {
            y = Emit("Pad", [y], Attributes.From(("pads", pads), ("mode", "constant"), ("value", 0f)));
        }

        SetDyn(n, 0, Emit("Reshape", [y], Attributes.From(("shape", Fixed(y))), Nm(n)));
    }

    /// <summary>Wrap (circular) padding: per padded axis, the source index sequence (i - begin) mod length as concatenated slices.</summary>
    private Value WrapPad(Value x, long[] pads, string name)
    {
        int rank = x.Rank;
        var y = x;
        for (int axis = 0; axis < rank; axis++)
        {
            long begin = pads[axis];
            long end = pads[rank + axis];
            if (begin == 0 && end == 0)
            {
                continue;
            }

            long length = y.Shape[axis].Fixed;
            long total = length + begin + end;
            var pieces = new List<Value>();
            long i = 0;
            while (i < total)
            {
                long source = (((i - begin) % length) + length) % length;
                long run = Math.Min(length - source, total - i);
                pieces.Add(_b.Slice(y, axis, source, run));
                i += run;
            }

            y = ConcatMany(axis, [.. pieces]);
        }

        return Emit("Reshape", [y], Attributes.From(("shape", Fixed(y))), name);
    }
}
