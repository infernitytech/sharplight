using Sharplight.Graph;

namespace Sharplight.Onnx;

// DeformConv maps onto the IR DeformConv op; weights, offsets, bias and mask may all be runtime tensors.
internal sealed partial class OnnxTranslator
{
    private void DeformConv(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        var w = X(ins, 1, Nm(n) + ".weight");
        if (x.Rank != 4)
        {
            throw new NotSupportedException($"DeformConv on a rank-{x.Rank} input (2-D only)");
        }

        long[] kernel = [w.Shape[2].Fixed, w.Shape[3].Fixed];
        var strides = n.Ints("strides") ?? [1, 1];
        var dilations = n.Ints("dilations") ?? [1, 1];
        long[] inSize = [x.Shape[2].Fixed, x.Shape[3].Fixed];
        var pads = n.Str("auto_pad", "NOTSET") switch
        {
            "SAME_UPPER" or "SAME_LOWER" => AutoPad(n.Str("auto_pad", "NOTSET"), inSize, kernel, strides, dilations),
            "VALID" => new long[4],
            _ => n.Ints("pads") ?? new long[4],
        };
        long outChannels = w.Shape[0].Fixed;
        var bias = Opt(ins, 3) is { } b
            ? AsValue(b, Nm(n) + ".bias")
            : ConstValue(OnnxConst.FromFloats(OnnxType.Float, [outChannels], new double[outChannels]), Nm(n) + ".bias");
        List<Value> inputs = [x, w, X(ins, 2), bias];
        if (Opt(ins, 4) is { } mask)
        {
            inputs.Add(AsValue(mask, Nm(n) + ".mask"));
        }

        var attributes = Attributes.From(("strides", strides), ("pads", pads), ("dilations", dilations), ("groups", n.Int("group", 1)),
            ("offset_groups", n.Int("offset_group", 1)));
        var y = Emit("DeformConv", [.. inputs], attributes, Nm(n));
        Heavy();
        SetDyn(n, 0, y);
    }
}
