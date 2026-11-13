using Sharplight.Graph;

namespace Sharplight.Onnx;

// Ops whose output size depends on tensor values (NonMaxSuppression, NonZero, Unique, Compress): the outer output dimension is a
// dynamic symbol with a compile-time bound; the op publishes its value when it runs (see docs/dynamic-shapes.md).
internal sealed partial class OnnxTranslator
{
    private void NonMaxSuppression(OnnxNode n, TranslatedTensor?[] ins)
    {
        var boxes = X(ins, 0);
        var scores = X(ins, 1);
        if (boxes.Rank != 3 || scores.Rank != 3)
        {
            throw new NotSupportedException("NonMaxSuppression needs boxes [B, S, 4] and scores [B, C, S]");
        }

        long maxPerClass = OptConst(ins, 2, "NonMaxSuppression max_output_boxes_per_class")?.ScalarInt() ?? 0;
        double iou = OptConst(ins, 3, "NonMaxSuppression iou_threshold")?.Floats()[0] ?? 0;
        double score = OptConst(ins, 4, "NonMaxSuppression score_threshold")?.Floats()[0] ?? double.NegativeInfinity;
        long boxCount = scores.Shape[2].Fixed;
        long perClass = Math.Max(0, Math.Min(boxCount, maxPerClass));
        var bound = scores.Shape[0] * scores.Shape[1].Fixed * perClass;
        string count = _b.DynamicSymbol("nms", bound);
        var attributes = Attributes.From(("count_symbol", count), ("max_per_class", perClass), ("iou_threshold", (float)iou),
            ("score_threshold", (float)score), ("center_point_box", n.Int("center_point_box", 0) != 0));
        SetDyn(n, 0, Emit("NonMaxSuppression", [boxes, scores], attributes, Nm(n)));
    }
}
