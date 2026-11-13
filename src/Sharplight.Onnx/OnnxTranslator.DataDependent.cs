using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// Ops whose output size depends on tensor values (NonMaxSuppression, NonZero, Unique, Compress): the outer output dimension is a
// dynamic symbol with a compile-time bound; the op publishes its value when it runs (see docs/dynamic-shapes.md).
internal sealed partial class OnnxTranslator
{
    private void NonMaxSuppression(OnnxNode n, TranslatedTensor?[] ins)
    {
        var boxes = X(ins, 0);
        var scores = X(ins, 1);
        // One batch and class over a symbolic number of boxes: [1, S, 4] and [1, 1, S] are boxes [S, 4] and scores [S] behind unit dims.
        bool single = ins[0]!.LeadingUnits == 1 && ins[1]!.LeadingUnits == 2 && boxes.Rank == 2 && scores.Rank == 1;
        if (!single && (boxes.Rank != 3 || scores.Rank != 3 || ins[0]!.LeadingUnits + ins[1]!.LeadingUnits > 0))
        {
            throw new NotSupportedException("NonMaxSuppression needs boxes [B, S, 4] and scores [B, C, S]");
        }

        long maxPerClass = OptConst(ins, 2, "NonMaxSuppression max_output_boxes_per_class")?.ScalarInt() ?? 0;
        double iou = OptConst(ins, 3, "NonMaxSuppression iou_threshold")?.Floats()[0] ?? 0;
        double score = OptConst(ins, 4, "NonMaxSuppression score_threshold")?.Floats()[0] ?? double.NegativeInfinity;
        var boxCount = scores.Shape[single ? 0 : 2];
        long perClass = Math.Max(0, boxCount.IsSymbolic ? maxPerClass : Math.Min(boxCount.Fixed, maxPerClass));
        var bound = single ? Dim.Min(boxCount, perClass) : scores.Shape[0] * scores.Shape[1].Fixed * perClass;
        string count = NewDynamicSymbol("nms", bound);
        var attributes = Attributes.From(("count_symbol", count), ("max_per_class", perClass), ("iou_threshold", (float)iou),
            ("score_threshold", (float)score), ("center_point_box", n.Int("center_point_box", 0) != 0));
        SetDyn(n, 0, Emit("NonMaxSuppression", [boxes, scores], attributes, Nm(n)));
    }

    /// <summary>NonZero → IR <c>[count, rank]</c> (see <see cref="TranslatedTensor.TransposedNonZero"/>).</summary>
    private void NonZero(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = ins[0]!.V ?? AsValue(ins[0]!);
        if (ins[0]!.LeadingUnits > 0 || x.Rank == 0)
        {
            throw new NotSupportedException("NonZero of a scalar or of a tensor with leading unit dims");
        }

        if (x.Rank == 1 && IsNonZeroFill(x))
        {
            // NonZero(ConstantOfShape([n], c != 0)) is the exporters' arange(n): every index, without a data-dependent count.
            var attributes = Attributes.From(("shape", new[] { 0L, 1L }), ("outer_symbol", x.Shape[0].Symbol!), ("start", 0.0), ("delta", 1.0),
                ("type", DataType.I32.Name));
            SetDyn(n, 0, Emit("Fill", [], attributes, Nm(n)));
        }
        else
        {
            var bound = Sym(x) ? x.Shape[0] * x.Shape.InnerElements : (Dim)x.Shape.Elements();
            string count = NewDynamicSymbol("nonzero", bound);
            SetDyn(n, 0, Emit("NonZero", [x], Attributes.From(("count_symbol", count)), Nm(n)));
        }

        _env[n.Outputs[0]].TransposedNonZero = true;
    }

    /// <summary>A symbolic Fill of a non-zero constant (through casts).</summary>
    private static bool IsNonZeroFill(Value x)
    {
        var producer = x.Producer;
        while (producer is { Op.Name: "Cast" })
        {
            producer = producer.Inputs[0].Producer;
        }

        return x.Shape[0].IsSymbolic && producer is { Op.Name: "Fill" } fill && fill.Attributes.Get("delta", 0.0) == 0 && fill.Attributes.Get("start", 0.0) != 0;
    }

    /// <summary>Compress(x, condition, axis) = Gather(x, NonZero(condition), axis); without axis over the flattened x.</summary>
    private void Compress(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = ins[0]!;
        var conditionTensor = ins[1]!;
        long axis = 0;
        if (n.Has("axis"))
        {
            axis = n.Int("axis", 0);
        }
        else
        {
            var flat = TranslatedTensor.Const(OnnxConst.FromInts(OnnxType.Int64, [1], [-1]));
            x = Synthetic("Reshape", [x, flat]);
            conditionTensor = Synthetic("Reshape", [conditionTensor, flat]);
        }

        var condition = conditionTensor.V ?? AsValue(conditionTensor);
        var bound = Sym(condition) ? condition.Shape[0] * condition.Shape.InnerElements : (Dim)condition.Shape.Elements();
        string count = NewDynamicSymbol("compress", bound);
        var rows = Emit("NonZero", [condition], Attributes.From(("count_symbol", count)));
        var indices = Rs(rows, 0);
        _env[n.Outputs[0]] = Synthetic("Gather", [x, TranslatedTensor.Dyn(indices)], IntAttribute("axis", axis));
    }

    /// <summary>Unique over the flattened input (outputs: values, first indices, inverse indices, counts).</summary>
    private void Unique(OnnxNode n, TranslatedTensor?[] ins)
    {
        if (n.Has("axis"))
        {
            throw new NotSupportedException("Unique along an axis");
        }

        var x = ins[0]!.V ?? AsValue(ins[0]!);
        var bound = Sym(x) ? x.Shape[0] * x.Shape.InnerElements : (Dim)Math.Max(1, x.Shape.Elements());
        string count = NewDynamicSymbol("unique", bound);
        var outputs = _b.Emit("Unique", [x], Attributes.From(("count_symbol", count), ("sorted", n.Int("sorted", 1) != 0)), Nm(n));
        for (int i = 0; i < n.Outputs.Count && i < outputs.Count; i++)
        {
            if (n.Outputs[i] != "")
            {
                SetDyn(n, i, outputs[i]);
            }
        }
    }

    /// <summary>Consumers of a NonZero result, whose IR value is the transpose of the ONNX tensor.</summary>
    private void NonZeroConsumer(OnnxNode n, TranslatedTensor?[] ins)
    {
        var t = ins[0]!;
        if (!t.TransposedNonZero || ins.Skip(1).Any(i => i is { TransposedNonZero: true }))
        {
            throw new NotSupportedException($"{n.OpType} of a NonZero result (only Transpose, Gather of a row, Squeeze, Shape, Cast and Identity are supported)");
        }

        var v = t.V!;   // [count, rank]
        long rank = v.Shape[1].Fixed;
        switch (n.OpType)
        {
            case "Transpose" when n.Ints("perm") is null or [1, 0]:
                SetDyn(n, 0, v);
                return;
            case "Identity" or "Cast":
                _env[n.Outputs[0]] = t;
                return;
            case "Shape":
                _env[n.Outputs[0]] = TranslatedTensor.Const(OnnxConst.FromInts(OnnxType.Int64, [2], [rank, SymbolicInts.Encode(v.Shape[0])]));
                return;
            case "Squeeze" when rank == 1:
                SetDyn(n, 0, Rs(v, 0));
                return;
            case "Gather" when NormalizeAxis(n.Int("axis", 0), 2) == 0 && ins[1] is { IsConst: true, C.Count: 1 } index:
            {
                long row = index.C!.Ints()[0];
                var column = Rs(_b.Slice(v, 1, row < 0 ? row + rank : row, 1), 0);
                SetDyn(n, 0, column);
                _env[n.Outputs[0]].LeadingUnits = index.C.Rank;   // a [1] index keeps a unit dim in front of the count
                return;
            }

            default:
                throw new NotSupportedException($"{n.OpType} of a NonZero result (only Transpose, Gather of a row, Squeeze, Shape, Cast and Identity are supported)");
        }
    }
}
