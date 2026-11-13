using Sharplight.Graph;

namespace Sharplight.Cpu;

// Ops with symbolic or data-dependent output sizes: the data-dependent ones publish their count symbol when they run.
internal sealed unsafe partial class CpuCompiledSegment
{
    private void LowerFill(Node n)
    {
        var y = NewDense(n.Output);
        AddStep(new FillOp(y, n.Attributes.Get("start", 0.0), n.Attributes.Get("delta", 0.0)), y);
        _refs[n.Output] = y;
    }

    private void LowerNonZero(Node n)
    {
        var x = Dense(Ref(n.Inputs[0]));
        var y = NewDense(n.Output);
        AddStep(new NonZeroOp(x, y, n.Attributes.Get<string>("count_symbol")), x, y);
        _refs[n.Output] = y;
    }

    private void LowerNonMaxSuppression(Node n)
    {
        var a = n.Attributes;
        var boxes = Dense(RefF32(n.Inputs[0]));
        var scores = Dense(RefF32(n.Inputs[1]));
        var y = NewDense(n.Output);
        var op = new NonMaxSuppressionOp(boxes, scores, y, a.Get<string>("count_symbol"), a.Get<long>("max_per_class"), a.Get("iou_threshold", 0f),
            a.Get("score_threshold", float.NegativeInfinity), a.Get("center_point_box", false));
        AddStep(op, boxes, scores, y);
        _refs[n.Output] = y;
    }

    private void LowerUnique(Node n)
    {
        var x = Dense(Ref(n.Inputs[0]));
        var outputs = n.Outputs.Select(NewDense).ToArray();
        AddStep(new UniqueOp(x, outputs[0], outputs[1], outputs[2], outputs[3], n.Attributes.Get<string>("count_symbol"), n.Attributes.Get("sorted", true)), [x, .. outputs]);
        for (int i = 0; i < outputs.Length; i++)
        {
            _refs[n.Outputs[i]] = outputs[i];
        }
    }
}
