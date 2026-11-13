using Sharplight.Graph;

namespace Sharplight.Cpu;

// Gather/scatter, scans and other index-driven ops.
internal sealed unsafe partial class CpuCompiledSegment
{
    private void LowerGatherAxis(Node n)
    {
        var a = n.Attributes;
        var x = Dense(Ref(n.Inputs[0]));
        var indices = Dense(Ref(n.Inputs[1]));
        var y = NewDense(n.Output);
        AddStep(new GatherAxisOp(x, indices, y, NormalizeAxis(a.Get("axis", 0L), x.Rank)), x, indices, y);
        _refs[n.Output] = y;
    }

    private void LowerGatherElements(Node n)
    {
        var a = n.Attributes;
        var x = Dense(Ref(n.Inputs[0]));
        var indices = Dense(Ref(n.Inputs[1]));
        var y = NewDense(n.Output);
        AddStep(new GatherElementsOp(x, indices, y, NormalizeAxis(a.Get("axis", 0L), x.Rank)), x, indices, y);
        _refs[n.Output] = y;
    }

    private void LowerGatherNd(Node n)
    {
        var x = Dense(Ref(n.Inputs[0]));
        var indices = Dense(Ref(n.Inputs[1]));
        var y = NewDense(n.Output);
        AddStep(new GatherNdOp(x, indices, y, (int)n.Attributes.Get("batch_dims", 0L)), x, indices, y);
        _refs[n.Output] = y;
    }

    /// <summary>ScatterElements (along an axis) and ScatterND (whole index tuples).</summary>
    private void LowerScatter(Node n)
    {
        var a = n.Attributes;
        var ins = n.Inputs;
        bool isNd = n.Op.Name == "ScatterND";
        var x = Dense(Ref(ins[0]));
        var indices = Dense(Ref(ins[1]));
        var updates = Dense(Ref(ins[2]));
        var y = NewDense(n.Output);
        int axis = isNd ? 0 : NormalizeAxis(a.Get("axis", 0L), x.Rank);
        AddStep(new ScatterOp(x, indices, updates, y, axis, isNd, a.Get("reduction", "none")), x, indices, updates, y);
        _refs[n.Output] = y;
    }

    private void LowerSequencePool(Node n)
    {
        var x = RowContig(RefF32(n.Inputs[0]));
        var spans = Dense(Ref(n.Inputs[1]));
        var y = NewDense(n.Output);
        AddStep(new SequencePoolOp(x, spans, y, n.Attributes.Get("mode", "mean")), x, spans, y);
        _refs[n.Output] = y;
    }

    private void LowerNGramHash(Node n)
    {
        var a = n.Attributes;
        var window = RefF32(n.Inputs[0]);
        var y = NewDense(n.Output);
        AddStep(new NGramHashOp(window, y, a.Get<long[]>("multipliers"), a.Get<long[]>("head_sizes"), a.Get<long[]>("head_offsets"),
            (int)a.Get<long>("heads_per_order"), a.Get<long>("stop_token")), window, y);
        _refs[n.Output] = y;
    }

    private void LowerCumSum(Node n)
    {
        var a = n.Attributes;
        var x = Dense(RefF32(n.Inputs[0]));
        var y = NewDense(n.Output);
        AddStep(new CumSumOp(x, y, NormalizeAxis(a.Get("axis", 0L), x.Rank), a.Get("exclusive", false), a.Get("reverse", false)), x, y);
        _refs[n.Output] = y;
    }

    private void LowerArgReduce(Node n)
    {
        var a = n.Attributes;
        var x = Dense(RefF32(n.Inputs[0]));
        var y = NewDense(n.Output);
        AddStep(new ArgReduceOp(x, y, NormalizeAxis(a.Get("axis", 0L), x.Rank), a.Get("mode", "max") == "max", a.Get("select_last_index", false)), x, y);
        _refs[n.Output] = y;
    }

    private void LowerTopK(Node n)
    {
        var a = n.Attributes;
        var x = Dense(RefF32(n.Inputs[0]));
        var values = NewDense(n.Outputs[0]);
        var indices = NewDense(n.Outputs[1]);
        AddStep(new TopKOp(x, values, indices, NormalizeAxis(a.Get("axis", -1L), x.Rank), a.Get("largest", true)), x, values, indices);
        _refs[n.Outputs[0]] = values;
        _refs[n.Outputs[1]] = indices;
    }

    private void LowerTrilu(Node n)
    {
        var a = n.Attributes;
        var x = Dense(Ref(n.Inputs[0]));
        var y = NewDense(n.Output);
        AddStep(new TriluOp(x, y, a.Get("upper", true), a.Get("k", 0L)), x, y);
        _refs[n.Output] = y;
    }

    private void LowerOneHot(Node n)
    {
        var a = n.Attributes;
        var indices = Dense(Ref(n.Inputs[0]));
        var y = NewDense(n.Output);
        AddStep(new OneHotOp(indices, y, NormalizeAxis(a.Get("axis", -1L), indices.Rank + 1), a.Get<long>("depth"), a.Get("on", 1f), a.Get("off", 0f)), indices, y);
        _refs[n.Output] = y;
    }
}
