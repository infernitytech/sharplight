using Sharplight.Graph;

namespace Sharplight.Cpu;

// Text features over token ids.
internal sealed unsafe partial class CpuCompiledSegment
{
    private void LowerTfIdf(Node n)
    {
        var a = n.Attributes;
        var x = Dense(Ref(n.Inputs[0]));
        var y = NewDense(n.Output);
        var op = new TfIdfOp(x, y, (int)a.Get<long>("min_gram"), (int)a.Get<long>("max_gram"), (int)a.Get<long>("max_skip"), a.Get<long[]>("pool"),
            a.Get<long[]>("ngram_counts"), a.Get<long[]>("ngram_indexes"), a.Get("weights", Array.Empty<float>()), a.Get<string>("mode"),
            (int)a.Get<long>("out_features"));
        AddStep(op, x, y);
        _refs[n.Output] = y;
    }
}
