using Sharplight.Graph;

namespace Sharplight.Cpu;

// Classical machine learning (decision forests, kernel machines).
internal sealed unsafe partial class CpuCompiledSegment
{
    private void LowerTreeEnsemble(Node n)
    {
        var x = Dense(RefF32(n.Inputs[0]));
        var y = NewDense(n.Outputs[0]);
        var hits = NewDense(n.Outputs[1]);
        AddStep(new TreeEnsembleOp(x, y, hits, n.Attributes), x, y, hits);
        _refs[n.Outputs[0]] = y;
        _refs[n.Outputs[1]] = hits;
    }

    private void LowerLabelEncode(Node n)
    {
        var a = n.Attributes;
        var x = Dense(Ref(n.Inputs[0]));
        var y = NewDense(n.Output);
        var op = new LabelEncodeOp(x, y, Objects(a["keys"]), Objects(a["values"]), a.TryGetValue("default", out var fallback) ? fallback : 0.0);
        AddStep(op, x, y);
        _refs[n.Output] = y;
    }

    private static object[] Objects(object array) => array switch
    {
        double[] numbers => [.. numbers.Cast<object>()],
        string[] strings => strings,
        _ => throw new ArgumentException($"LabelEncode keys and values are double[] or string[], got {array.GetType().Name}."),
    };

    private void LowerOneVsOne(Node n)
    {
        var a = n.Attributes;
        var d = Dense(RefF32(n.Inputs[0]));
        var winner = NewDense(n.Outputs[0]);
        var probabilities = n.Outputs.Length > 1 ? NewDense(n.Outputs[1]) : null;
        var op = new OneVsOneOp(d, winner, probabilities, (int)a.Get<long>("classes"), a.Get("prob_a", Array.Empty<float>()), a.Get("prob_b", Array.Empty<float>()));
        AddStep(op, probabilities is null ? [d, winner] : [d, winner, probabilities]);
        _refs[n.Outputs[0]] = winner;
        if (probabilities is not null)
        {
            _refs[n.Outputs[1]] = probabilities;
        }
    }
}
