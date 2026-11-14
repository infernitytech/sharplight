using Sharplight.Graph;

namespace Sharplight.Cpu;

// Random number generation (deterministic, see Runtime.CounterRandom).
internal sealed unsafe partial class CpuCompiledSegment
{
    private void LowerRandom(Node n)
    {
        var a = n.Attributes;
        bool normal = a.Get<string>("distribution") == "normal";
        var y = NewDense(n.Output);
        var op = normal
            ? new RandomOp(y, true, a.Get("mean", 0f), a.Get("scale", 1f), a.Get<long>("seed"))
            : new RandomOp(y, false, a.Get("low", 0f), a.Get("high", 1f), a.Get<long>("seed"));
        AddStep(op, y);
        _refs[n.Output] = y;
    }

    private void LowerMultinomial(Node n)
    {
        var x = Dense(RefF32(n.Inputs[0]));
        var y = NewDense(n.Output);
        AddStep(new MultinomialOp(x, y, n.Attributes.Get<long>("sample_size"), n.Attributes.Get<long>("seed")), x, y);
        _refs[n.Output] = y;
    }
}
