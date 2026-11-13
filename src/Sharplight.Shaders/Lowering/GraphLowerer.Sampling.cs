using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

/// <summary>Lowering of <see cref="StandardOps.Sample"/>: one workgroup per logits row (<c>sample.slang</c>).</summary>
public sealed partial class GraphLowerer
{
    /// <summary>Threads per workgroup of the sampling kernel (the CPU reference mirrors its candidate order).</summary>
    private const int SampleWorkgroup = 256;

    private void LowerSample(Node n)
    {
        var x = AsDenseF32(Ref(n.Inputs[0]), "sample");
        var parameters = Ref(n.Inputs[1]);
        var tokens = Produce(n.Outputs[0]);
        var logprobs = Produce(n.Outputs[1]);
        var k = Lib("sample.slang", "sample", Macros(("V", x.Shape[^1]), ("WG", SampleWorkgroup), ("NC", SamplingHead.MaxNucleusCandidates), ("VEC", x.Shape[^1] % 4 == 0)));
        var tx = x;
        Emit($"sample {n.Outputs[0].Name}", [n], [Always(k, b => new Grid((uint)Rows(b, tx)))],
            [Pointer(x), Pointer(parameters), Pointer(tokens), Pointer(logprobs), TokenArg(x)], [x, parameters], [tokens, logprobs]);
    }
}
