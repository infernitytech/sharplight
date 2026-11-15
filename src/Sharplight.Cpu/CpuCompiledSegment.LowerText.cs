using Sharplight.Graph;

namespace Sharplight.Cpu;

// Text: features over token ids and string tensors.
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

    private void LowerStringConst(Node n)
    {
        var y = NewDense(n.Output);
        AddStep(new StringConstOp(y, n.Attributes.Get<string[]>("strings")), y);
        _refs[n.Output] = y;
    }

    private void LowerStringBinary(Node n)
    {
        var a = Dense(Ref(n.Inputs[0]));
        var b = Dense(Ref(n.Inputs[1]));
        var y = NewDense(n.Output);
        AddStep(new StringBinaryOp(a, b, y, n.Attributes.Get<string>("mode") == "equal"), a, b, y);
        _refs[n.Output] = y;
    }

    private void LowerStringNormalize(Node n)
    {
        var a = n.Attributes;
        var x = Dense(Ref(n.Inputs[0]));
        var y = NewDense(n.Output);
        var op = new StringNormalizeOp(x, y, a.Get("stopwords", Array.Empty<string>()), a.Get("case_sensitive", false), a.Get("case", "NONE"),
            a.Get<string>("count_symbol"));
        AddStep(op, x, y);
        _refs[n.Output] = y;
    }

    private void LowerRegexMatch(Node n)
    {
        var x = Dense(Ref(n.Inputs[0]));
        var y = NewDense(n.Output);
        AddStep(new RegexMatchOp(x, y, n.Attributes.Get<string>("pattern")), x, y);
        _refs[n.Output] = y;
    }

    private void LowerStringCast(Node n)
    {
        var x = Dense(Ref(n.Inputs[0]));
        var y = NewDense(n.Output);
        AddStep(new StringCastOp(x, y), x, y);
        _refs[n.Output] = y;
    }

    private void LowerMurmurHash3(Node n)
    {
        var x = Dense(Ref(n.Inputs[0]));
        var y = NewDense(n.Output);
        AddStep(new MurmurHash3Op(x, y, unchecked((uint)n.Attributes.Get("seed", 0L)), (int)n.Attributes.Get("element_bytes", 4L)), x, y);
        _refs[n.Output] = y;
    }

    private void LowerStringSplit(Node n)
    {
        var a = n.Attributes;
        var x = Dense(Ref(n.Inputs[0]));
        var tokens = NewDense(n.Outputs[0]);
        var counts = NewDense(n.Outputs[1]);
        AddStep(new StringSplitOp(x, tokens, counts, a.Get("delimiter", ""), a.Get("maxsplit", -1L), a.Get<string>("count_symbol"), a.Get<long>("max_tokens")),
            [x, tokens, counts]);
        _refs[n.Outputs[0]] = tokens;
        _refs[n.Outputs[1]] = counts;
    }
}
