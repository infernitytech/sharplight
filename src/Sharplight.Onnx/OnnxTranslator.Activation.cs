using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// Newer elementwise and sequence operators: Swish (alpha), SwiGLU, Shrink, CumProd, ReverseSequence.
internal sealed partial class OnnxTranslator
{
    /// <summary>Swish (opset 24): x · sigmoid(alpha · x); alpha 1 is the IR Silu.</summary>
    private void Swish(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        SetDyn(n, 0, SwishOf(x, n.Float("alpha", 1f), Nm(n)));
    }

    private Value SwishOf(Value x, float alpha, string? name = null)
        => alpha == 1f ? _b.Unary("Silu", x, name) : _b.Mul(x, _b.Sigmoid(_b.Scale(x, alpha)), name);

    /// <summary>SwiGLU (opset 28): Swish_alpha(A) · B for A and B of equal shape.</summary>
    private void SwiGlu(OnnxNode n, TranslatedTensor?[] ins)
    {
        SetDyn(n, 0, _b.Mul(SwishOf(X(ins, 0), n.Float("alpha", 1f)), X(ins, 1), Nm(n)));
    }

    /// <summary>
    /// CausalConvWithState (opset 27): depthwise causal conv of input [B, C, L] with weight [C, 1, K] over [past_state (or zeros), input],
    /// optional bias [C] and SiLU; output #1 is the last K-1 padded positions. Composed as K shifted multiply-adds (K is small).
    /// </summary>
    private void CausalConvWithState(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        var weight = X(ins, 1);
        long channels = x.Shape[1].Fixed;
        long length = x.Shape[2].Fixed;
        long k = weight.Shape[2].Fixed;
        var taps = Rs(weight, channels, k);
        var padded = x;
        if (k > 1)
        {
            padded = Opt(ins, 3) is { } state
                ? Emit("Concat", [AsValue(state), x], Attributes.From(("axis", 2L)))
                : Emit("Pad", [x], Attributes.From(("pads", L(0, 0, k - 1, 0, 0, 0)), ("mode", "constant"), ("value", 0f)));
        }

        Value? y = null;
        for (long j = 0; j < k; j++)
        {
            var term = _b.Mul(_b.Slice(padded, 2, j, length), _b.Slice(taps, 1, j, 1));
            y = y is null ? term : _b.Add(y, term);
        }

        if (Opt(ins, 2) is { } bias)
        {
            y = _b.Add(y!, Rs(AsValue(bias), channels, 1));
        }

        string activation = n.Str("activation", "none");
        y = activation switch
        {
            "none" => y!,
            "silu" or "swish" => _b.Silu(y!),
            _ => throw new NotSupportedException($"CausalConvWithState activation '{activation}'"),
        };
        SetDyn(n, 0, y);
        if (n.Outputs.Count > 1 && n.Outputs[1] != "")
        {
            if (k == 1)
            {
                throw new NotSupportedException("CausalConvWithState with kernel size 1 has an empty present_state");
            }

            SetDyn(n, 1, _b.Slice(padded, 2, length, k - 1));
        }
    }

    /// <summary>Shrink:x + bias where x &lt; -lambd, x - bias where x &gt; lambd, 0 in between.</summary>
    private void Shrink(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        float lambd = n.Float("lambd", 0.5f);
        float bias = n.Float("bias", 0f);
        var zero = ConstValue(OnnxConst.Scalar(0f), Nm(n) + ".zero");
        var upper = Emit("Where", [_b.Binary("Greater", x, ConstValue(OnnxConst.Scalar(lambd), Nm(n) + ".lambd")), _b.Scale(x, 1f, -bias), zero]);
        var below = _b.Binary("Less", x, ConstValue(OnnxConst.Scalar(-lambd), Nm(n) + ".neg_lambd"));
        SetDyn(n, 0, Emit("Where", [below, _b.Scale(x, 1f, bias), upper], null, Nm(n)));
    }

    /// <summary>CumProd (opset 26): the IR CumSum in product mode.</summary>
    private void CumProd(OnnxNode n, TranslatedTensor?[] ins)
    {
        var attributes = Attributes.From(("axis", ConstIn(ins, 1, "CumProd axis").ScalarInt()), ("exclusive", n.Int("exclusive", 0) != 0),
            ("reverse", n.Int("reverse", 0) != 0), ("product", true));
        SetDyn(n, 0, Emit("CumSum", [X(ins, 0)], attributes, Nm(n)));
    }

    /// <summary>
    /// ReverseSequence: reverses the first sequence_lens[b] entries along time_axis for every batch entry b. Built as one gather along
    /// the time axis with indices t &lt; len ? len - 1 - t : t computed from the (possibly runtime) lengths.
    /// </summary>
    private void ReverseSequence(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        int batchAxis = (int)n.Int("batch_axis", 1);
        int timeAxis = (int)n.Int("time_axis", 0);
        if (Sym(x) || batchAxis + timeAxis != 1)
        {
            throw new NotSupportedException("ReverseSequence needs a fixed shape and batch/time axes 0 and 1");
        }

        long steps = x.Shape[timeAxis].Fixed;
        long batch = x.Shape[batchAxis].Fixed;
        long[] timeShape = timeAxis == 0 ? [steps, 1] : [1, steps];
        long[] batchShape = batchAxis == 0 ? [batch, 1] : [1, batch];
        var time = _b.Constant(Nm(n) + ".time", DataType.F32, new Shape(timeShape.Select(d => (Dim)d).ToArray()),
            ToBytes([.. Enumerable.Range(0, (int)steps).Select(t => (float)t)]));
        var lengths = Rs(AsValue(ins[1]!), batchShape);
        var reversed = _b.Sub(_b.Scale(lengths, 1f, -1f), time);
        var index = Emit("Where", [_b.Binary("Less", time, lengths), reversed, time]);
        long[] full = Fixed(x);
        index = Rs(_b.Emit1("Cast", [index], Attributes.From(("type", "i32"))), [.. full.Take(2), .. Enumerable.Repeat(1L, x.Rank - 2)]);
        if (x.Rank > 2)
        {
            index = Emit("Expand", [index], Attributes.From(("shape", full)));
        }

        SetDyn(n, 0, Emit("GatherElements", [x, index], Attributes.From(("axis", (long)timeAxis)), Nm(n)));
    }
}
