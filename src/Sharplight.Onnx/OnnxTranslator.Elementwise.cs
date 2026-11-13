using Sharplight.Graph;

namespace Sharplight.Onnx;

// Elementwise, variadic and cast operators.
internal sealed partial class OnnxTranslator
{
    private static readonly Dictionary<string, string> s_unary = new()
    {
        ["Relu"] = "Relu",
        ["Sigmoid"] = "Sigmoid",
        ["Tanh"] = "Tanh",
        ["Exp"] = "Exp",
        ["Log"] = "Log",
        ["Sqrt"] = "Sqrt",
        ["Neg"] = "Neg",
        ["Abs"] = "Abs",
        ["Erf"] = "Erf",
        ["Reciprocal"] = "Recip",
        ["HardSwish"] = "HardSwish",
        ["Softplus"] = "Softplus",
        ["Mish"] = "Mish",
        ["Sin"] = "Sin",
        ["Cos"] = "Cos",
        ["Tan"] = "Tan",
        ["Asin"] = "Asin",
        ["Acos"] = "Acos",
        ["Atan"] = "Atan",
        ["Sinh"] = "Sinh",
        ["Cosh"] = "Cosh",
        ["Asinh"] = "Asinh",
        ["Acosh"] = "Acosh",
        ["Atanh"] = "Atanh",
        ["Floor"] = "Floor",
        ["Ceil"] = "Ceil",
        ["Round"] = "Round",
        ["Sign"] = "Sign",
        ["Softsign"] = "Softsign",
        ["Not"] = "Not",
        ["IsNaN"] = "IsNaN",
        ["Silu"] = "Silu",
        ["Swish"] = "Silu",
    };

    /// <summary>ONNX unary ops with float attributes: (attribute, default) pairs, same names on the IR op.</summary>
    private static readonly Dictionary<string, (string Ir, (string Name, float Def)[] Attrs)> s_unaryAttr = new()
    {
        ["Elu"] = ("Elu", [("alpha", 1f)]),
        ["Celu"] = ("Celu", [("alpha", 1f)]),
        ["Selu"] = ("Selu", [("alpha", 1.67326319217681884765625f), ("gamma", 1.05070102214813232421875f)]),
        ["LeakyRelu"] = ("LeakyRelu", [("alpha", 0.01f)]),
        ["ThresholdedRelu"] = ("ThresholdedRelu", [("alpha", 1f)]),
        ["HardSigmoid"] = ("HardSigmoid", [("alpha", 0.2f), ("beta", 0.5f)]),
    };

    private static readonly Dictionary<string, string> s_binary = new()
    {
        ["Add"] = "Add",
        ["Sub"] = "Sub",
        ["Mul"] = "Mul",
        ["Div"] = "Div",
        ["Pow"] = "Pow",
        ["PRelu"] = "PRelu",
        ["Equal"] = "Equal",
        ["Less"] = "Less",
        ["Greater"] = "Greater",
        ["LessOrEqual"] = "LessEqual",
        ["GreaterOrEqual"] = "GreaterEqual",
        ["And"] = "And",
        ["Or"] = "Or",
        ["Xor"] = "Xor",
    };

    private Value Act(string? name, Value v, float[]? p)
    {
        switch (name)
        {
            case null:
            case "":
            case "Identity":
                return v;
            case "Relu":
                return _b.Unary("Relu", v);
            case "Sigmoid":
                return _b.Unary("Sigmoid", v);
            case "Tanh":
                return _b.Unary("Tanh", v);
            case "LeakyRelu":
                return Emit("LeakyRelu", [v], Attributes.From(("alpha", p is { Length: > 0 } ? p[0] : 0.01f)));
            case "Clip":
                return Emit("Clip", [v], Attributes.From(("min", p![0]), ("max", p[1])));
            case "HardSigmoid":
                return Emit("HardSigmoid", [v], Attributes.From(("alpha", p is { Length: > 0 } ? p[0] : 0.2f), ("beta", p is { Length: > 1 } ? p[1] : 0.5f)));
            default:
                throw new NotSupportedException($"activation '{name}'");
        }
    }

    private void Unary(OnnxNode n, TranslatedTensor?[] ins, string irOp)
    {
        if (s_integerUnary.Contains(n.OpType) && AllIntegers(ins))
        {
            IntegerUnary(n, ins[0]!, irOp);
            return;
        }

        SetDyn(n, 0, Emit(irOp, [X(ins, 0)], null, Nm(n)));
    }

    private void UnaryWithAttributes(OnnxNode n, TranslatedTensor?[] ins, (string Ir, (string Name, float Def)[] Attrs) spec)
    {
        var attributes = Attributes.From(spec.Attrs.Select(a => (a.Name, (object)n.Float(a.Name, a.Def))).ToArray());
        SetDyn(n, 0, Emit(spec.Ir, [X(ins, 0)], attributes, Nm(n)));
    }

    private void Binary(OnnxNode n, TranslatedTensor?[] ins, string irOp)
    {
        if (s_integerBinary.Contains(n.OpType) && AllIntegers(ins))
        {
            IntegerBinary(n, ins, irOp);
            return;
        }

        string name = Nm(n);
        SetDyn(n, 0, Emit(irOp, BroadcastOperands(ins, X(ins, 0, name + ".a"), X(ins, 1, name + ".b")), null, name));
    }

    /// <summary>
    /// Operands of a broadcasting op where an input has leading unit dims (<see cref="TranslatedTensor.LeadingUnits"/>): the matching unit
    /// dims of the other operands are dropped as well, so that the IR broadcast aligns like the ONNX one.
    /// </summary>
    private Value[] BroadcastOperands(TranslatedTensor?[] ins, params Value[] operands)
    {
        int lead = ins.Max(t => t?.LeadingUnits ?? 0);
        if (lead == 0)
        {
            return operands;
        }

        int irRank = ins.Where(t => t is { LeadingUnits: > 0 }).Max(t => t!.V!.Rank);
        for (int i = 0; i < operands.Length; i++)
        {
            var v = operands[i];
            if (ins[i] is { LeadingUnits: 0 } && v.Rank > irRank)
            {
                int drop = v.Rank - irRank;
                if (v.Shape.Dims.Take(drop).Any(d => d.IsSymbolic || d.Fixed != 1))
                {
                    throw new NotSupportedException("broadcasting a tensor with leading unit dims against one with more outer dims");
                }

                operands[i] = Rs(v, v.Shape.Dims.Skip(drop).Select(d => d.IsSymbolic ? 0 : d.Fixed).ToArray());
            }
        }

        return operands;
    }

    private void Mod(OnnxNode n, TranslatedTensor?[] ins)
    {
        var attributes = Attributes.From(("fmod", n.Int("fmod", 0) != 0));
        if (AllIntegers(ins))
        {
            IntegerBinary(n, ins, "Mod", attributes);
            return;
        }

        SetDyn(n, 0, Emit("Mod", [X(ins, 0), X(ins, 1)], attributes, Nm(n)));
    }

    private void Clip(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        float? min = n.Has("min") ? n.Float("min", 0) : OptConst(ins, 1, "Clip min") is { } minConst ? minConst.ScalarF32() : null;
        float? max = n.Has("max") ? n.Float("max", 0) : OptConst(ins, 2, "Clip max") is { } maxConst ? maxConst.ScalarF32() : null;
        var bounds = new List<(string, object)>();
        if (min is { } lower)
        {
            bounds.Add(("min", lower));
        }

        if (max is { } upper)
        {
            bounds.Add(("max", upper));
        }

        SetDyn(n, 0, Emit("Clip", [x], Attributes.From(bounds.ToArray()), Nm(n)));
    }

    private void Gelu(OnnxNode n, TranslatedTensor?[] ins)
    {
        string irOp = n.Str("approximate", "none") == "tanh" ? "GeluTanh" : "Gelu";
        SetDyn(n, 0, Emit(irOp, [X(ins, 0)], null, Nm(n)));
    }

    /// <summary>Max, Min, Sum and Mean over any number of inputs, folded pairwise.</summary>
    private void Variadic(OnnxNode n, TranslatedTensor?[] ins)
    {
        string op = n.OpType;
        string irOp = op == "Max" ? "Max" : op == "Min" ? "Min" : "Add";
        bool mean = op == "Mean";
        var acc = X(ins, 0);
        for (int i = 1; i < ins.Length; i++)
        {
            string? name = i == ins.Length - 1 && !mean ? Nm(n) : null;
            acc = Emit(irOp, [acc, X(ins, i)], null, name);
        }

        if (mean)
        {
            acc = Emit("Scale", [acc], Attributes.From(("scale", 1f / ins.Length), ("bias", 0f)), Nm(n));
        }

        SetDyn(n, 0, acc);
    }

    private void Where(OnnxNode n, TranslatedTensor?[] ins)
    {
        SetDyn(n, 0, Emit("Where", [X(ins, 0), X(ins, 1), X(ins, 2)], null, Nm(n)));
    }

    private void Identity(OnnxNode n, TranslatedTensor?[] ins)
    {
        _env[n.Outputs[0]] = ins[0]!;
    }

    private void Dropout(OnnxNode n, TranslatedTensor?[] ins)
    {
        CheckUnusedOutputs(n, 1);
        _env[n.Outputs[0]] = ins[0]!;
    }
}
