using Sharplight.Graph;
using Sharplight.Tensors;

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
        ["IsInf"] = "IsInf",
        ["Silu"] = "Silu",
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
        SetDyn(n, 0, Emit(irOp, [X(ins, 0)], null, Nm(n)));
    }

    private void UnaryWithAttributes(OnnxNode n, TranslatedTensor?[] ins, (string Ir, (string Name, float Def)[] Attrs) spec)
    {
        var attributes = Attributes.From(spec.Attrs.Select(a => (a.Name, (object)n.Float(a.Name, a.Def))).ToArray());
        SetDyn(n, 0, Emit(spec.Ir, [X(ins, 0)], attributes, Nm(n)));
    }

    private void Binary(OnnxNode n, TranslatedTensor?[] ins, string irOp)
    {
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
        SetDyn(n, 0, Emit("Mod", [X(ins, 0), X(ins, 1)], attributes, Nm(n)));
    }

    /// <summary>Clip with attribute or constant bounds (the IR Clip op); runtime bounds become Max / Min against the bound value.</summary>
    private void Clip(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        var (minConst, minValue) = ClipBound(n, ins, "min", 1);
        var (maxConst, maxValue) = ClipBound(n, ins, "max", 2);
        if (minValue is null && maxValue is null)
        {
            var bounds = new List<(string, object)>();
            if (minConst is { } lower)
            {
                bounds.Add(("min", lower));
            }

            if (maxConst is { } upper)
            {
                bounds.Add(("max", upper));
            }

            SetDyn(n, 0, Emit("Clip", [x], Attributes.From(bounds.ToArray()), Nm(n)));
            return;
        }

        // ONNX applies the lower bound first, then the upper one (min > max yields max everywhere). Integer tensors are
        // clipped in f32 (exact for the ranges i32 carries here) because the CPU Max / Min kernels are f32-only.
        bool integer = x.Type == DataType.I32;
        Value ToF32(Value v) => v.Type == DataType.I32 ? Emit("Cast", [v], Attributes.From(("type", "f32"))) : v;
        var y = ToF32(x);
        minValue = minValue is null ? null : ToF32(minValue);
        maxValue = maxValue is null ? null : ToF32(maxValue);
        if (minConst is { } constLower)
        {
            y = Emit("Clip", [y], Attributes.From(("min", constLower)));
        }

        if (minValue is not null)
        {
            y = Emit("Max", [y, minValue]);
        }

        if (maxConst is { } constUpper)
        {
            y = Emit("Clip", [y], Attributes.From(("max", constUpper)));
        }

        if (maxValue is not null)
        {
            y = Emit("Min", [y, maxValue]);
        }

        SetDyn(n, 0, integer ? Emit("Cast", [y], Attributes.From(("type", "i32")), Nm(n)) : y);
    }

    /// <summary>A Clip bound: an attribute or constant input, or a runtime value.</summary>
    private static (float? Constant, Value? Runtime) ClipBound(OnnxNode n, TranslatedTensor?[] ins, string attribute, int input)
    {
        if (n.Has(attribute))
        {
            return (n.Float(attribute, 0), null);
        }

        if (ins.Length <= input || ins[input] is not { } bound)
        {
            return (null, null);
        }

        return bound.IsConst ? (bound.C!.ScalarF32(), null) : (null, bound.V!);
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

    /// <summary>
    /// Dropout in inference: y = x and the mask (output #1) all true. Training mode (opset 12+: a constant training_mode input) is the
    /// identity only for ratio 0; random dropout is not supported.
    /// </summary>
    private void Dropout(OnnxNode n, TranslatedTensor?[] ins)
    {
        bool training = OptConst(ins, 2, "Dropout training_mode") is { } mode && mode.Ints()[0] != 0;
        float ratio = OptConst(ins, 1, "Dropout ratio") is { } r ? r.ScalarF32() : n.Float("ratio", 0.5f);
        if (training && ratio != 0)
        {
            throw new NotSupportedException("Dropout in training mode (random masks are not reproducible)");
        }

        _env[n.Outputs[0]] = ins[0]!;
        if (n.Outputs.Count > 1 && n.Outputs[1] != "")
        {
            var x = X(ins, 0);
            var one = ConstValue(OnnxConst.Scalar(1f), Nm(n) + ".one");
            SetDyn(n, 1, Emit("Where", [_b.Unary("IsNaN", x), one, one]));
        }
    }

    private void Cast(OnnxNode n, TranslatedTensor?[] ins)
    {
        int to = (int)n.Int("to", OnnxType.Float);
        var x = ins[0]!;
        Value v = x.V ?? AsValue(x);
        if (to == OnnxType.Bool)
        {
            v = Emit("NotEqual", [AsValue(x), _b.Constant("zero", DataType.F32, new Shape(1), new byte[4])], null, Nm(n));
        }
        else if (OnnxType.IsFloat(to))
        {
            v = x.V is { Type.Kind: DataTypeKind.Int } ? AsValue(x) : v;
        }
        else if (v.Type == DataType.F32)
        {
            v = Emit("Cast", [v], Attributes.From(("type", "i32")), Nm(n));
        }

        _env[n.Outputs[0]] = TranslatedTensor.Dyn(v);
    }

    /// <summary>CastLike is a Cast to the element type of the second input (float when that input is dynamic).</summary>
    private void CastLike(OnnxNode n, TranslatedTensor?[] ins)
    {
        var like = ins[1]!;
        int to = like.IsConst ? like.C!.ElemType : OnnxType.Float;
        var toAttribute = new OnnxAttribute { Name = "to", Type = OnnxAttrType.Int, I = to };
        var cast = new OnnxNode
        {
            OpType = "Cast",
            Inputs = n.Inputs.Take(1).ToList(),
            Outputs = n.Outputs,
            Attributes = new Dictionary<string, OnnxAttribute> { ["to"] = toAttribute },
        };
        Cast(cast, ins);
    }
}
