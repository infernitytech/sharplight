using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// String tensors: graph inputs of ONNX type string become DataType.Str values (handles into the run's string heap), string constants become
// StringConst nodes where a runtime value is needed, and the string operators map to the host-only IR ops of StandardOps.Text.
internal sealed partial class OnnxTranslator
{
    private static readonly HashSet<string> s_stringOps = ["StringNormalizer", "StringConcat", "RegexFullMatch"];

    // Operators of other categories that take string inputs themselves.
    private static readonly HashSet<string> s_stringInputOps = ["TfIdfVectorizer"];

    private static bool IsStringTensor(TranslatedTensor? t) => t is not null && (t.V?.Type == DataType.Str || t.C?.Strings is not null);

    /// <summary>String operators, and any node with a string input or a string result type.</summary>
    private static bool IsStringNode(OnnxNode n, TranslatedTensor?[] ins)
        => s_stringOps.Contains(n.OpType) || (ins.Any(IsStringTensor) && !s_stringInputOps.Contains(n.OpType))
            || (n.OpType == "Cast" && n.Int("to", 0) == OnnxType.String) || (n.OpType == "CastLike" && ins.Length > 1 && IsStringTensor(ins[1]));

    private void TranslateStrings(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        switch (n.OpType)
        {
            case "StringConcat":
                SetDyn(n, 0, Emit("StringBinary", [StrValue(ins[0]!, name + ".a"), StrValue(ins[1]!, name + ".b")], Attributes.From(("mode", "concat")), name));
                break;
            case "Equal":
                SetDyn(n, 0, Emit("StringBinary", [StrValue(ins[0]!, name + ".a"), StrValue(ins[1]!, name + ".b")], Attributes.From(("mode", "equal")), name));
                break;
            case "RegexFullMatch":
                SetDyn(n, 0, Emit("RegexMatch", [StrValue(ins[0]!, name + ".x")], Attributes.From(("pattern", n.Str("pattern", ""))), name));
                break;
            case "StringNormalizer":
                StringNormalizer(n, ins);
                break;
            case "Identity":
                SetDyn(n, 0, StrValue(ins[0]!, name));
                break;
            case "Cast" or "CastLike":
                {
                    int to = n.OpType == "Cast" ? (int)n.Int("to", 0) : IsStringTensor(ins[1]) ? OnnxType.String : ins[1]!.C?.ElemType ?? OnnxType.Float;
                    SetDyn(n, 0, StringCast(ins[0]!, to, name));
                    break;
                }

            default:
                throw new NotSupportedException($"{n.OpType} on string tensors");
        }
    }

    /// <summary>
    /// ONNX StringNormalizer on <c>[C]</c> (or <c>[1, C]</c>, carried as <c>[C]</c> with a leading unit dim): the number of kept strings is
    /// a dynamic symbol bounded by C.
    /// </summary>
    private void StringNormalizer(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = StrValue(ins[0]!, Nm(n) + ".x");
        bool leadingUnit = x.Rank == 2 && !x.Shape[0].IsSymbolic && x.Shape[0].Fixed == 1;
        if (leadingUnit)
        {
            x = Rs(x, x.Shape[1].Fixed);
        }
        else if (x.Rank != 1 || x.Shape[0].IsSymbolic)
        {
            throw new NotSupportedException($"StringNormalizer of shape {x.Shape} (expects [C] or [1, C])");
        }

        string count = NewDynamicSymbol("strnorm", (Dim)Math.Max(1, x.Shape[0].Fixed));
        var stopwords = n.Attributes.TryGetValue("stopwords", out var s) ? s.Strings.ToArray() : [];
        var attributes = Attributes.From(
            ("count_symbol", count), ("stopwords", stopwords), ("case_sensitive", n.Int("is_case_sensitive", 0) != 0),
            ("case", n.Str("case_change_action", "NONE")));
        SetDyn(n, 0, Emit("StringNormalize", [x], attributes, Nm(n)));
        _env[n.Outputs[0]].LeadingUnits = leadingUnit ? 1 : 0;
    }

    /// <summary>Cast from or to strings: numbers format with the shortest round-trip text, strings parse with the invariant culture.</summary>
    private Value StringCast(TranslatedTensor x, int to, string name)
    {
        if (IsStringTensor(x))
        {
            if (to == OnnxType.String)
            {
                return StrValue(x, name);
            }

            string type = OnnxType.IsFloat(to) ? "f32" : OnnxType.IsInteger(to) ? "i32"
                : throw new NotSupportedException($"Cast from string to {OnnxType.Name(to)}");
            return Emit("StringCast", [StrValue(x, name + ".x")], Attributes.From(("type", type)), name);
        }

        var source = x.V is { Type.Kind: DataTypeKind.Int } || x.C is { IsFloat: false } ? Idx(x) : AsValue(x);
        return Emit("StringCast", [source], Attributes.From(("type", "str")), name);
    }

    /// <summary>The str value of a string tensor: the runtime value, or a StringConst node for a constant.</summary>
    private Value StrValue(TranslatedTensor t, string name)
    {
        if (t.V is { } v)
        {
            return v.Type == DataType.Str ? v : throw new NotSupportedException($"expected a string tensor, got {v.Type}");
        }

        var c = t.C!;
        var strings = c.Strings ?? throw new NotSupportedException($"expected a string tensor, got {OnnxType.Name(c.ElemType)}");
        return Emit("StringConst", [], Attributes.From(("strings", strings), ("shape", c.Rank == 0 ? new long[] { 1 } : c.Dims)), name);
    }
}
