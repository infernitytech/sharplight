using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

// Typed elementwise code: i32 values (exact integer arithmetic, bitwise ops, Cast to i32, Reinterpret) and Narrow. Generated kernels keep
// f32 locals for f32 graphs; i32 tensors are read into int locals, and a node computes in int when all of its operands are int.
public sealed partial class GraphLowerer
{
    // A property (not a field) because s_elementwise in another part of this class reads it during static initialization.
    private static string[] NumericElementwiseNames => ["Narrow", "Reinterpret", "BitNot", .. StandardOps.BitwiseBinaryNames];

    private static readonly HashSet<string> s_integerCapable =
        [.. StandardOps.IntegerPreservingBinaryNames, .. StandardOps.IntegerPreservingUnaryNames, .. StandardOps.BitwiseBinaryNames, "BitNot",
         "Equal", "NotEqual", "Less", "LessEqual", "Greater", "GreaterEqual"];

    /// <summary>Element types the generated elementwise kernels read and write.</summary>
    private static OpSupport SupportsElementwise(Node node)
    {
        if (node.Op.Name is "Cast" or "Where")
        {
            return OpSupport.Yes;   // typed outputs use LowerCastTyped / LowerWhereTyped, which handle every element type
        }

        foreach (var input in node.Inputs)
        {
            var t = input.Type;
            if (!(t == DataType.F32 || t == DataType.F16 || t == DataType.BF16 || t == DataType.I32 || t == DataType.I64 || t == DataType.U32))
            {
                return OpSupport.No($"{node.Op.Name}: elementwise input type {t}");
            }
        }

        var output = node.Output.Type;
        if (!(output == DataType.F32 || output == DataType.I32 || output == DataType.F16 || output == DataType.BF16))
        {
            return OpSupport.No($"{node.Op.Name}: elementwise output type {output}");
        }

        if (node.Op.Name == "Narrow")
        {
            var format = NumericFormats.Parse(node.Attributes.Get<string>("format"));
            if (format is NumericFormat.F8E8M0 or NumericFormat.I32 or NumericFormat.U32)
            {
                return OpSupport.No($"Narrow to {format}");
            }
        }

        return OpSupport.Yes;
    }

    /// <summary>Whether a tensor of type <paramref name="t"/> is held in an int local.</summary>
    private static bool IsIntLocal(DataType t) => t == DataType.I32;

    /// <summary>
    /// Slang expression of an elementwise node over typed operands, and whether its value is an int. Plain f32 nodes use
    /// <see cref="OpExpr"/> unchanged.
    /// </summary>
    private static (string Expr, bool IsInt) TypedOpExpr(Node node, string[] args, bool[] argIsInt)
    {
        bool outInt = IsIntLocal(node.Output.Type);
        string op = node.Op.Name;
        if (!outInt && !argIsInt.Any(b => b) && op is not ("Narrow" or "Reinterpret"))
        {
            return (OpExpr(node, args), false);
        }

        string AsFloat(int i) => argIsInt[i] ? $"float({args[i]})" : args[i];
        string AsInt(int i) => argIsInt[i] ? args[i] : $"int({args[i]})";
        switch (op)
        {
            case "Cast":
                return (outInt ? AsInt(0) : AsFloat(0), outInt);
            case "Reinterpret":
                return (outInt ? $"asint({AsFloat(0)})" : $"asfloat({AsInt(0)})", outInt);
            case "Where":
            {
                string condition = argIsInt[0] ? $"{args[0]} != 0" : $"{args[0]} != 0.0";
                return outInt ? ($"({condition} ? {AsInt(1)} : {AsInt(2)})", true) : ($"({condition} ? {AsFloat(1)} : {AsFloat(2)})", false);
            }
            case "Narrow":
                return NarrowExpr(node, args[0], argIsInt[0]);
        }

        if (s_integerCapable.Contains(op) && argIsInt.All(b => b))
        {
            return (IntOpExpr(node, args), outInt);
        }

        return (OpExpr(node, [.. args.Select((_, i) => AsFloat(i))]), false);
    }

    /// <summary>Slang expression of an integer elementwise node (all operands int).</summary>
    private static string IntOpExpr(Node node, string[] a) => node.Op.Name switch
    {
        "Add" => $"({a[0]} + {a[1]})",
        "Sub" => $"({a[0]} - {a[1]})",
        "Mul" => $"({a[0]} * {a[1]})",
        "Div" => $"idiv({a[0]}, {a[1]})",
        "Mod" => $"imod({a[0]}, {a[1]}, {Bool(node.Attributes.Get("fmod", false))})",
        "Max" => $"max({a[0]}, {a[1]})",
        "Min" => $"min({a[0]}, {a[1]})",
        "Pow" => $"ipow({a[0]}, {a[1]})",
        "BitAnd" => $"({a[0]} & {a[1]})",
        "BitOr" => $"({a[0]} | {a[1]})",
        "BitXor" => $"({a[0]} ^ {a[1]})",
        "ShiftLeft" => $"ishl({a[0]}, {a[1]})",
        "ShiftRight" => node.Attributes.Get("unsigned", false) ? $"ushr({a[0]}, {a[1]})" : $"ishr({a[0]}, {a[1]})",
        "Neg" => $"(-{a[0]})",
        "Abs" => $"abs({a[0]})",
        "Sign" => $"sign({a[0]})",
        "BitNot" => $"(~{a[0]})",
        "Equal" => $"({a[0]} == {a[1]} ? 1.0 : 0.0)",
        "NotEqual" => $"({a[0]} != {a[1]} ? 1.0 : 0.0)",
        "Less" => $"({a[0]} < {a[1]} ? 1.0 : 0.0)",
        "LessEqual" => $"({a[0]} <= {a[1]} ? 1.0 : 0.0)",
        "Greater" => $"({a[0]} > {a[1]} ? 1.0 : 0.0)",
        "GreaterEqual" => $"({a[0]} >= {a[1]} ? 1.0 : 0.0)",
        _ => throw new NotSupportedException($"integer {node.Op.Name}"),
    };

    private static (string Expr, bool IsInt) NarrowExpr(Node node, string x, bool isInt)
    {
        var format = NumericFormats.Parse(node.Attributes.Get<string>("format"));
        bool saturate = node.Attributes.Get("saturate", true);
        if (!NumericFormats.IsFloat(format))
        {
            var (min, max) = NumericFormats.IntegerRange(format);
            string wrapped(string v) => $"wrap_int({v}, {NumericFormats.Bits(format)}, {Bool(min < 0)})";
            return isInt
                ? (saturate ? $"clamp({x}, {min}, {max})" : wrapped(x), true)
                : (saturate ? $"clamp(trunc({x}), {F((float)min)}, {F((float)max)})" : $"float({wrapped($"int(trunc({x}))")})", false);
        }

        string expr = format switch
        {
            NumericFormat.BF16 => $"narrow_bf16({x})",
            NumericFormat.F16 => MiniFloatExpr(x, 10, -14, NumericFormat.F16, overflow: 1, fnuz: false),
            NumericFormat.F8E4M3FN => MiniFloatExpr(x, 3, -6, format, saturate ? 0 : 2, fnuz: false),
            NumericFormat.F8E4M3FNUZ => MiniFloatExpr(x, 3, -7, format, saturate ? 0 : 2, fnuz: true),
            NumericFormat.F8E5M2 => MiniFloatExpr(x, 2, -14, format, saturate ? 0 : 1, fnuz: false),
            NumericFormat.F8E5M2FNUZ => MiniFloatExpr(x, 2, -15, format, saturate ? 0 : 2, fnuz: true),
            NumericFormat.F4E2M1 => $"narrow_float({x}, 1, 0, {F((float)NumericFormats.MaxFinite(format))}, 0, false, true)",
            _ => throw new NotSupportedException($"Narrow to {format}"),
        };
        return (expr, false);
    }

    private static string MiniFloatExpr(string x, int mantissaBits, int minExponent, NumericFormat format, int overflow, bool fnuz)
        => $"narrow_float({x}, {mantissaBits}, {minExponent}, {F((float)NumericFormats.MaxFinite(format))}, {overflow}, {Bool(fnuz)}, false)";

    private static string Bool(bool b) => b ? "true" : "false";
}
