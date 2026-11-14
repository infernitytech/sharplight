using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// ONNX element types of dynamic values, casts, and exact integer arithmetic.
//
// Transport types: every float type travels as f32 and every integer type as i32 (int64/uint64 keep their low 32 bits; uint32 keeps its
// bit pattern), bool as an f32 0/1 mask. The translator tracks each value's ONNX element type (graph inputs, value_info, and the rules of
// InferElemType) so that
//  * Cast and CastLike round to the target's value set (IR Narrow: f16, bf16, FP8 with saturation, FP4, E8M0) or wrap to a narrow
//    integer type, instead of keeping the wider transport value;
//  * arithmetic, comparison and bitwise ops whose operands are all integers compute in i32 (exact up to 32 bits: Div truncates, Mod
//    follows fmod, bit shifts and masks are exact), wrapping results of 2- to 16-bit types to their width.
internal sealed partial class OnnxTranslator
{
    private static readonly HashSet<string> s_integerBinary = ["Add", "Sub", "Mul", "Div", "Max", "Min", "Mod", "Pow", "Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual"];
    private static readonly HashSet<string> s_integerUnary = ["Neg", "Abs", "Sign"];

    /// <summary>Integer ops whose result can leave the range of a narrow operand type (and is wrapped back).</summary>
    private static readonly HashSet<string> s_overflowing = ["Add", "Sub", "Mul", "Pow", "Neg", "Abs", "Div", "ShiftLeft", "BitNot"];

    private static readonly HashSet<string> s_maskResults = ["Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual", "And", "Or", "Xor", "Not", "IsNaN", "IsInf"];

    /// <summary>
    /// Ops whose bfloat16 result is exactly one of their bfloat16 inputs' values; every other op producing a bfloat16 value is followed by a
    /// Narrow to bf16, so that bfloat16 graphs round like an engine storing bfloat16 tensors (bf16's 8-bit significand is far coarser than
    /// the comparison tolerance of float results; float16 values are not rounded per op, only at Casts).
    /// </summary>
    private static readonly HashSet<string> s_bfloat16Exact =
    [
        "Identity", "Dropout", "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Transpose", "Slice", "Gather", "GatherElements", "GatherND",
        "Concat", "Split", "Expand", "Tile", "Pad", "Max", "Min", "Relu", "Neg", "Abs", "Where", "Cast", "CastLike", "ReduceMax", "ReduceMin",
        "MaxPool", "DepthToSpace", "SpaceToDepth", "TopK", "Clip", "Sign", "Floor", "Ceil", "Round", "QuantizeLinear", "DequantizeLinear",
    ];

    private Dictionary<string, int>? _declaredTypes;

    /// <summary>ONNX element type of a tensor (declared, inferred, or derived from the IR transport type).</summary>
    private static int ElemTypeOf(TranslatedTensor t)
    {
        if (t.C is { } c)
        {
            return c.ElemType;
        }

        if (t.ElemType != 0)
        {
            return t.ElemType;
        }

        return t.V!.Type.Kind == DataTypeKind.Int ? OnnxType.Int64 : OnnxType.Float;
    }

    /// <summary>Records the ONNX element type of each dynamic output of a translated node that has none yet.</summary>
    private void AssignElemTypes(OnnxNode n, TranslatedTensor?[] ins)
    {
        _declaredTypes ??= _graph.ValueInfo.Concat(_graph.Outputs).Where(v => v.ElemType != 0)
            .GroupBy(v => v.Name, StringComparer.Ordinal).ToDictionary(g => g.Key, g => g.First().ElemType, StringComparer.Ordinal);
        for (int k = 0; k < n.Outputs.Count; k++)
        {
            string name = n.Outputs[k];
            if (name != "" && _env.TryGetValue(name, out var t) && !t.IsConst && t.ElemType == 0)
            {
                t.ElemType = _declaredTypes.TryGetValue(name, out int declared) ? declared : InferElemType(n, ins, k);
                if (t.ElemType == OnnxType.BFloat16 && t.V!.Type == DataType.F32 && !s_bfloat16Exact.Contains(n.OpType))
                {
                    var rounded = Emit("Narrow", [t.V], Attributes.From(("format", NumericFormats.Name(NumericFormat.BF16))));
                    _env[name] = Typed(rounded, OnnxType.BFloat16);
                }
            }
        }
    }

    /// <summary>ONNX type inference for output <paramref name="k"/> of <paramref name="n"/> (the cases the translator meets).</summary>
    private static int InferElemType(OnnxNode n, TranslatedTensor?[] ins, int k)
    {
        int First() => ins.FirstOrDefault(t => t is not null) is { } t ? ElemTypeOf(t) : OnnxType.Float;
        int Input(int i) => i < ins.Length && ins[i] is { } t ? ElemTypeOf(t) : 0;
        int Attribute(string name, int fallback) => n.Has(name) ? (int)n.Int(name, fallback) : fallback;
        if (s_maskResults.Contains(n.OpType))
        {
            return OnnxType.Bool;
        }

        return n.OpType switch
        {
            "Cast" or "BitCast" => Attribute("to", OnnxType.Float),
            "CastLike" => Input(1),
            "Shape" or "Size" or "NonZero" or "NonMaxSuppression" or "ArgMax" or "ArgMin" => OnnxType.Int64,
            "TopK" => k == 1 ? OnnxType.Int64 : First(),
            "Unique" => k == 0 ? First() : OnnxType.Int64,
            "QuantizeLinear" => Attribute("output_dtype", 0) is > 0 and var dtype ? dtype : Input(2) is > 0 and var zp ? zp : OnnxType.UInt8,
            "DequantizeLinear" => Attribute("output_dtype", 0) is > 0 and var dtype ? dtype : Input(1),
            "DynamicQuantizeLinear" => k == 1 ? OnnxType.Float : OnnxType.UInt8,
            "MatMulInteger" or "ConvInteger" => OnnxType.Int32,
            "QLinearMatMul" or "QLinearConv" => Input(7) is > 0 and var zp ? zp : Input(0),
            "ConstantOfShape" => n.Attributes.TryGetValue("value", out var value) && value.T is { } tensor ? tensor.ElemType : OnnxType.Float,
            "Where" => Input(1),
            "OneHot" => Input(2),
            "EyeLike" or "RandomNormalLike" or "RandomUniformLike" or "Bernoulli" => Attribute("dtype", First()),
            "RandomNormal" or "RandomUniform" => Attribute("dtype", OnnxType.Float),
            "Multinomial" => Attribute("dtype", OnnxType.Int32),
            _ => First(),
        };
    }

    /// <summary>An integer operand as an i32 IR value (dynamic f32 values holding integers are cast, constants become i32 constants).</summary>
    private Value IntValue(TranslatedTensor t) => Idx(t);

    private TranslatedTensor Typed(Value v, int elemType) => new() { V = v, ElemType = elemType };

    /// <summary>Wraps an i32 result to the width of a narrow integer type (no-op for 32- and 64-bit types).</summary>
    private Value WrapTo(Value v, int elemType, string? name = null)
        => OnnxType.NarrowFormat(elemType) is { } format && !NumericFormats.IsFloat(format)
            ? Emit("Narrow", [v], Attributes.From(("format", NumericFormats.Name(format)), ("saturate", false)), name)
            : v;

    private static bool AllIntegers(TranslatedTensor?[] ins) => ins.All(t => t is null || OnnxType.IsInteger(ElemTypeOf(t)));

    /// <summary>A binary op over integer operands in i32; comparisons yield masks.</summary>
    private void IntegerBinary(OnnxNode n, TranslatedTensor?[] ins, string irOp, Attributes? attributes = null)
    {
        string name = Nm(n);
        int type = ElemTypeOf(ins[0]!);
        var operands = BroadcastOperands(ins, IntValue(ins[0]!), IntValue(ins[1]!));
        bool mask = s_maskResults.Contains(n.OpType);
        bool wrap = !mask && s_overflowing.Contains(irOp) && OnnxType.NarrowFormat(type) is not null;
        var v = Emit(irOp, operands, attributes, wrap ? null : name);
        _env[n.Outputs[0]] = Typed(wrap ? WrapTo(v, type, name) : v, mask ? OnnxType.Bool : type);
    }

    private void IntegerUnary(OnnxNode n, TranslatedTensor x, string irOp)
    {
        string name = Nm(n);
        int type = ElemTypeOf(x);
        bool wrap = s_overflowing.Contains(irOp) && OnnxType.NarrowFormat(type) is not null;
        var v = Emit(irOp, [IntValue(x)], null, wrap ? null : name);
        _env[n.Outputs[0]] = Typed(wrap ? WrapTo(v, type, name) : v, type);
    }

    private void BitShift(OnnxNode n, TranslatedTensor?[] ins)
    {
        bool left = n.Str("direction", "LEFT") == "LEFT";
        var attributes = left ? null : Attributes.From(("unsigned", OnnxType.IsUnsigned(ElemTypeOf(ins[0]!))));
        IntegerBinary(n, ins, left ? "ShiftLeft" : "ShiftRight", attributes);
    }

    private void Bitwise(OnnxNode n, TranslatedTensor?[] ins)
    {
        if (n.OpType == "BitwiseNot")
        {
            IntegerUnary(n, ins[0]!, "BitNot");
            return;
        }

        IntegerBinary(n, ins, n.OpType switch
        {
            "BitwiseAnd" => "BitAnd",
            "BitwiseOr" => "BitOr",
            _ => "BitXor",
        });
    }

    /// <summary>IsInf with detect_positive / detect_negative: the infinity mask restricted to one sign.</summary>
    private void IsInf(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        bool positive = n.Int("detect_positive", 1) != 0;
        bool negative = n.Int("detect_negative", 1) != 0;
        var mask = Emit("IsInf", [x], null, positive && negative ? Nm(n) : null);
        var zero = ConstValue(OnnxConst.Scalar(0f).Reshaped([1]), "zero");
        if (!positive)
        {
            mask = Emit("And", [mask, Emit("Less", [x, zero])], null, negative ? Nm(n) : null);
        }

        if (!negative)
        {
            mask = Emit("And", [mask, Emit("Greater", [x, zero])], null, Nm(n));
        }

        SetDyn(n, 0, mask);
    }

    private void Cast(OnnxNode n, TranslatedTensor?[] ins)
        => CastTo(n, ins[0]!, (int)n.Int("to", OnnxType.Float), n.Int("saturate", 1) != 0, n.Str("round_mode", "up"));

    /// <summary>CastLike is a Cast to the element type of the second input.</summary>
    private void CastLike(OnnxNode n, TranslatedTensor?[] ins)
        => CastTo(n, ins[0]!, ElemTypeOf(ins[1]!), n.Int("saturate", 1) != 0, n.Str("round_mode", "up"));

    private void CastTo(OnnxNode n, TranslatedTensor x, int to, bool saturate, string roundMode)
    {
        string name = Nm(n);
        int from = ElemTypeOf(x);
        Value v;
        if (to == OnnxType.Bool)
        {
            v = Emit("NotEqual", [AsValue(x), ConstValue(OnnxConst.Scalar(0f).Reshaped([1]), "zero")], null, name);
        }
        else if (OnnxType.IsFloat(to))
        {
            v = AsValue(x);
            if (OnnxType.NarrowFormat(to) is { } format && !Represents(to, from))
            {
                var attributes = Attributes.From(("format", NumericFormats.Name(format)), ("saturate", saturate), ("rounding", roundMode));
                v = Emit("Narrow", [v], attributes, name);
            }
        }
        else if (OnnxType.IsInteger(to))
        {
            v = IntValue(x);
            if (!Represents(to, from))
            {
                v = WrapTo(v, to, name);
            }
        }
        else
        {
            throw new NotSupportedException($"Cast to {OnnxType.Name(to)}");
        }

        _env[n.Outputs[0]] = Typed(v, to);
    }

    /// <summary>
    /// Whether every value of ONNX type <paramref name="from"/> is exactly a value of the narrow type <paramref name="to"/>, so that a
    /// Cast needs no Narrow (types without a narrow format hold every transported value).
    /// </summary>
    private static bool Represents(int to, int from)
    {
        if (to == from || OnnxType.NarrowFormat(to) is not { } target)
        {
            return true;
        }

        var source = OnnxType.NarrowFormat(from);
        bool smallSource = from == OnnxType.Bool || (source is not null && from != OnnxType.Float8E8M0 && OnnxType.ElementBits(from) <= 8);
        if (NumericFormats.IsFloat(target))
        {
            return target is NumericFormat.F16 or NumericFormat.BF16 && smallSource;
        }

        if (OnnxType.IsFloat(from))
        {
            return false;
        }

        var (min, max) = NumericFormats.IntegerRange(target);
        var (fromMin, fromMax) = from == OnnxType.Bool ? (0L, 1L) : source is { } narrow ? NumericFormats.IntegerRange(narrow) : (long.MinValue, long.MaxValue);
        return fromMin >= min && fromMax <= max;
    }

    /// <summary>BitCast between types of the same width: f32 ↔ i32 bit patterns, or a wrap between 8- and 16-bit integer types.</summary>
    private void BitCast(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = ins[0]!;
        int from = ElemTypeOf(x);
        int to = (int)n.Int("to", OnnxType.Float);
        string name = Nm(n);
        if (OnnxType.ElementBits(from) == 64 || OnnxType.ElementBits(to) == 64)
        {
            throw new NotSupportedException("BitCast of 64-bit types (they travel as 32-bit values)");
        }

        if (OnnxType.IsFloat(from) != OnnxType.IsFloat(to) && (from != OnnxType.Float && to != OnnxType.Float))
        {
            throw new NotSupportedException($"BitCast {OnnxType.Name(from)} -> {OnnxType.Name(to)} (16-bit floats travel as f32)");
        }

        Value v = from == OnnxType.Float || to == OnnxType.Float
            ? Emit("Reinterpret", [from == OnnxType.Float ? AsValue(x) : IntValue(x)], Attributes.From(("type", to == OnnxType.Float ? "f32" : "i32")), name)
            : WrapTo(IntValue(x), to, name);
        _env[n.Outputs[0]] = Typed(v, to);
    }
}
