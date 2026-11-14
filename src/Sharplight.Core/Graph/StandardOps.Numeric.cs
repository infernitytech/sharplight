using Sharplight.Tensors;

namespace Sharplight.Graph;

// Element formats and integer arithmetic.
//  * Values of narrow formats (f16, bf16, FP8, FP4, 2- to 16-bit integers) may travel in a wider storage type (f32 for floats, i32 for
//    integers); Narrow rounds or wraps them to the narrow format's value set, so results match an engine that stores the narrow type.
//  * The arithmetic ops Add, Sub, Mul, Div, Max, Min, Mod and Pow, and the unary Neg, Abs and Sign compute in i32 when every input is
//    i32 (wrapping on overflow; Div truncates toward zero, Mod follows its fmod attribute, division by zero yields 0) and return i32.
//    Comparisons of two i32 inputs compare exactly and return f32 masks as usual.
public static partial class StandardOps
{
    /// <summary>Binary ops that return i32 when both inputs are i32 (otherwise f32).</summary>
    public static readonly string[] IntegerPreservingBinaryNames = ["Add", "Sub", "Mul", "Div", "Max", "Min", "Mod", "Pow"];

    /// <summary>Unary ops that return i32 for an i32 input (otherwise f32).</summary>
    public static readonly string[] IntegerPreservingUnaryNames = ["Neg", "Abs", "Sign"];

    /// <summary>
    /// Two-operand bitwise ops on i32 with numpy-style broadcasting, returning i32: <c>BitAnd</c>, <c>BitOr</c>, <c>BitXor</c>,
    /// <c>ShiftLeft</c> and <c>ShiftRight</c>. A shift amount outside 0..31 shifts every bit out (left: 0; right: 0, or -1 for a
    /// negative value). ShiftRight is arithmetic; attribute <c>unsigned</c> (bool) makes it logical.
    /// </summary>
    public static readonly string[] BitwiseBinaryNames = ["BitAnd", "BitOr", "BitXor", "ShiftLeft", "ShiftRight"];

    /// <summary>Bitwise complement of an i32 tensor. Output: i32, shape of x.</summary>
    public static readonly OpSchema BitNot = new("BitNot", OpCategory.Elementwise, 1, 1, (i, a) => [new(RequireI32(i[0], "BitNot"), i[0].Shape)]);

    /// <summary>
    /// Rounds (floats) or wraps/clamps (integers) x to the value set of a narrower <see cref="NumericFormat"/>, keeping x's storage type
    /// (see <see cref="NumericFormats.Round"/>). Input: x, f32 or i32 (i32 only with integer formats). Attributes: <c>format</c> (string,
    /// <see cref="NumericFormats.Name"/>), <c>saturate</c> (bool, default true; integer formats wrap when false), <c>rounding</c>
    /// (string up | down | nearest, default up; only <see cref="NumericFormat.F8E8M0"/>). Output: type and shape of x.
    /// </summary>
    public static readonly OpSchema Narrow = new("Narrow", OpCategory.Elementwise, 1, 1, (i, a) =>
    {
        var format = NumericFormats.Parse(a.Get<string>("format"));
        if (i[0].Type != DataType.F32 && !(i[0].Type == DataType.I32 && !NumericFormats.IsFloat(format)))
        {
            throw new ArgumentException($"Narrow to {format} needs an f32 input (or i32 for integer formats), got {i[0].Type}.");
        }

        return [new(i[0].Type, i[0].Shape)];
    });

    /// <summary>
    /// Reinterprets the bits of each element as another 4-byte type (f32 ↔ i32). Input: x (f32 or i32). Attribute: <c>type</c> (string,
    /// "f32" or "i32"). Output: that type, shape of x.
    /// </summary>
    public static readonly OpSchema Reinterpret = new("Reinterpret", OpCategory.Elementwise, 1, 1, (i, a) =>
    {
        var type = DataType.Get(a.Get<string>("type"));
        if (type.BlockBytes != 4 || i[0].Type.BlockBytes != 4 || type.BlockSize != 1 || i[0].Type.BlockSize != 1)
        {
            throw new ArgumentException($"Reinterpret {i[0].Type} as {type}: both types must be 4-byte scalars.");
        }

        return [new(type, i[0].Shape)];
    });

    private static DataType RequireI32(Value v, string op)
        => v.Type == DataType.I32 ? DataType.I32 : throw new ArgumentException($"{op} needs i32 inputs, got {v.Type}.");

    private static IReadOnlyList<TensorType> IntegerPreservingBroadcast(IReadOnlyList<Value> i, Attributes a) =>
        [new(i[0].Type == DataType.I32 && i[1].Type == DataType.I32 ? DataType.I32 : DataType.F32, BroadcastShapes(i[0].Shape, i[1].Shape))];

    private static IReadOnlyList<TensorType> IntegerPreservingSame(IReadOnlyList<Value> i, Attributes a) =>
        [new(i[0].Type == DataType.I32 ? DataType.I32 : DataType.F32, i[0].Shape)];

    private static IReadOnlyList<TensorType> BitwiseBroadcast(IReadOnlyList<Value> i, Attributes a)
    {
        RequireI32(i[0], "bitwise op");
        RequireI32(i[1], "bitwise op");
        return [new(DataType.I32, BroadcastShapes(i[0].Shape, i[1].Shape))];
    }

    private static void RegisterNumeric()
    {
        foreach (var n in IntegerPreservingBinaryNames)
        {
            OpRegistry.Register(new OpSchema(n, OpCategory.Elementwise, 2, 2, IntegerPreservingBroadcast));
        }

        foreach (var n in IntegerPreservingUnaryNames)
        {
            OpRegistry.Register(new OpSchema(n, OpCategory.Elementwise, 1, 1, IntegerPreservingSame));
        }

        foreach (var n in BitwiseBinaryNames)
        {
            OpRegistry.Register(new OpSchema(n, OpCategory.Elementwise, 2, 2, BitwiseBroadcast));
        }
    }
}
