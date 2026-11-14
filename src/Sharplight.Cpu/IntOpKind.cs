namespace Sharplight.Cpu;

/// <summary>Integer elementwise operations of <see cref="IntElementwiseOp"/>; the member names equal the graph op names.</summary>
internal enum IntOpKind
{
    Add, Sub, Mul, Div, Mod, Max, Min, Pow, BitAnd, BitOr, BitXor, ShiftLeft, ShiftRight, Neg, Abs, Sign, BitNot,
    Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual,
}
