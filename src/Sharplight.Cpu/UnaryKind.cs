namespace Sharplight.Cpu;

/// <summary>Unary elementwise operations; the member names equal the graph op names (Scale and Clip carry attributes).</summary>
internal enum UnaryKind { Neg, Exp, Log, Sqrt, Rsqrt, Sigmoid, Silu, Gelu, GeluTanh, Tanh, Softplus, Relu, Abs, Square, Recip, Scale, Erf, HardSwish, HardSigmoid, Clip,
    Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Asinh, Acosh, Atanh, Floor, Ceil, Round, Sign, Softsign, Mish,
    Elu, Celu, Selu, LeakyRelu, ThresholdedRelu, Not, IsNaN, IsInf }
