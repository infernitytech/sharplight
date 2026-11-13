using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Slang expression of an elementwise node applied to operand expressions.</summary>
    private static string OpExpr(Node node, string[] args)
    {
    return node.Op.Name switch
        {
            "Neg" => $"(-{args[0]})",
            "Exp" => $"exp({args[0]})",
            "Log" => $"log({args[0]})",
            "Sqrt" => $"sqrt({args[0]})",
            "Rsqrt" => $"rsqrt({args[0]})",
            "Sigmoid" => $"(1.0 / (1.0 + exp(-{args[0]})))",
            "Silu" => $"({args[0]} / (1.0 + exp(-{args[0]})))",
            "Gelu" => $"gelu_erf({args[0]})",
            "GeluTanh" => $"gelu_tanh({args[0]})",
            "Tanh" => $"tanh({args[0]})",
            "Softplus" => $"({args[0]} > 20.0 ? {args[0]} : log(1.0 + exp({args[0]})))",
            "Relu" => $"max({args[0]}, 0.0)",
            "Abs" => $"abs({args[0]})",
            "Square" => $"({args[0]} * {args[0]})",
            "Recip" => $"(1.0 / {args[0]})",
            "Add" => $"({args[0]} + {args[1]})",
            "Sub" => $"({args[0]} - {args[1]})",
            "Mul" => $"({args[0]} * {args[1]})",
            "Div" => $"({args[0]} / {args[1]})",
            "Max" => $"max({args[0]}, {args[1]})",
            "Min" => $"min({args[0]}, {args[1]})",
            "Scale" => $"({args[0]} * {F(node.Attributes.Get("scale", 1f))} + {F(node.Attributes.Get("bias", 0f))})",
            "Cast" => args[0],
            "Sin" => $"sin({args[0]})", "Cos" => $"cos({args[0]})", "Tan" => $"tan({args[0]})",
            "Asin" => $"asin({args[0]})", "Acos" => $"acos({args[0]})", "Atan" => $"atan({args[0]})",
            "Sinh" => $"sinh({args[0]})", "Cosh" => $"cosh({args[0]})",
            "Asinh" => $"asinh_f({args[0]})",
            "Acosh" => $"log({args[0]} + sqrt({args[0]} * {args[0]} - 1.0))",
            "Atanh" => $"(0.5 * log((1.0 + {args[0]}) / (1.0 - {args[0]})))",
            "Floor" => $"floor({args[0]})", "Ceil" => $"ceil({args[0]})", "Round" => $"round_even({args[0]})", "Sign" => $"sign_f({args[0]})",
            "Softsign" => $"({args[0]} / (1.0 + abs({args[0]})))",
            "Mish" => $"({args[0]} * tanh({args[0]} > 20.0 ? {args[0]} : log(1.0 + exp({args[0]}))))",
            "Elu" => $"({args[0]} > 0.0 ? {args[0]} : {F(node.Attributes.Get("alpha", 1f))} * (exp({args[0]}) - 1.0))",
            "Celu" => $"(max({args[0]}, 0.0) + min(0.0, {F(node.Attributes.Get("alpha", 1f))} * (exp({args[0]} / {F(node.Attributes.Get("alpha", 1f))}) - 1.0)))",
            "Selu" => $"({F(node.Attributes.Get("gamma", 1.0507009873554805f))} * ({args[0]} > 0.0 ? {args[0]} : {F(node.Attributes.Get("alpha", 1.6732632423543772f))} * (exp({args[0]}) - 1.0)))",
            "LeakyRelu" => $"({args[0]} >= 0.0 ? {args[0]} : {F(node.Attributes.Get("alpha", 0.01f))} * {args[0]})",
            "ThresholdedRelu" => $"({args[0]} > {F(node.Attributes.Get("alpha", 1f))} ? {args[0]} : 0.0)",
            "Not" => $"({args[0]} == 0.0 ? 1.0 : 0.0)",
            "IsNaN" => $"(isnan({args[0]}) ? 1.0 : 0.0)",
            "IsInf" => $"(isinf({args[0]}) ? 1.0 : 0.0)",
            "Mod" => node.Attributes.Get("fmod", false) ? $"fmod({args[0]}, {args[1]})" : $"mod_py({args[0]}, {args[1]})",
            "PRelu" => $"({args[0]} >= 0.0 ? {args[0]} : {args[1]} * {args[0]})",
            "Equal" => $"({args[0]} == {args[1]} ? 1.0 : 0.0)", "NotEqual" => $"({args[0]} != {args[1]} ? 1.0 : 0.0)",
            "Less" => $"({args[0]} < {args[1]} ? 1.0 : 0.0)", "LessEqual" => $"({args[0]} <= {args[1]} ? 1.0 : 0.0)",
            "Greater" => $"({args[0]} > {args[1]} ? 1.0 : 0.0)", "GreaterEqual" => $"({args[0]} >= {args[1]} ? 1.0 : 0.0)",
            "And" => $"(({args[0]} != 0.0 && {args[1]} != 0.0) ? 1.0 : 0.0)", "Or" => $"(({args[0]} != 0.0 || {args[1]} != 0.0) ? 1.0 : 0.0)",
            "Xor" => $"(({args[0]} != 0.0) != ({args[1]} != 0.0) ? 1.0 : 0.0)",
            "Where" => $"({args[0]} != 0.0 ? {args[1]} : {args[2]})",
            "Clip" => ClipExpr(args[0], node.Attributes.Get("min", float.NegativeInfinity), node.Attributes.Get("max", float.PositiveInfinity)),
            "Erf" => $"erf_accurate({args[0]})",
            "HardSwish" => $"({args[0]} * clamp({args[0]} * {F(1f / 6f)} + 0.5, 0.0, 1.0))",
            "HardSigmoid" => $"clamp({args[0]} * {F(node.Attributes.Get("alpha", 0.2f))} + {F(node.Attributes.Get("beta", 0.5f))}, 0.0, 1.0)",
            "Pow" => $"pow_onnx({args[0]}, {args[1]})",
            _ => throw new NotSupportedException(node.Op.Name),
        };
    }

    private static string ClipExpr(string x, float lo, float hi)
    {
        var e = x;
        if (!float.IsNegativeInfinity(lo))
        {
            e = $"max({e}, {F(lo)})";
        }

        if (!float.IsPositiveInfinity(hi))
        {
            e = $"min({e}, {F(hi)})";
        }

        return $"({e})";
    }
}
