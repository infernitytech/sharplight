namespace Sharplight.Studio.Services.Inspection;

/// <summary>Maps op names to an <see cref="OpCategory"/> by name patterns (ops of any importer, no fixed list).</summary>
internal static class OpCategories
{
    /// <summary>The category of <paramref name="op"/>.</summary>
    public static OpCategory Of(string op)
    {
        if (Has(op, "Attention"))
        {
            return OpCategory.Attention;
        }

        if (Has(op, "Cache") || Has(op, "State") || Has(op, "Scan"))
        {
            return OpCategory.State;
        }

        if (Has(op, "MatMul") || Has(op, "Gemm") || Has(op, "Conv") || Has(op, "Linear") || Has(op, "Einsum"))
        {
            return OpCategory.Linear;
        }

        if (Has(op, "Norm"))
        {
            return OpCategory.Normalization;
        }

        if (Has(op, "Reshape") || Has(op, "Transpose") || Has(op, "Concat") || Has(op, "Slice") || Has(op, "Gather")
            || Has(op, "Split") || Has(op, "Squeeze") || Has(op, "Expand") || Has(op, "Permute") || Has(op, "Pad")
            || Has(op, "Tile") || Has(op, "Scatter") || Has(op, "View") || Has(op, "Flatten") || Has(op, "Cast"))
        {
            return OpCategory.Layout;
        }

        if (op is "Add" or "Sub" or "Mul" or "Div" or "Pow" or "Neg" or "Exp" or "Log" or "Sqrt" or "Abs" or "Max" or "Min"
            || Has(op, "Silu") || Has(op, "Gelu") || Has(op, "Relu") || Has(op, "Sigmoid") || Has(op, "Tanh") || Has(op, "Softmax")
            || Has(op, "Rope") || Has(op, "Swiglu") || Has(op, "Glu") || Has(op, "Scale") || Has(op, "Clamp") || Has(op, "Where"))
        {
            return OpCategory.Elementwise;
        }

        return OpCategory.Other;
    }

    private static bool Has(string op, string part) => op.Contains(part, StringComparison.OrdinalIgnoreCase);
}
