using Sharplight.Graph;

namespace Sharplight.Onnx;

// Folding of a constant scale into the product it scales (attention scores: Div(Q·Kᵀ, √d)).
internal sealed partial class OnnxTranslator
{
    /// <summary>
    /// Translates <paramref name="n"/> (<c>Div(p, c)</c> or <c>Mul(p, c)</c> with a scalar constant c) as the BatchMatMul
    /// producing p with its alpha scaled, when that product is read only by <paramref name="n"/>: one pass over the
    /// attention scores fewer. The unscaled product is left to dead-node elimination. Returns false when the pattern does
    /// not match.
    /// </summary>
    private bool TryScaledBatchMatMul(OnnxNode n)
    {
        if (ConstOperand(n, out int other) is not { Count: 1 } c || (n.OpType == "Div" && other != 0))
        {
            return false;
        }

        string product = n.Inputs[other];
        double scale = n.OpType == "Div" ? 1.0 / c.ScalarF32() : c.ScalarF32();
        if (!double.IsFinite(scale) || scale == 0 || Producer(product, "MatMul") is null || SoleConsumer(product, n.OpType) != n
            || !_env.TryGetValue(product, out var t) || t is not { LeadingUnits: 0, V: { Producer: { Op.Name: "BatchMatMul" } bmm } v }
            || v.Type != Tensors.DataType.F32)
        {
            return false;
        }

        var a = bmm.Attributes;
        var attributes = Attributes.From(("trans_a", a.Get("trans_a", false)), ("trans_b", a.Get("trans_b", false)),
            ("alpha", (float)(a.Get("alpha", 1f) * scale)));
        SetDyn(n, 0, Emit("BatchMatMul", [.. bmm.Inputs], attributes, Nm(n)));
        return true;
    }
}
