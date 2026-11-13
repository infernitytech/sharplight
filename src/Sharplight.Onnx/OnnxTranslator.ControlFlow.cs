namespace Sharplight.Onnx;

// Control flow: If with a compile-time constant condition is inlined.
internal sealed partial class OnnxTranslator
{
    // ---- If --------------------------------------------------------------------------------------------------------
    private void IfOp(OnnxNode n, TranslatedTensor?[] ins)
    {
        var cond = ConstIn(ins, 0, "If condition");
        var g = (cond.Floats()[0] != 0 ? n.Attributes["then_branch"] : n.Attributes["else_branch"]).G!;
        foreach (var t in g.Initializers)
        {
            if (t.Source is not null)
            {
                _env[t.Name] = TranslatedTensor.Const(OnnxConst.FromTensor(t));
            }
        }

        foreach (var sub in g.Nodes)
        {
            if (!IsSupportedNode(sub))
            {
                throw new NotSupportedException($"If branch uses unsupported operator {sub.OpType}");
            }

            HandleNode(sub);
        }
        for (int i = 0; i < n.Outputs.Count; i++)
        {
            if (n.Outputs[i] != "")
            {
                _env[n.Outputs[i]] = Get(g.Outputs[i].Name);
            }
        }
    }
}
