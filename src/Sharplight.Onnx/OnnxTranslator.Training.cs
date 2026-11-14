namespace Sharplight.Onnx;

// ai.onnx.preview.training optimizer steps (Adagrad, Adam, Momentum) as elementwise formulas over every optimized tensor.
internal sealed partial class OnnxTranslator
{
    private static readonly HashSet<string> s_trainingOps = ["Adagrad", "Adam", "Momentum"];

    /// <summary>Translates a node of the ai.onnx.preview.training domain (inputs R, T, then the tensors X..., G... and the optimizer states).</summary>
    private void TranslateTraining(OnnxNode n, TranslatedTensor?[] ins)
    {
        int groups = n.OpType == "Adam" ? 4 : 3;
        int count = (ins.Length - 2) / groups;
        string name = Nm(n);
        var rate = X(ins, 0);
        var step = X(ins, 1);
        var stepPositive = _b.Binary("Greater", step, F32Const([0f], name + ".zero"));
        for (int i = 0; i < count; i++)
        {
            var x = X(ins, 2 + i);
            var gradient = X(ins, 2 + count + i);
            float decay = n.Float("norm_coefficient", 0f);
            var regularized = decay == 0 ? gradient : _b.Add(_b.Scale(x, decay), gradient);
            switch (n.OpType)
            {
                case "Adagrad":
                {
                    // r = R / (1 + T · decay_factor); H' = H + g²; X' = X - r · g / (sqrt(H') + epsilon)
                    var r = _b.Binary("Div", rate, _b.Scale(step, n.Float("decay_factor", 0f), 1f));
                    var h = _b.Add(X(ins, 2 + 2 * count + i), _b.Mul(regularized, regularized));
                    var adaptive = _b.Scale(_b.Unary("Sqrt", h), 1f, n.Float("epsilon", 1e-6f));
                    SetDyn(n, i, _b.Sub(x, _b.Mul(r, _b.Binary("Div", regularized, adaptive))));
                    SetDyn(n, count + i, h);
                    break;
                }

                case "Momentum":
                {
                    // V' = alpha · V + beta' · g (beta' = beta after the first step, else 1); standard X' = X - R · V',
                    // nesterov X' = X - R · (g + alpha · V')
                    float alpha = n.Float("alpha", 0f);
                    float beta = n.Float("beta", 0f);
                    var betaAdjusted = Emit("Where", [stepPositive, F32Const([beta], name + ".beta"), F32Const([1f], name + ".one")]);
                    var v = _b.Add(_b.Scale(X(ins, 2 + 2 * count + i), alpha), _b.Mul(betaAdjusted, regularized));
                    var direction = n.Str("mode", "standard") == "nesterov" ? _b.Add(regularized, _b.Scale(v, alpha)) : v;
                    SetDyn(n, i, _b.Sub(x, _b.Mul(rate, direction)));
                    SetDyn(n, count + i, v);
                    break;
                }

                default:
                {
                    // V' = alpha · V + (1 - alpha) · g; H' = beta · H + (1 - beta) · g²;
                    // R' = R · sqrt(1 - beta^T) / (1 - alpha^T) after the first step; X' = (1 - post) · (X - R' · V' / (sqrt(H') + epsilon))
                    float alpha = n.Float("alpha", 0.9f);
                    float beta = n.Float("beta", 0.999f);
                    var v = _b.Add(_b.Scale(X(ins, 2 + 2 * count + i), alpha), _b.Scale(regularized, 1f - alpha));
                    var h = _b.Add(_b.Scale(X(ins, 2 + 3 * count + i), beta), _b.Scale(_b.Mul(regularized, regularized), 1f - beta));
                    var alphaPower = _b.Unary("Exp", _b.Scale(step, MathF.Log(alpha)));
                    var betaPower = _b.Unary("Exp", _b.Scale(step, MathF.Log(beta)));
                    var corrected = _b.Mul(rate, _b.Binary("Div", _b.Unary("Sqrt", _b.Scale(betaPower, -1f, 1f)), _b.Scale(alphaPower, -1f, 1f)));
                    var adjusted = Emit("Where", [stepPositive, corrected, rate]);
                    var update = _b.Mul(adjusted, _b.Binary("Div", v, _b.Scale(_b.Unary("Sqrt", h), 1f, n.Float("epsilon", 1e-6f))));
                    SetDyn(n, i, _b.Scale(_b.Sub(x, update), 1f - n.Float("norm_coefficient_post", 0f)));
                    SetDyn(n, count + i, v);
                    SetDyn(n, 2 * count + i, h);
                    break;
                }
            }
        }
    }
}
