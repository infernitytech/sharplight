using Sharplight.Graph;

namespace Sharplight.Onnx;

// Dynamic quantization and the com.microsoft QLinear* family. Quantized activations are carried as f32 holding integer values
// (see QuantOps); scales/zero points may be compile-time constants or runtime values (DynamicQuantizeLinear).
internal sealed partial class OnnxTranslator
{
    private static readonly HashSet<string> s_quantContrib =
    [
        "QLinearAdd", "QLinearMul", "QLinearGlobalAveragePool", "QLinearAveragePool", "QLinearSigmoid", "QLinearLeakyRelu", "QLinearConcat", "QLinearSoftmax",
    ];

    /// <summary>A scale or zero point as an f32 value broadcastable against a tensor (constant, or a runtime scalar).</summary>
    private Value QParam(Tn t, string name) => t.IsConst ? ConstValue(t.C!.CastTo(OnnxType.Float).Count == 1 ? t.C.CastTo(OnnxType.Float).Reshaped([1]) : t.C.CastTo(OnnxType.Float), name) : t.V!;

    /// <summary>(q - zp) * scale.</summary>
    private Value Deq(Tn x, Tn scale, Tn? zp, string name)
    {
        var v = AsValue(x);
        bool zeroZp = zp is null || (zp.IsConst && zp.C!.Floats().All(z => z == 0));
        if (!zeroZp)
        {
            v = _b.Sub(v, QParam(zp!, name + ".zp"));
        }

        if (scale.IsConst && scale.C!.Count == 1)
        {
            return _b.Scale(v, scale.C.ScalarF32());
        }

        return _b.Mul(v, QParam(scale, name + ".scale"));
    }

    /// <summary>clip(round(y / scale) + zp, range of the zero-point type).</summary>
    private Value Req(Value y, Tn scale, Tn? zp, string name)
    {
        var q = Emit("Round", [_b.Binary("Div", y, QParam(scale, name + ".ys"))]);   // a true division (not * 1/s) so ties round like onnxruntime
        int et = zp is { IsConst: true } ? zp.C!.ElemType : OnnxType.UInt8;
        if (zp is not null)
        {
            q = _b.Add(q, QParam(zp, name + ".yzp"));
        }

        var (lo, hi) = OnnxFold.Range(et);
        return Emit("Clip", [q], Attributes.From(("min", (float)lo), ("max", (float)hi)), name);
    }

    private Tn? Opt(Tn?[] ins, int i) => i < ins.Length ? ins[i] : null;

    private bool QuantExtra(OnnxNode n, Tn?[] ins)
    {
        string name = Nm(n);
        if (n.Domain is "" or "ai.onnx" && n.OpType == "DynamicQuantizeLinear")
        {
            // y = clip(round(x / s) + zp, 0, 255) with s = (max(x,0) - min(x,0)) / 255 and zp = clip(round(-min(x,0) / s), 0, 255), over the whole tensor.
            var x = X(ins, 0);
            var all = Enumerable.Range(0, x.Rank).Select(i => (long)i).ToArray();
            var hi = Emit("Clip", [Emit("Reduce", [x], Attributes.From(("mode", "max"), ("axes", all), ("keepdims", true)))], Attributes.From(("min", 0f)));
            var lo = Emit("Clip", [Emit("Reduce", [x], Attributes.From(("mode", "min"), ("axes", all), ("keepdims", true)))], Attributes.From(("max", 0f)));
            var scale = _b.Scale(_b.Sub(hi, lo), 1f / 255f);
            var zpv = Emit("Clip", [Emit("Round", [_b.Binary("Div", _b.Scale(lo, -1f), scale)])], Attributes.From(("min", 0f), ("max", 255f)));
            var y = Emit("Clip", [_b.Add(Emit("Round", [_b.Binary("Div", x, scale)]), zpv)], Attributes.From(("min", 0f), ("max", 255f)), name);
            SetDyn(n, 0, y);
            if (n.Outputs.Count > 1 && n.Outputs[1] != "")
            {
                SetDyn(n, 1, Rs(scale, 1));
            }

            if (n.Outputs.Count > 2 && n.Outputs[2] != "")
            {
                SetDyn(n, 2, Rs(zpv, 1));
            }

            return true;
        }
        if (n.Domain != "com.microsoft" || !s_quantContrib.Contains(n.OpType))
        {
            return false;
        }

        switch (n.OpType)
        {
            case "QLinearAdd":
            case "QLinearMul":
                {
                    var a = Deq(ins[0]!, ins[1]!, Opt(ins, 2), name + ".a");
                    var b = Deq(ins[3]!, ins[4]!, Opt(ins, 5), name + ".b");
                    var r = n.OpType == "QLinearAdd" ? _b.Add(a, b) : _b.Mul(a, b);
                    SetDyn(n, 0, Req(r, ins[6]!, Opt(ins, 7), name));
                    return true;
                }
            case "QLinearSigmoid":
                SetDyn(n, 0, Req(_b.Sigmoid(Deq(ins[0]!, ins[1]!, Opt(ins, 2), name + ".x")), ins[3]!, Opt(ins, 4), name)); return true;
            case "QLinearLeakyRelu":
                SetDyn(n, 0, Req(Emit("LeakyRelu", [Deq(ins[0]!, ins[1]!, Opt(ins, 2), name + ".x")], Attributes.From(("alpha", n.Float("alpha", 0.01f)))), ins[3]!, Opt(ins, 4), name)); return true;
            case "QLinearSoftmax":
                {
                    var sm = new OnnxNode { OpType = "Softmax", Name = n.Name, Inputs = ["x"], Outputs = [name + ".sm"], Attributes = n.Attributes };
                    Dynamic(sm, [Tn.Dyn(Deq(ins[0]!, ins[1]!, Opt(ins, 2), name + ".x"))]);
                    SetDyn(n, 0, Req(_env[name + ".sm"].V!, ins[3]!, Opt(ins, 4), name)); return true;
                }
            case "QLinearGlobalAveragePool":
            case "QLinearAveragePool":
                {
                    var x = Deq(ins[0]!, ins[1]!, Opt(ins, 2), name + ".x");
                    if (n.Int("channels_last", 0) != 0)
                    {
                        x = Permute(x, [0, 3, 1, 2]);
                    }

                    var pool = new OnnxNode { OpType = n.OpType == "QLinearGlobalAveragePool" ? "GlobalAveragePool" : "AveragePool", Name = n.Name, Inputs = ["x"], Outputs = [name + ".pool"], Attributes = n.Attributes };
                    Dynamic(pool, [Tn.Dyn(x)]);
                    var y = _env[name + ".pool"].V!;
                    if (n.Int("channels_last", 0) != 0)
                    {
                        y = Permute(y, [0, 2, 3, 1]);
                    }

                    SetDyn(n, 0, Req(y, ins[3]!, Opt(ins, 4), name)); return true;
                }
            case "QLinearConcat":
                {
                    var parts = new List<Value>();
                    for (int i = 2; i + 2 < ins.Length; i += 3)
                    {
                        parts.Add(Deq(ins[i]!, ins[i + 1]!, Opt(ins, i + 2), name + $".in{i}"));
                    }

                    var cat = Emit("Concat", [.. parts], Attributes.From(("axis", n.Int("axis", 1))));
                    SetDyn(n, 0, Req(cat, ins[0]!, Opt(ins, 1), name)); return true;
                }
        }
        return false;
    }
}
