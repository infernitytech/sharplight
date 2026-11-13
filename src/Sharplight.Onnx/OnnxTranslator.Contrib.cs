using System.Buffers.Binary;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// com.microsoft contrib operators.
internal sealed partial class OnnxTranslator
{
    private Value AttentionCore(Value q, Value k, Value v, int heads, float scale, bool causal, string name)
    {
        long hidden = q.Shape[2].Fixed;
        long dh = hidden / heads;
        long S = q.Shape[1].Fixed;
        long Sk = k.Shape[1].Fixed;
        long vh = v.Shape[2].Fixed / heads;
        if (scale == 0)
        {
            scale = 1f / MathF.Sqrt(dh);
        }

        Value Heads(Value x, long d, long len) => _b.Permute(Rs(x, 0, len, heads, d), 0, 2, 1, 3);
        var qh = Heads(q, dh, S);
        var kh = Heads(k, dh, Sk);
        var vv = Heads(v, vh, Sk);
        var scores = Emit("BatchMatMul", [qh, kh], Attributes.From(("trans_b", true), ("alpha", scale)));
        Heavy();
        if (causal)
        {
            var mask = new byte[S * Sk * 4];
            for (long i = 0; i < S; i++)
            {
                for (long j = 0; j < Sk; j++)
                {
                    BinaryPrimitives.WriteSingleLittleEndian(mask.AsSpan((int)((i * Sk + j) * 4), 4), j > i + (Sk - S) ? -1e9f : 0f);
                }
            }

            scores = _b.Add(scores, _b.Constant(name + ".causal", DataType.F32, new Shape(S, Sk), mask));
        }
        var p = _b.Emit1("Softmax", [scores]);
        var o = Emit("BatchMatMul", [p, vv]);
        Heavy();
        return Emit("Reshape", [_b.Permute(o, 0, 2, 1, 3)], Attributes.From(("shape", L(0, S, heads * vh))), name);
    }

    private void ContribAttention(OnnxNode n, TranslatedTensor?[] ins)
    {
        if ((ins.Length > 3 && ins[3] is not null) || (ins.Length > 4 && ins[4] is not null) || (ins.Length > 5 && ins[5] is not null))
        {
            throw new NotSupportedException("Attention with mask_index, past or relative position bias");
        }

        if (n.Has("qkv_hidden_sizes"))
        {
            throw new NotSupportedException("Attention with unequal qkv_hidden_sizes");
        }

        var x = X(ins, 0);
        var w = ConstIn(ins, 1, "Attention weights");
        var qkv = _b.MatMul(x, WeightValue(w.Transposed([1, 0]), true, Nm(n) + ".qkv"));
        Heavy();
        if (ins.Length > 2 && ins[2] is not null)
        {
            qkv = _b.Add(qkv, X(ins, 2));
        }

        long h = qkv.Shape[2].Fixed / 3;
        SetDyn(n, 0, AttentionCore(_b.Slice(qkv, 2, 0, h), _b.Slice(qkv, 2, h, h), _b.Slice(qkv, 2, 2 * h, h), (int)n.Int("num_heads", 1), n.Has("scale") ? n.Float("scale", 0) : 0f, n.Int("unidirectional", 0) != 0, Nm(n)));
    }

    /// <summary>Translates a node of the com.microsoft domain.</summary>
    private void TranslateContrib(OnnxNode n, TranslatedTensor?[] ins)
    {
        switch (n.OpType)
        {
            case "Gelu":
                SetDyn(n, 0, Emit("Gelu", [X(ins, 0)], null, Nm(n)));
                break;
            case "BiasGelu":
                SetDyn(n, 0, Emit("Gelu", [_b.Add(X(ins, 0), X(ins, 1))], null, Nm(n)));
                break;
            case "FastGelu":
                FastGelu(n, ins);
                break;
            case "QuickGelu":
                QuickGelu(n, ins);
                break;
            case "BiasAdd":
                BiasAdd(n, ins);
                break;
            case "FusedConv":
                FusedConv(n, ins);
                break;
            case "FusedMatMul":
                FusedMatMul(n, ins);
                break;
            case "SkipLayerNormalization" or "SkipSimplifiedLayerNormalization":
                SkipLayerNorm(n, ins);
                break;
            case "SimplifiedLayerNormalization":
                RmsNorm(n, ins);
                break;
            case "MatMulNBits":
                MatMulNBits(n, ins);
                break;
            case "Attention":
                ContribAttention(n, ins);
                break;
            case "MultiHeadAttention":
                MultiHeadAttention(n, ins);
                break;
            default:
                if (!s_quantContrib.Contains(n.OpType))
                {
                    throw new NotSupportedException(n.OpType);
                }

                TranslateQuantContrib(n, ins);
                break;
        }
    }

    private void FastGelu(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        if (ins.Length > 1 && ins[1] is not null)
        {
            x = _b.Add(x, X(ins, 1));
        }

        SetDyn(n, 0, Emit("GeluTanh", [x], null, Nm(n)));
    }

    private void QuickGelu(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        var gate = _b.Sigmoid(_b.Scale(x, n.Float("alpha", 1.702f)));
        SetDyn(n, 0, Emit("Mul", [x, gate], null, Nm(n)));
    }

    private void BiasAdd(OnnxNode n, TranslatedTensor?[] ins)
    {
        var y = _b.Add(X(ins, 0), X(ins, 1));
        if (ins.Length > 2 && ins[2] is not null)
        {
            y = _b.Add(y, X(ins, 2));
        }

        SetDyn(n, 0, y);
    }

    /// <summary>Conv with an optional fused residual input and activation.</summary>
    private void FusedConv(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        var conv = new OnnxNode { OpType = "Conv", Name = n.Name, Inputs = n.Inputs.Take(3).ToList(), Outputs = [name + ".conv"], Attributes = n.Attributes };
        Conv(conv, ins.Take(3).ToArray());
        var y = _env[name + ".conv"].V!;
        if (ins.Length > 3 && ins[3] is not null)
        {
            y = _b.Add(y, X(ins, 3));
        }

        y = Act(n.Has("activation") ? n.Str("activation", "") : null, y, n.Floats("activation_params"));
        SetDyn(n, 0, y);
    }

    private void FusedMatMul(OnnxNode n, TranslatedTensor?[] ins)
    {
        var attributes = Attributes.From(("trans_a", n.Int("transA", 0) != 0), ("trans_b", n.Int("transB", 0) != 0), ("alpha", n.Float("alpha", 1f)));
        SetDyn(n, 0, Emit("BatchMatMul", [X(ins, 0), X(ins, 1)], attributes, Nm(n)));
        Heavy();
    }

    /// <summary>Residual add (plus optional bias) followed by LayerNorm or RMSNorm; output #3 is the pre-norm sum.</summary>
    private void SkipLayerNorm(OnnxNode n, TranslatedTensor?[] ins)
    {
        string name = Nm(n);
        var sum = _b.Add(X(ins, 0), X(ins, 1));
        bool simplified = n.OpType == "SkipSimplifiedLayerNormalization";
        int biasIndex = simplified ? 3 : 4;
        if (ins.Length > biasIndex && ins[biasIndex] is not null)
        {
            sum = _b.Add(sum, X(ins, biasIndex));
        }

        var attributes = Attributes.From(("eps", n.Float("epsilon", 1e-12f)));
        Value y;
        if (simplified)
        {
            y = Emit("RmsNorm", [sum, X(ins, 2)], attributes, name);
        }
        else
        {
            var args = new List<Value> { sum, X(ins, 2) };
            if (ins.Length > 3 && ins[3] is not null)
            {
                args.Add(X(ins, 3));
            }

            y = Emit("LayerNorm", [.. args], attributes, name);
        }

        SetDyn(n, 0, y);
        if (n.Outputs.Count > 3 && n.Outputs[3] != "")
        {
            SetDyn(n, 3, sum);
        }
    }

    private void MultiHeadAttention(OnnxNode n, TranslatedTensor?[] ins)
    {
        if (ins.Length > 4 && ins.Skip(4).Any(t => t is not null))
        {
            throw new NotSupportedException("MultiHeadAttention with mask/past/bias inputs beyond bias");
        }

        var q = X(ins, 0);
        var k = X(ins, 1);
        var v = X(ins, 2);
        if (ins.Length > 3 && ins[3] is not null)
        {
            long hidden = q.Shape[2].Fixed;
            var bias = X(ins, 3);
            q = _b.Add(q, _b.Slice(bias, 0, 0, hidden));
            k = _b.Add(k, _b.Slice(bias, 0, hidden, hidden));
            v = _b.Add(v, _b.Slice(bias, 0, 2 * hidden, hidden));
        }

        float scale = n.Has("scale") ? n.Float("scale", 0) : 0f;
        SetDyn(n, 0, AttentionCore(q, k, v, (int)n.Int("num_heads", 1), scale, false, Nm(n)));
    }
}
