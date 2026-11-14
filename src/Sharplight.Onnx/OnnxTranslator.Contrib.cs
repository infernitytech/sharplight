using Sharplight.Graph;

namespace Sharplight.Onnx;

// com.microsoft contrib operators.
internal sealed partial class OnnxTranslator
{
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
            case "FusedMatMul" or "TransposeMatMul":
                FusedMatMul(n, ins);
                break;
            case "FusedGemm":
                FusedGemm(n, ins);
                break;
            case "GemmFastGelu":
                GemmFastGelu(n, ins);
                break;
            case "ScaledSiLU":
                ScaledSilu(n, ins);
                break;
            case "BiasSplitGelu":
                BiasSplitGelu(n, ins);
                break;
            case "GatedRMSNorm":
                GatedRmsNorm(n, ins);
                break;
            case "CDist":
                CDist(n, ins);
                break;
            case "ComplexMul" or "ComplexMulConj":
                ComplexMul(n, ins, n.OpType == "ComplexMulConj");
                break;
            case "GroupNorm" or "SkipGroupNorm":
                ContribGroupNorm(n, ins, n.OpType == "SkipGroupNorm");
                break;
            case "NhwcMaxPool":
                NhwcMaxPool(n, ins);
                break;
            case "UnfoldTensor":
                UnfoldTensor(n, ins);
                break;
            case "EmbedLayerNormalization":
                EmbedLayerNorm(n, ins);
                break;
            case "CropAndResize":
                CropAndResize(n, ins);
                break;
            case "MoE":
                Moe(n, ins);
                break;
            case "GatherBlockQuantized":
                GatherBlockQuantized(n, ins);
                break;
            case "NGramRepeatBlock":
                NGramRepeatBlock(n, ins);
                break;
            case "Inverse":
                Inverse(n, ins);
                break;
            case "ExpandDims":
                ExpandDims(n, ins);
                break;
            case "MatMulIntegerToFloat":
                MatMulIntegerToFloat(n, ins);
                break;
            case "DynamicQuantizeMatMul":
                DynamicQuantizeMatMul(n, ins);
                break;
            case "QGemm":
                QGemm(n, ins);
                break;
            case "MatMulInteger16":
                MatMulInteger16(n, ins);
                break;
            case "QLinearWhere":
                QLinearWhere(n, ins);
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
            case "GroupQueryAttention":
                GroupQueryAttention(n, ins);
                break;
            case "RotaryEmbedding":
                RotaryEmbedding(n, ins);
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
        if (Wants(n, 1) || Wants(n, 2))
        {
            var (mean, invStd) = NormStatistics(sum, sum.Rank - 1, n.Float("epsilon", 1e-12f), simplified);
            if (Wants(n, 1) && mean is not null)
            {
                SetDyn(n, 1, mean);
            }

            if (Wants(n, 2))
            {
                SetDyn(n, 2, invStd);
            }
        }

        if (Wants(n, 3))
        {
            SetDyn(n, 3, sum);
        }
    }
}
