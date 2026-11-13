namespace Sharplight.Onnx;

// Operator dispatch: routes each ONNX node to the translator of its category.
internal sealed partial class OnnxTranslator
{
    // Operators this translator handles with runtime-dependent inputs; keep in sync with Dynamic and TranslateContrib.
    private static readonly HashSet<string> s_dynamicOps =
    [
        "Conv", "BatchNormalization", "MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool", "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin",
        "ReduceL2", "ReduceSumSquare", "Gemm", "MatMul", "Relu", "Sigmoid", "Tanh", "Exp", "Log", "Sqrt", "Neg", "Abs", "Erf", "Reciprocal", "HardSigmoid",
        "HardSwish", "Clip", "LeakyRelu", "Gelu", "Softplus", "Mish", "PRelu", "Add", "Sub", "Mul", "Div", "Pow", "Max", "Min", "Sum", "Mean", "Flatten",
        "Reshape", "Squeeze", "Unsqueeze", "Transpose", "Concat", "Slice", "Softmax", "LogSoftmax", "LayerNormalization", "InstanceNormalization", "Pad",
        "Resize", "Upsample", "Identity", "Dropout", "Split", "Gather", "Expand", "Shape", "Size", "Cast",
        "Elu", "Celu", "Selu", "ThresholdedRelu", "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Sinh", "Cosh", "Asinh", "Acosh", "Atanh", "Floor", "Ceil", "Round",
        "Sign", "Softsign", "Not", "IsNaN", "IsInf", "Swish", "Silu", "Mod", "Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual", "And", "Or", "Xor", "Where",
        "ReduceL1", "ReduceProd", "ReduceLogSum", "ReduceLogSumExp", "ArgMax", "ArgMin", "TopK", "Hardmax", "GroupNormalization", "LpNormalization", "LRN",
        "MeanVarianceNormalization", "ConvTranspose", "LpPool", "Einsum", "Tile", "GatherElements", "GatherND", "ScatterElements", "ScatterND", "Scatter", "CumSum",
        "DepthToSpace", "SpaceToDepth", "Trilu", "OneHot", "GridSample", "CastLike", "QuantizeLinear", "DequantizeLinear", "MatMulInteger", "ConvInteger",
        "QLinearMatMul", "QLinearConv", "If", "LSTM", "GRU", "RNN", "RMSNormalization", "NonZero", "EyeLike", "DynamicQuantizeLinear",
        "Range", "ConstantOfShape", "NonMaxSuppression", "Loop", "Scan", "Unique", "Compress", "RoiAlign",
    ];

    private static readonly HashSet<string> s_contribOps =
    [
        "Gelu", "BiasGelu", "FastGelu", "QuickGelu", "FusedConv", "FusedMatMul", "SkipLayerNormalization", "SkipSimplifiedLayerNormalization",
        "SimplifiedLayerNormalization", "Attention", "MultiHeadAttention", "BiasAdd", "MatMulNBits", "GroupQueryAttention", "RotaryEmbedding",
    ];

    // Contrib ops that onnxruntime also registers in the default domain (onnxruntime-genai exports use it).
    private static readonly HashSet<string> s_contribInDefaultDomain = ["SimplifiedLayerNormalization"];

    public static bool IsSupported(string opType) => s_dynamicOps.Contains(opType) || OnnxFold.Ops.Contains(opType) || s_contribInDefaultDomain.Contains(opType);

    /// <summary>Translates a node with at least one runtime-dependent input into IR values stored in the environment.</summary>
    private void Dynamic(OnnxNode n, TranslatedTensor?[] ins)
    {
        if (n.Domain == "com.microsoft" || s_contribInDefaultDomain.Contains(n.OpType))
        {
            TranslateContrib(n, ins);
            return;
        }

        string op = n.OpType;
        if (s_unary.TryGetValue(op, out var irUnary))
        {
            Unary(n, ins, irUnary);
            return;
        }

        if (s_unaryAttr.TryGetValue(op, out var unarySpec))
        {
            UnaryWithAttributes(n, ins, unarySpec);
            return;
        }

        if (s_binary.TryGetValue(op, out var irBinary))
        {
            Binary(n, ins, irBinary);
            return;
        }

        switch (op)
        {
            // Elementwise and variadic math.
            case "Mod":
                Mod(n, ins);
                break;
            case "Clip":
                Clip(n, ins);
                break;
            case "Gelu":
                Gelu(n, ins);
                break;
            case "Max" or "Min" or "Sum" or "Mean":
                Variadic(n, ins);
                break;
            case "Where":
                Where(n, ins);
                break;
            case "Identity":
                Identity(n, ins);
                break;
            case "Dropout":
                Dropout(n, ins);
                break;
            case "Cast":
                Cast(n, ins);
                break;
            case "CastLike":
                CastLike(n, ins);
                break;

            // Spatial.
            case "Conv":
                Conv(n, ins);
                break;
            case "ConvTranspose":
                ConvTranspose(n, ins);
                break;
            case "BatchNormalization":
                BatchNorm(n, ins);
                break;
            case "MaxPool" or "AveragePool":
                Pool(n, ins);
                break;
            case "GlobalAveragePool" or "GlobalMaxPool":
                GlobalPool(n, ins);
                break;
            case "LpPool":
                LpPool(n, ins);
                break;
            case "Pad":
                Pad(n, ins);
                break;
            case "Resize" or "Upsample":
                Resize(n, ins);
                break;
            case "DepthToSpace" or "SpaceToDepth":
                DepthSpace(n, ins);
                break;
            case "RoiAlign":
                RoiAlign(n, ins);
                break;
            case "GridSample":
                GridSample(n, ins);
                break;

            // Reductions and selection.
            case "ReduceMean" or "ReduceSum" or "ReduceMax" or "ReduceMin" or "ReduceL2" or "ReduceL1" or "ReduceProd"
                or "ReduceSumSquare" or "ReduceLogSum" or "ReduceLogSumExp":
                Reduce(n, ins);
                break;
            case "ArgMax" or "ArgMin":
                ArgReduce(n, ins);
                break;
            case "TopK":
                TopK(n, ins);
                break;
            case "NonMaxSuppression":
                NonMaxSuppression(n, ins);
                break;
            case "NonZero":
                NonZero(n, ins);
                break;
            case "Unique":
                Unique(n, ins);
                break;
            case "Compress":
                Compress(n, ins);
                break;
            case "CumSum":
                CumSum(n, ins);
                break;

            // Linear algebra.
            case "Gemm":
                Gemm(n, ins);
                break;
            case "MatMul":
                MatMul(n, ins);
                break;
            case "Einsum":
                Einsum(n, ins);
                break;

            // Shape manipulation.
            case "Shape":
                ShapeOf(n, ins);
                break;
            case "Size":
                SizeOf(n, ins);
                break;
            case "Flatten":
                Flatten(n, ins);
                break;
            case "Reshape":
                Reshape(n, ins);
                break;
            case "Squeeze" or "Unsqueeze":
                SqueezeUnsqueeze(n, ins);
                break;
            case "Transpose":
                Transpose(n, ins);
                break;
            case "Concat":
                Concat(n, ins);
                break;
            case "Slice":
                Slice(n, ins);
                break;
            case "Split":
                Split(n, ins);
                break;
            case "Expand":
                Expand(n, ins);
                break;
            case "Range":
                Range(n, ins);
                break;
            case "ConstantOfShape":
                ConstantOfShape(n, ins);
                break;
            case "Tile":
                Tile(n, ins);
                break;
            case "Trilu":
                Trilu(n, ins);
                break;

            // Indexing.
            case "Gather":
                Gather(n, ins);
                break;
            case "GatherElements":
                GatherElements(n, ins);
                break;
            case "GatherND":
                GatherNd(n, ins);
                break;
            case "ScatterElements" or "Scatter":
                ScatterElements(n, ins);
                break;
            case "ScatterND":
                ScatterNd(n, ins);
                break;
            case "OneHot":
                OneHot(n, ins);
                break;

            // Normalization and softmax family.
            case "Softmax" or "LogSoftmax":
                Softmax(n, ins);
                break;
            case "Hardmax":
                Hardmax(n, ins);
                break;
            case "LayerNormalization":
                LayerNorm(n, ins);
                break;
            case "InstanceNormalization":
                InstanceNorm(n, ins);
                break;
            case "GroupNormalization":
                GroupNorm(n, ins);
                break;
            case "RMSNormalization":
                RmsNorm(n, ins);
                break;
            case "LpNormalization":
                LpNormalization(n, ins);
                break;
            case "LRN":
                LocalResponseNorm(n, ins);
                break;
            case "MeanVarianceNormalization":
                MeanVarianceNorm(n, ins);
                break;

            // Quantization.
            case "QuantizeLinear" or "DequantizeLinear":
                QuantizeDequantize(n, ins);
                break;
            case "DynamicQuantizeLinear":
                DynamicQuantize(n, ins);
                break;
            case "MatMulInteger":
                MatMulInteger(n, ins);
                break;
            case "QLinearMatMul":
                QLinearMatMul(n, ins);
                break;
            case "ConvInteger" or "QLinearConv":
                QuantizedConv(n, ins);
                break;

            // Control flow and recurrent networks.
            case "If":
                IfOp(n, ins);
                break;
            case "Loop":
                LoopOp(n, ins);
                break;
            case "Scan":
                ScanOp(n, ins);
                break;
            case "LSTM" or "GRU" or "RNN":
                Rnn(n, ins);
                break;

            case "EyeLike":
                throw new NotSupportedException($"{op} of a runtime-dependent tensor has a data-dependent shape");
            default:
                throw new NotSupportedException($"{op} on a runtime-dependent tensor is not supported (only constant-foldable uses are).");
        }
    }
}
