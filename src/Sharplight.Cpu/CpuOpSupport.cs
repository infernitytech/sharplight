using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>
/// Decides which graph nodes the CPU provider can run. The checks mirror the lowering in
/// <see cref="CpuCompiledSegment"/>: every op accepted here must be lowerable there. They are split by op category
/// into the partial files <c>CpuOpSupport.*.cs</c>.
/// </summary>
internal static partial class CpuOpSupport
{
    private static readonly HashSet<string> s_unaryOps = new(StandardOps.UnaryNames);
    private static readonly HashSet<string> s_binaryOps = new(StandardOps.BinaryNames);
    private static readonly HashSet<string> s_rowOps = ["RmsNorm", "L2Norm", "LayerNorm", "Softmax", "ReduceSum", "ReduceMax", "ArgMax"];

    /// <summary>Whether the CPU provider can run <paramref name="node"/>, and if not, why.</summary>
    public static OpSupport Check(Node node)
    {
        string op = node.Op.Name;
        if (s_unaryOps.Contains(op) || s_binaryOps.Contains(op) || s_rowOps.Contains(op))
        {
            return AllF32(node);
        }

        return op switch
        {
            "Scale" or "Clip" or "BatchMatMul" or "CumSum" or "ArgReduce" or "TopK" or "GridSample" => AllF32(node),
            "Conv" => CheckConv(node),
            "ConvTranspose" => CheckConvTranspose(node),
            "Pool" => CheckPool(node),
            "PoolArgMax" => CheckPool(node),
            "MaxRoiPool" or "DeformConv" => AllF32(node),
            "Reduce" => CheckReduce(node),
            "Pad" => CheckPad(node),
            "Resize" => CheckResize(node),
            "GroupNorm" => CheckGroupNorm(node),
            "Where" => CheckWhere(node),
            "Expand" or "Tile" => CheckPlainInput(node),
            "GatherAxis" or "GatherElements" or "GatherND" => CheckGatherFamily(node),
            "ScatterElements" or "ScatterND" => CheckScatter(node),
            "Trilu" => CheckTrilu(node),
            "OneHot" => CheckOneHot(node),
            "Reshape" or "Slice" or "Permute" => OpSupport.Yes,
            "Contiguous" or "Transfer" => node.Inputs[0].Type.IsQuantized ? OpSupport.No($"{op} of quantized data") : OpSupport.Yes,
            "Concat" => CheckConcat(node),
            "Cast" => CheckCast(node),
            "MatMul" => CheckMatMul(node),
            "MatMulId" => CheckMatMulId(node),
            "Gather" => CheckGather(node),
            "Rope" => CheckRope(node),
            "CacheWrite" => CheckCacheWrite(node),
            "CachedAttention" => CheckCachedAttention(node),
            "Attention" => CheckAttention(node),
            "SequencePool" => CheckSequencePool(node),
            "NGramHash" => node.Inputs[0].Type == DataType.F32 && node.Inputs[0].Rank == 2 ? OpSupport.Yes : OpSupport.No("NGramHash window must be f32 [T, N]"),
            "CausalConv1d" => CheckCausalConv1d(node),
            "SelectiveScan" => CheckSelectiveScan(node),
            "GatedDeltaRule" => CheckGatedDeltaRule(node),
            "Wkv7" => CheckWkv(node, 7),
            "Wkv6" => CheckWkv(node, 6),
            "Fill" => CheckFill(node),
            "NonZero" or "Unique" => CheckDataDependentInput(node),
            "NonMaxSuppression" => AllF32(node),
            "RoiAlign" => node.Inputs[0].Type == DataType.F32 && node.Inputs[1].Type == DataType.F32 && node.Inputs[2].Type == DataType.I32
                ? OpSupport.Yes : OpSupport.No("RoiAlign expects f32 x and rois and i32 batch indices"),
            "Sample" => OpSupport.Yes,
            _ => OpSupport.No($"op '{op}' is not implemented by the CPU provider"),
        };
    }

    private static bool IsStatic(Value v) => v.Kind is ValueKind.Weight or ValueKind.Constant;

    /// <summary>Plain f32 data, or a weight/constant the provider can dequantize to f32.</summary>
    private static bool F32Like(Value v) => v.Type == DataType.F32 || (IsStatic(v) && Quant.FormatOf(v.Type) is not null);

    /// <summary>A non-quantized element type of 1, 2, 4 or 8 bytes that can be moved around without interpretation.</summary>
    private static bool Plain(Value v) => !v.Type.IsQuantized && v.Type.BlockSize == 1 && v.Type.BlockBytes is 1 or 2 or 4 or 8;

    private static bool IsIndex(Value v) => v.Type == DataType.I32 || v.Type == DataType.I64;

    private static bool IsKvCacheType(DataType t) => t == DataType.F32 || t == DataType.F16;

    /// <summary>Quantized KV-cache storage: q8_0 / q4_0 rows of whole 32-value blocks.</summary>
    private static bool IsQuantizedKvCache(Value cache) => (cache.Type == DataType.Q8_0 || cache.Type == DataType.Q4_0)
        && cache.Shape[-1] is { IsSymbolic: false } rowDims && rowDims.Fixed % 32 == 0;

    private static OpSupport AllF32(Node node, string what = "inputs")
    {
        foreach (var v in node.Inputs)
        {
            if (!F32Like(v))
            {
                return OpSupport.No($"{node.Op.Name}: {what} must be f32 (or a convertible weight), got {v.Type} for {v.Name}");
            }
        }

        return OpSupport.Yes;
    }

    /// <summary>The first input is the activation of a spatial op and must be f32.</summary>
    private static OpSupport RequireF32Input(Node node)
    {
        var type = node.Inputs[0].Type;
        return type == DataType.F32 ? OpSupport.Yes : OpSupport.No($"{node.Op.Name} input must be f32, got {type}");
    }

    /// <summary>The weight must be a static tensor whose element count is a whole number of quantization blocks.</summary>
    private static OpSupport CheckStaticWeight(string op, Value weight)
    {
        if (!IsStatic(weight))
        {
            return OpSupport.No($"{op} weight must be a weight or constant (not an activation)");
        }

        var format = Quant.FormatOf(weight.Type);
        if (format is null)
        {
            return OpSupport.No($"{op} weight type {weight.Type} is not supported by the CPU provider");
        }

        long elements = weight.Shape.Dims.Aggregate(1L, (acc, d) => acc * d.Fixed);
        if (elements % Quant.BlockSize(format.Value) != 0)
        {
            return OpSupport.No($"{op} weight has {elements} elements, not a multiple of the {weight.Type} block size");
        }

        return OpSupport.Yes;
    }

    private static OpSupport CheckCast(Node node)
    {
        var source = node.Inputs[0].Type;
        var target = node.Output.Type;
        return IsCastable(source) && IsCastable(target) ? OpSupport.Yes : OpSupport.No($"Cast {source} -> {target}");
    }

    private static bool IsCastable(DataType t)
    {
        return t == DataType.F32 || t == DataType.F16 || t == DataType.BF16 || t == DataType.I32
            || t == DataType.I64 || t == DataType.I8 || t == DataType.U8 || t == DataType.Bool;
    }

    private static OpSupport CheckConcat(Node node)
    {
        var ins = node.Inputs;
        if (ins.Any(v => v.Type.IsQuantized))
        {
            return OpSupport.No("Concat of quantized data");
        }

        return ins.Any(v => v.Type != ins[0].Type) ? OpSupport.No("Concat inputs must share a type") : OpSupport.Yes;
    }

    private static OpSupport CheckPlainInput(Node node)
    {
        var input = node.Inputs[0];
        return Plain(input) ? OpSupport.Yes : OpSupport.No($"{node.Op.Name} on {input.Type}");
    }

    private static OpSupport CheckWhere(Node node)
    {
        var ins = node.Inputs;
        var condition = ins[0].Type;
        if (!(condition == DataType.F32 || condition == DataType.Bool || condition == DataType.I32 || condition == DataType.U8))
        {
            return OpSupport.No($"Where condition must be an f32/bool/i32 mask, got {condition}");
        }

        if (ins[1].Type != ins[2].Type)
        {
            return OpSupport.No("Where branches must share a type");
        }

        return Plain(ins[1]) ? OpSupport.Yes : OpSupport.No($"Where on {ins[1].Type}");
    }
}
