using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    private static readonly HashSet<string> s_elementwise = [.. StandardOps.UnaryNames, .. StandardOps.BinaryNames, "Scale", "Cast", "Clip", "Where"];

    private static readonly HashSet<string> s_supported =
        [.. s_elementwise, "Reshape", "Slice", "Permute", "Contiguous", "Concat", "Gather", "MatMul", "RmsNorm", "L2Norm",
         "Rope", "CacheWrite", "CachedAttention", "Attention", "CausalConv1d", "GatedDeltaRule", "SelectiveScan", "Wkv7", "Wkv6", "MatMulId", "SequencePool", "Sample", "NGramHash", .. SpatialOpNames, .. GeneralOpNames];

    /// <summary>Kernel weight-format ids (the <c>WFMT</c> macro of the kernel library) per tensor type.</summary>
    private static readonly Dictionary<DataType, int> s_weightFormats = new()
    {
        [DataType.F32] = 0,
        [DataType.F16] = 1,
        [DataType.BF16] = 2,
        [DataType.Q8_0] = 8,
        [ShaderTypes.Q8_0R] = 108,
        [DataType.Q4_0] = 2001,
        [DataType.Q4_K] = 12,
        [DataType.Q5_K] = 13,
        [DataType.Q6_K] = 14,
        [DataType.Q4_1] = 3,
        [DataType.Q5_0] = 6,
        [DataType.Q5_1] = 7,
        [DataType.IQ4_NL] = 20,
        [DataType.IQ4_XS] = 23,
        [DataType.Q1_0] = 41,
        [DataType.Q2_0] = 42,
        [DataType.Q2_0_G128] = 142,
        [DataType.Tern1_G128] = 143,
        [DataType.Q2_K] = 10,
        [DataType.Q3_K] = 11,
        [DataType.IQ2_XXS] = 16,
        [DataType.IQ2_XS] = 17,
        [DataType.IQ3_XXS] = 18,
        [DataType.IQ1_S] = 19,
        [DataType.IQ3_S] = 21,
        [DataType.IQ2_S] = 22,
        [DataType.IQ1_M] = 29,
        [DataType.TQ1_0] = 34,
        [DataType.TQ2_0] = 35,
        [DataType.MXFP4] = 39,
        [ShaderTypes.MXFP4R] = 139,
        [DataType.NVFP4] = 40,
        [DataType.F8E4M3] = 50,
        [DataType.F8E5M2] = 51,
        [DataType.F8E4M3_B32] = 52,
        [DataType.F8E4M3_B128] = 53,
    };

    /// <summary>Tells whether the lowering has a kernel for <paramref name="node"/> (and, if not, why), so callers can partition unsupported nodes to another provider.</summary>
    /// <param name="node">The node to check, with its static input and output types and shapes.</param>
    /// <returns><see cref="OpSupport.Yes"/>, or a refusal with a human-readable reason.</returns>
    public static OpSupport Supports(Node node)
    {
        string op = node.Op.Name;
        if (!s_supported.Contains(op))
        {
            return OpSupport.No($"no kernel for {op}");
        }

        if (s_spatialOps.Contains(op))
        {
            return SupportsSpatial(node);
        }

        if (s_generalOps.Contains(op))
        {
            return SupportsGeneral(node);
        }

        if (op is "Wkv7" or "Wkv6")
        {
            return SupportsWkv(node, op == "Wkv7" ? 7 : 6);
        }

        if (op == "SelectiveScan")
        {
            return SupportsSelectiveScan(node);
        }

        if (op == "MatMulId")
        {
            return SupportsMoe(node);
        }

        var opSpecific = op switch
        {
            "CachedAttention" => SupportsCachedAttention(node),
            "CacheWrite" => SupportsKvRows(node.Inputs[0]),
            "Attention" => SupportsAttention(node),
            "SequencePool" => SupportsSequencePool(node),
            "Gather" => node.Inputs[0].Shape[-1] is { IsSymbolic: false } d && d.Fixed % 8 == 0 ? OpSupport.Yes : OpSupport.No("Gather: rows must be a multiple of 8 values"),
            "MatMul" => SupportsMatMul(node),
            "Rope" => SupportsRope(node),
            "Concat" when node.Output.Shape[0].IsSymbolic && node.Inputs.Any(i => i.Shape[0] != node.Output.Shape[0])
                => OpSupport.No("Concat along the symbolic axis (offsets known only at run time)"),
            _ => OpSupport.Yes,
        };
        if (!opSpecific.IsSupported)
        {
            return opSpecific;
        }

        foreach (var v in node.Inputs)
        {
            if (v.Type.IsQuantized && op is not ("MatMul" or "Gather" or "CacheWrite" or "CachedAttention"))
            {
                return OpSupport.No($"{op} cannot read quantized {v.Type}");
            }

            if (v.Type.IsQuantized && WeightFormat(v.Type) is null)
            {
                return OpSupport.No($"weight format {v.Type} not supported");
            }
        }

        return OpSupport.Yes;
    }

    private static OpSupport SupportsCachedAttention(Node node)
    {
        if (node.Attributes.Has("alibi_slopes"))
        {
            return OpSupport.No("CachedAttention: ALiBi score bias is not implemented in the attention kernels");
        }

        if (node.Attributes.Get("sinks", false) && node.Inputs[3].Type != DataType.F32)
        {
            return OpSupport.No("attention sinks must be f32");
        }

        // The key dot products read 8 (f16) / 4 (f32) elements per load.
        if (node.Inputs[0].Shape[2] is { IsSymbolic: false } keyDims && keyDims.Fixed % 8 != 0)
        {
            return OpSupport.No($"CachedAttention: key head size {keyDims.Fixed} is not a multiple of 8");
        }

        if (node.Inputs[1].Type != node.Inputs[2].Type)
        {
            return OpSupport.No($"CachedAttention: K and V caches of different types ({node.Inputs[1].Type}, {node.Inputs[2].Type})");
        }

        return SupportsKvRows(node.Inputs[1]) is { IsSupported: false } keys ? keys : SupportsKvRows(node.Inputs[2]);
    }

    /// <summary>KV-cache storage the kernels read and write: f32, f16, or q8_0 / q4_0 rows of whole 32-value blocks.</summary>
    private static OpSupport SupportsKvRows(Value cache)
    {
        if (cache.Type == DataType.F32 || cache.Type == DataType.F16)
        {
            return OpSupport.Yes;
        }

        if (!IsQuantizedKv(cache.Type))
        {
            return OpSupport.No($"KV cache type {cache.Type}");
        }

        return cache.Shape[-1] is { IsSymbolic: false } rowDims && rowDims.Fixed % 32 == 0
            ? OpSupport.Yes
            : OpSupport.No($"{cache.Type} KV cache rows must be a multiple of 32 values");
    }

    /// <summary>Bidirectional attention over plain K/V tensors, optionally with i32 [rows, 2] key spans.</summary>
    private static OpSupport SupportsAttention(Node node)
    {
        if (node.Attributes.Has("alibi_slopes"))
        {
            return OpSupport.No("Attention: ALiBi score bias is not implemented in the attention kernels");
        }

        var inputs = node.Inputs;
        bool spans = inputs.Length == 4 && inputs[3].Type == DataType.I32 && inputs[3].Rank == 2 && inputs[3].Shape[1] == 2;
        bool causal = node.Attributes.Get("causal", false);
        if (inputs.Length != 3 && !spans)
        {
            return OpSupport.No("Attention: an f32 mask input is not implemented (i32 spans are)");
        }

        bool headsMatch = inputs[1].Shape[1] == inputs[2].Shape[1];
        bool headSizesAligned = inputs[0].Shape[2].Fixed % 8 == 0 && inputs[1].Shape[2].Fixed % 8 == 0 && inputs[2].Shape[2].Fixed % 8 == 0;
        if (!headsMatch || !headSizesAligned)
        {
            return OpSupport.No("Attention: k/v head counts must match and head sizes be multiples of 8");
        }

        if (inputs[0].Shape[1].Fixed % inputs[1].Shape[1].Fixed != 0)
        {
            return OpSupport.No("Attention: query heads must be a multiple of key heads");
        }

        bool selfAttention = inputs[0].Shape[0] == inputs[1].Shape[0];
        if (!spans && (causal || !selfAttention))
        {
            return OpSupport.No("Attention: causal or cross attention needs i32 spans");
        }

        if (spans && causal && !selfAttention)
        {
            return OpSupport.No("Attention: causal attention must be self-attention");
        }

        return OpSupport.Yes;
    }

    private static OpSupport SupportsSequencePool(Node node)
    {
        bool validMode = node.Attributes.Get("mode", "mean") is "mean" or "sum" or "max" or "first" or "last";
        if (node.Inputs[1].Type != DataType.I32 || node.Inputs[0].Type.IsQuantized || !validMode)
        {
            return OpSupport.No("SequencePool: i32 spans and mode mean | sum | max | first | last");
        }

        return OpSupport.Yes;
    }

    private static OpSupport SupportsMatMul(Node node)
    {
        if (node.Inputs[1].Type.IsQuantized && node.Inputs[1].Shape[-1] is { IsSymbolic: false } k && k.Fixed % 8 != 0)
        {
            return OpSupport.No($"quantized MatMul K={k.Fixed} is not a multiple of 8");
        }

        return OpSupport.Yes;
    }

    private static OpSupport SupportsRope(Node node)
    {
        if (node.Inputs.Length > 2 && (node.Inputs[2].Type != DataType.F32 || node.Inputs[2].Shape.Rank != 1))
        {
            return OpSupport.No("Rope frequency factors must be a rank-1 f32 tensor");
        }

        return OpSupport.Yes;
    }

    /// <summary>The kernel weight-format id of <paramref name="t"/>, or null when no kernel reads that type.</summary>
    private static int? WeightFormat(DataType t) => s_weightFormats.TryGetValue(t, out int format) ? format : null;
}
