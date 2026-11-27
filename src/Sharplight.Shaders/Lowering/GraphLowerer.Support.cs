using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    private static readonly HashSet<string> s_elementwise = [.. StandardOps.UnaryNames, .. StandardOps.BinaryNames, "Scale", "Cast", "Clip", "Where", .. NumericElementwiseNames];

    private static readonly HashSet<string> s_supported =
        [.. s_elementwise, "Reshape", "Slice", "Permute", "After", "Contiguous", "Concat", "Gather", "MatMul", "RmsNorm", "L2Norm",
         "Rope", "CacheWrite", "CachedAttention", "IndexerTopK", "Attention", "CausalConv1d", "GatedDeltaRule", "SelectiveScan", "Wkv7", "Wkv6", "MatMulId", "SequencePool", "Sample", "NGramHash", "WindowPool", .. SpatialOpNames, .. GeneralOpNames, .. DetectionOpNames, .. ExtendedOpNames];

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
        [DataType.TrellisV4K8_G128] = 144,
        [DataType.TrellisV4K6_G128] = 145,
        [DataType.TrellisV4K7_G128] = 146,
        [DataType.TrellisV2K5_G128] = 147,
        [DataType.TrellisV2K6_G128] = 148,
        [DataType.TrellisV2K7_G128] = 149,
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
        [ShaderTypes.F8E4M3_B32R] = 152,
        [ShaderTypes.F8E4M3_B128R] = 153,
        [ShaderTypes.F8E4M3_B128G] = 154,
        [DataType.NVFP4_GS] = 60,
        [DataType.Q4Z_G32] = 61,
        [DataType.Q4Z_G128] = 62,
        [DataType.Q8Z_G32] = 63,
        [DataType.Q8Z_G128] = 64,
        [DataType.NF4_B64] = 65,
        [DataType.FP4_B64] = 66,
        [DataType.Q8A_G32] = 67,
        [DataType.Q2S_G128] = 68,
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

        if (s_detectionOps.Contains(op))
        {
            return SupportsDetection(node);
        }

        if (s_extendedOps.Contains(op))
        {
            return SupportsExtended(node);
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
            "IndexerTopK" => SupportsIndexerTopK(node),
            "GatedDeltaRule" => SupportsDeltaRule(node),
            "CacheWrite" => node.Attributes.Get("compression", 1L) > 1 && node.Inputs[0].Type != DataType.F32 && node.Inputs[0].Type != DataType.F16
                ? OpSupport.No("CacheWrite: a compressed cache must be f32 or f16") : SupportsKvRows(node.Inputs[0]),
            "WindowPool" => node.Inputs[4].Type == DataType.F32 && node.Inputs[0].Shape[1] is { IsSymbolic: false }
                ? OpSupport.Yes : OpSupport.No("WindowPool: the ring state must be f32 and the channels fixed"),
            "Attention" => SupportsAttention(node),
            "SequencePool" => SupportsSequencePool(node),
            "Gather" => node.Inputs[0].Shape[-1] is { IsSymbolic: false } d && (d.Fixed % 8 == 0 || !node.Inputs[0].Type.IsQuantized) ? OpSupport.Yes : OpSupport.No("Gather: quantized rows must be a multiple of 8 values"),
            "MatMul" => SupportsMatMul(node),
            "Rope" => SupportsRope(node),
            _ when s_elementwise.Contains(op) => SupportsElementwise(node),
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

    /// <summary>Gated delta rule: f32 activations, an f32, f16 or bf16 state.</summary>
    private static OpSupport SupportsDeltaRule(Node node)
    {
        if (node.Inputs.Take(5).FirstOrDefault(v => v.Type != DataType.F32) is { } input)
        {
            return OpSupport.No($"GatedDeltaRule needs f32 inputs, got {input.Type}");
        }

        var state = node.Inputs[5].Type;
        return state == DataType.F32 || state == DataType.F16 || state == DataType.BF16
            ? OpSupport.Yes : OpSupport.No($"GatedDeltaRule state must be f32, f16 or bf16, got {state}");
    }

    private static OpSupport SupportsIndexerTopK(Node node)
    {
        if (node.Inputs[1].Type != DataType.F32 && node.Inputs[1].Type != DataType.F16)
        {
            return OpSupport.No($"IndexerTopK: key cache type {node.Inputs[1].Type} (f32 or f16)");
        }

        var selection = IndexerSelection.From(node.Attributes);
        if (selection.Budget / selection.Group > MaxIndexerGroups)
        {
            return OpSupport.No($"IndexerTopK: more than {MaxIndexerGroups} selected groups per row");
        }

        return node.Inputs[0].Shape[2] is { IsSymbolic: false } keyDims && keyDims.Fixed % 8 == 0 && node.Inputs[0].Shape[1] is { IsSymbolic: false }
            ? OpSupport.Yes
            : OpSupport.No("IndexerTopK: the key size must be a multiple of 8");
    }

    private static OpSupport SupportsCachedAttention(Node node)
    {
        bool compressed = node.Attributes.Has("compression");
        if (compressed)
        {
            var cache = node.Inputs[node.Attributes.Get("sinks", false) ? 4 : 3].Type;
            if (cache != node.Inputs[1].Type || (cache != DataType.F32 && cache != DataType.F16))
            {
                return OpSupport.No("CachedAttention: the compressed cache must be f32 or f16 like the key cache");
            }
        }

        if (node.Attributes.Get<float[]?>("alibi_slopes", null) is { } slopes && slopes.Length != node.Inputs[0].Shape[1].Fixed)
        {
            return OpSupport.No("CachedAttention: ALiBi needs one slope per query head");
        }

        if (node.Attributes.Get<float[]?>("position_bias", null) is { } bias
            && bias.Length != node.Inputs[0].Shape[1].Fixed * (2L * node.Attributes.Get("position_bias_range", 0) + 1))
        {
            return OpSupport.No("CachedAttention: position_bias needs heads x (2 * range + 1) entries");
        }

        if (node.Attributes.Get("sinks", false) && node.Inputs[3].Type != DataType.F32)
        {
            return OpSupport.No("attention sinks must be f32");
        }

        int selectionInput = (node.Attributes.Get("sinks", false) ? 4 : 3) + (compressed ? 1 : 0);
        if (node.Attributes.Get("selection", false)
            && (node.Inputs[selectionInput].Type != DataType.I32 || (!compressed && (node.Attributes.Get("window", 0) > 0 || node.Attributes.Get("chunk", 0) > 0))))
        {
            return OpSupport.No("CachedAttention: the selection must be i32 and cannot be combined with window or chunk");
        }

        int relativeInput = selectionInput + (node.Attributes.Get("selection", false) ? 1 : 0);
        if (node.Attributes.Get("relative_bias", 0) > 0
            && (compressed || node.Inputs.Length <= relativeInput || node.Inputs[relativeInput].Type != DataType.F32 || node.Inputs[relativeInput].Shape.Rank != 3))
        {
            return OpSupport.No("CachedAttention: the relative bias must be f32 [T, HQ, E] and cannot be combined with compression");
        }

        int lookaheadInput = relativeInput + (node.Attributes.Get("relative_bias", 0) > 0 ? 1 : 0);
        if (node.Attributes.Get("lookahead", false)
            && (node.Inputs.Length <= lookaheadInput || node.Inputs[lookaheadInput].Type != DataType.I32 || node.Inputs[lookaheadInput].Shape.Rank != 1
                || !node.Inputs[lookaheadInput].Shape[0].IsSymbolic || !node.Attributes.Get("causal", true) || compressed
                || node.Attributes.Get("selection", false) || node.Attributes.Get("chunk", 0) > 0))
        {
            return OpSupport.No("CachedAttention: the lookahead must be i32 [A] (symbolic) on causal attention without compression, selection or chunks");
        }

        // Head sizes that are not a multiple of 8 run on the per-head kernel, which then reads one key element per load.
        if (node.Inputs[1].Type != node.Inputs[2].Type)
        {
            return OpSupport.No($"CachedAttention: K and V caches of different types ({node.Inputs[1].Type}, {node.Inputs[2].Type})");
        }

        return SupportsKvRows(node.Inputs[1]) is { IsSupported: false } keys ? keys : SupportsKvRows(node.Inputs[2]);
    }

    /// <summary>KV-cache storage the kernels read and write: f32, f16, or q8_0 / q4_0 / f8e4m3_b32 rows of whole 32-value blocks.</summary>
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
        if (node.Attributes.Get<float[]?>("alibi_slopes", null) is { } slopes && slopes.Length != node.Inputs[0].Shape[1].Fixed)
        {
            return OpSupport.No("Attention: ALiBi needs one slope per query head");
        }

        if (node.Attributes.Get<float[]?>("position_bias", null) is { } bias
            && bias.Length != node.Inputs[0].Shape[1].Fixed * (2L * node.Attributes.Get("position_bias_range", 0) + 1))
        {
            return OpSupport.No("Attention: position_bias needs heads x (2 * range + 1) entries");
        }

        var inputs = node.Inputs;
        bool spans = inputs.Length == 4 && inputs[3].Type == DataType.I32 && inputs[3].Rank == 2 && inputs[3].Shape[1] == 2;
        bool mask = inputs.Length == 4 && inputs[3].Type == DataType.F32 && inputs[3].Rank is 1 or 2;
        bool causal = node.Attributes.Get("causal", false);
        if (inputs.Length != 3 && !spans && !mask)
        {
            return OpSupport.No("Attention: the fourth input must be i32 spans [T, 2] or an f32 mask [T, S] or [S]");
        }

        if (BatchedAttention.SequenceRows(node) > 0 && !BatchedAttention.IsValid(node))
        {
            return OpSupport.No("Attention: the batched form needs sequence_keys > 0 and no spans, causal mask, window or position bias");
        }

        bool scoreTerms = causal || node.Attributes.Has("alibi_slopes") || node.Attributes.Has("position_bias") || node.Attributes.Get("window", 0) > 0;
        if (mask && scoreTerms)
        {
            return OpSupport.No("Attention: an f32 mask cannot be combined with causal, window or position biases");
        }

        if (inputs[1].Shape[1] != inputs[2].Shape[1])
        {
            return OpSupport.No("Attention: k/v head counts must match");
        }

        if (inputs[0].Shape[1].Fixed % inputs[1].Shape[1].Fixed != 0)
        {
            return OpSupport.No("Attention: query heads must be a multiple of key heads");
        }

        bool selfAttention = inputs[0].Shape[0] == inputs[1].Shape[0];
        if (!spans && causal && !selfAttention)
        {
            return OpSupport.No("Attention: causal attention must be self-attention");
        }

        if (!spans && causal)
        {
            return OpSupport.No("Attention: causal attention needs i32 spans");
        }

        bool scoreBias = node.Attributes.Has("alibi_slopes") || node.Attributes.Has("position_bias") || node.Attributes.Get("window", 0) > 0;
        if (!spans && !selfAttention && scoreBias)
        {
            return OpSupport.No("Attention: cross-attention takes no position bias or window");
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
