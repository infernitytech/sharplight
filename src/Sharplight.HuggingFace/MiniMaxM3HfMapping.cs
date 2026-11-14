using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>minimax_m3_vl</c> / <c>minimax_m3_vl_text</c> (MiniMax-M3, the language model): Gemma-style (1 + w) RMS norms (also the
/// per-head q/k norms), partial rotary embedding, leading dense layers (<c>mlp_layer_types</c> / <c>moe_layer_freq</c>) and MoE layers with a
/// sigmoid router with selection bias whose weights are renormalized and scaled by <c>routed_scaling_factor</c>, a shared expert, and the
/// clamped SwiGLU-OAI activation with non-interleaved gate and up projections. Both the published layout (<c>language_model.model.*</c>,
/// <c>block_sparse_moe</c> with per-expert w1/w2/w3) and the transformers layout (<c>model.language_model.*</c>, fused <c>gate_up_proj</c>) load.
/// </summary>
/// <remarks>
/// MiniMax sparse attention: the lightning indexer of the sparse layers picks <c>index_topk_blocks</c> key blocks of <c>index_block_size</c>
/// keys per query. While a query sees at most that many blocks every block is picked, so dense attention is exact; block selection beyond
/// that is not implemented yet, so the context is capped there and the indexer weights are not imported. The vision tower is not imported.
/// </remarks>
internal sealed partial class MiniMaxM3HfMapping : HfMoeFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layer = With(LlamaLayerTensors,
        ("block_sparse_moe.gate_proj.weight", "mlp.gate.weight"),
        ("block_sparse_moe.up_proj.weight", "mlp.up.weight"),
        ("block_sparse_moe.down_proj.weight", "mlp.down.weight"),
        ("block_sparse_moe.shared_experts.gate_proj.weight", "mlp.shared.gate.weight"),
        ("block_sparse_moe.shared_experts.up_proj.weight", "mlp.shared.up.weight"),
        ("block_sparse_moe.shared_experts.down_proj.weight", "mlp.shared.down.weight"));

    private static readonly IReadOnlyDictionary<string, string> s_global = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["model.embed_tokens.weight"] = "token_embedding.weight",
        ["model.norm.weight"] = "output_norm.weight",
        ["lm_head.weight"] = "lm_head.weight",
        ["language_model.model.embed_tokens.weight"] = "token_embedding.weight",
        ["language_model.model.norm.weight"] = "output_norm.weight",
        ["language_model.lm_head.weight"] = "lm_head.weight",
        ["model.language_model.embed_tokens.weight"] = "token_embedding.weight",
        ["model.language_model.norm.weight"] = "output_norm.weight",
    };

    // The language model's layers: text-only (model.layers), published composite (language_model.model.layers) or transformers composite
    // (model.language_model.layers).
    [GeneratedRegex(@"^(?:language_model\.)?model\.(?:language_model\.)?layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerNameRegex();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["minimax_m3_vl", "minimax_m3_vl_text"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layer;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_global;

    /// <inheritdoc/>
    protected override Regex LayerName => LayerNameRegex();

    /// <inheritdoc/>
    protected override string Activation(HfConfig c) => "swiglu_oai";

    /// <inheritdoc/>
    protected override int FeedForwardSize(HfConfig c) => c.GetInt("dense_intermediate_size") ?? c.Int("intermediate_size");

    /// <inheritdoc/>
    protected override float NormEpsilon(HfConfig c) => (float)(c.GetDouble("rms_norm_eps") ?? 1e-6);

    /// <inheritdoc/>
    protected override float RopeBase(HfConfig c) => (float)(c.RopeDouble("rope_theta", "rope_theta") ?? 5000000.0);

    /// <inheritdoc/>
    protected override int RopeDimension(HfConfig c, int headDim)
        => c.RopeDouble("partial_rotary_factor", "partial_rotary_factor") is { } factor ? (int)(headDim * factor) : c.GetInt("rotary_dim") ?? headDim;

    /// <inheritdoc/>
    protected override int? ContextLength(HfConfig c)
    {
        int context = c.GetInt("max_position_embeddings") ?? 524288;
        return SparseLayers(c).Any(sparse => sparse) ? Math.Min(context, Sparse(c, "index_topk_blocks", "sparse_topk_blocks", 16) * Sparse(c, "index_block_size", "sparse_block_size", 128)) : context;
    }

    /// <inheritdoc/>
    protected override WeightInfo NormWeight(WeightInfo w, string name) => HfTransforms.NormPlusOne(w, name);

    /// <inheritdoc/>
    protected override bool IsIgnored(string name)
        => base.IsIgnored(name) || name.Contains(".self_attn.index", StringComparison.Ordinal) || name.StartsWith("vision_tower.", StringComparison.Ordinal)
           || name.StartsWith("model.vision_tower.", StringComparison.Ordinal) || name.StartsWith("multi_modal_projector.", StringComparison.Ordinal)
           || name.StartsWith("model.multi_modal_projector.", StringComparison.Ordinal) || name.StartsWith("patch_merge_mlp.", StringComparison.Ordinal);

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
        => new(c.Int("num_local_experts", "num_experts"), c.Int("num_experts_per_tok"), c.Int("intermediate_size"), "sigmoid", Normalize: true,
            Scale: (float)(c.GetDouble("routed_scaling_factor") ?? 1.0));

    /// <inheritdoc/>
    protected override void AddFamilyHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        md["decoder.mlp.swiglu_alpha"] = (float)(c.GetDouble("swiglu_alpha") ?? 1.702);
        md["decoder.mlp.swiglu_limit"] = (float)(c.GetDouble("swiglu_limit") ?? 7.0);
    }

    /// <inheritdoc/>
    protected override bool TryMapFamilyTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        // The transformers layout fuses the gate and up projections of the dense MLP and of the shared expert (gate rows first).
        string? target = suffix switch
        {
            "mlp.gate_up_proj.weight" => "mlp.",
            "mlp.shared_experts.gate_up_proj.weight" => "mlp.shared.",
            _ => null,
        };
        if (target is null)
        {
            return false;
        }

        int rows = checked((int)w.Shape[0].Fixed) / 2;
        collector.Add(HfTransforms.Plain(w.RowSlice("", 0, rows), layerPrefix + target + "gate.weight"));
        collector.Add(HfTransforms.Plain(w.RowSlice("", rows, rows), layerPrefix + target + "up.weight"));
        return true;
    }

    // Which layers use MiniMax sparse attention: layer_types ("minimax_m3_sparse") or the published sparse_attention_freq flags.
    private static IEnumerable<bool> SparseLayers(HfConfig c)
        => c.GetStringArray("layer_types") is { } types ? types.Select(t => t == "minimax_m3_sparse")
            : SparseConfig(c) is { } sparse && sparse.TryGetProperty("sparse_attention_freq", out var freq) && freq.ValueKind == JsonValueKind.Array
                ? freq.EnumerateArray().Select(f => f.GetDouble() != 0) : [];

    private static JsonElement? SparseConfig(HfConfig c) => c.Get("sparse_attention_config") is { ValueKind: JsonValueKind.Object } sparse ? sparse : null;

    // An indexer setting: the flat transformers key or the published sparse_attention_config entry.
    private static int Sparse(HfConfig c, string flat, string published, int fallback)
        => c.GetInt(flat) ?? (SparseConfig(c) is { } sparse && sparse.TryGetProperty(published, out var value) ? value.GetInt32() : fallback);
}
