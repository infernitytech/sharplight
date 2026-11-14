namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>glm5-next</c> (GLM-5.3-Flash): Kimi delta-attention layers (low-rank decay and gate, the bounded gate <c>kda.gate_lower_bound</c>)
/// and DSA layers whose latent attention has no rotary dimensions, dense first blocks and MoE layers whose MLPs clamp the SwiGLU
/// (<c>swiglu_clamp_exp</c> / <c>swiglu_clamp_shexp</c>, the gate before the SiLU), and hyper-connections (<c>hyper_connection.*</c>, Sinkhorn
/// mixing) averaged before the output norm. The k-pool indexer picks <c>attention.indexer.top_k</c> keys: whole pools of
/// <c>attention.indexer.kpool</c> positions scored by their gated pooled keys (<c>indexer_compressor_{gate,ape}</c>), plus the query's incomplete
/// pool (<c>attention.indexer.kpool_select_tail</c>); layers whose <c>attention.indexer.types</c> entry is false reuse the previous selection.
/// </summary>
internal sealed partial class LatentDecoderGgufMapping
{
    private static bool IsGlm5Next(GgufMetadataView g) => g.GgufArchitecture == "glm5-next";

    private static void AddGlm5NextHyperParameters(Dictionary<string, object> md, string a, GgufMetadataView g)
    {
        int layers = TrunkLayers(g);
        if (!g.Has("expert_gating_func"))
        {
            md[a + "moe.router_activation"] = "sigmoid";   // llama.cpp's default for this architecture
        }

        md[a + "hyper_connections.streams"] = g.Int("hyper_connection.count");
        md[a + "hyper_connections.eps"] = g.Float("hyper_connection.epsilon", 1e-6f);
        md[a + "hyper_connections.sinkhorn_iterations"] = g.Int("hyper_connection.sinkhorn_iterations", 20);
        if (g.Has("swiglu_clamp_exp"))
        {
            md[a + "moe.swiglu_clamp"] = GgufMappingHelpers.PerLayerFloat(g, "swiglu_clamp_exp", layers);
            md[a + "moe.swiglu_clamp_input"] = true;
        }

        if (g.Has("swiglu_clamp_shexp") || g.Has("swiglu_clamp_exp"))
        {
            md[a + "mlp.swiglu_clamp"] = GgufMappingHelpers.PerLayerFloat(g, g.Has("swiglu_clamp_shexp") ? "swiglu_clamp_shexp" : "swiglu_clamp_exp", layers);
            md[a + "mlp.swiglu_clamp_input"] = true;
        }

        md[a + "attention.indexer.pool_size"] = g.Int("attention.indexer.kpool", 1);
        md[a + "attention.indexer.pool_tail"] = g.Bool("attention.indexer.kpool_select_tail", false);
    }

    /// <summary>The hyper-connection modules of a layer, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_hyperConnections = new(StringComparer.Ordinal)
    {
        ["hc_attn_fn.weight"] = "attention_hc.fn.weight",
        ["hc_attn_base.weight"] = "attention_hc.base",
        ["hc_attn_scale.weight"] = "attention_hc.scale",
        ["hc_ffn_fn.weight"] = "mlp_hc.fn.weight",
        ["hc_ffn_base.weight"] = "mlp_hc.base",
        ["hc_ffn_scale.weight"] = "mlp_hc.scale",
    };

    /// <summary>The k-pool tensors of the GLM-5.3-Flash indexer, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_poolIndexer = new(StringComparer.Ordinal)
    {
        ["indexer_compressor_gate.weight"] = "attention.indexer.pool_gate.weight",
        ["indexer_compressor_ape.weight"] = "attention.indexer.pool_position.weight",
    };
}
