namespace Sharplight.Import.Gguf;

/// <summary>
/// llama.cpp <c>kimi-k3</c> and <c>kimi-k31</c> (Kimi K3 / K3.1): Kimi-Linear layers with block attention residuals (<c>attn_res.block_size</c>,
/// the score vectors <c>attn_res_score</c> / <c>ffn_res_score</c> / <c>output_res_score</c>), the situ activation
/// (<c>activation.situ_beta</c> / <c>activation.situ_linear_beta</c>), latent MoE (<c>ffn_routed_{down,norm,up}</c>), full-rank delta-attention
/// gates (<c>ssm_g</c>) with the bounded decay (<c>kda.gate_lower_bound</c>, one rate per head as the converter stores <c>ssm_a</c>) and a
/// sigmoid output gate on the latent attention (<c>attn_gate</c>). K3.1 adds a DSA indexer without rotary embedding to the latent-attention
/// layers (<c>attention.indexer.*</c>, key LayerNorm epsilon <c>attention.layer_norm_epsilon</c>).
/// </summary>
internal sealed partial class LatentDecoderGgufMapping
{
    private static bool IsKimiK3(GgufMetadataView g) => g.GgufArchitecture is "kimi-k3" or "kimi-k31";

    private static void AddKimiK3HyperParameters(Dictionary<string, object> md, string a, GgufMetadataView g)
    {
        md[a + "attention_residuals.block_size"] = g.Int("attn_res.block_size", 0);
        md[a + "mlp.activation"] = "situ";
        md[a + "mlp.situ_beta"] = g.Float("activation.situ_beta", 1f);
        md[a + "mlp.situ_linear_beta"] = g.Float("activation.situ_linear_beta", 0f);
        if (g.Has("kda.gate_lower_bound"))
        {
            md[a + "linear_attention.decay_bound"] = g.Float("kda.gate_lower_bound");
        }

        if (g.Has("attention.indexer.top_k"))
        {
            md[a + "attention.indexer.rope_style"] = "none";
            md[a + "attention.indexer.norm_eps"] = g.Float("attention.layer_norm_epsilon", 1e-6f);
        }
    }

    /// <summary>The per-layer tensors K3 adds, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_kimiK3 = new(StringComparer.Ordinal)
    {
        ["attn_res_score.weight"] = "attention_res.score.weight",
        ["ffn_res_score.weight"] = "mlp_res.score.weight",
        ["ffn_routed_down.weight"] = "mlp.latent_down.weight",
        ["ffn_routed_norm.weight"] = "mlp.latent_out_norm.weight",
        ["ffn_routed_up.weight"] = "mlp.latent_up.weight",
        ["attn_gate.weight"] = "attention.gate.weight",
    };
}
