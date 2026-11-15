using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Gguf;

/// <summary>
/// llama.cpp <c>metis16</c> (Metis 1.6) → canonical <c>depth_memory</c> (documented on <c>DepthMemoryAdapter</c>). The GGUF block count counts the
/// virtual layers of all passes, while the file holds the tensors of the physical layers only; a layer with a positive key/value head count has
/// attention, the others Mamba-2. The converter already folds the stream mixing (Sinkhorn), the tanh of the pass gates and A = −exp(A_log).
/// </summary>
internal sealed class Metis16GgufMapping : IGgufArchitectureMapping
{
    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["metis16"];

    /// <inheritdoc/>
    public string Architecture => "depth_memory";

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int passes = g.Int("metis.pass_count");
        int virtualLayers = g.Int("block_count");
        if (passes < 1 || virtualLayers % passes != 0)
        {
            throw new InvalidDataException($"metis16: {virtualLayers} layers do not split into {passes} passes.");
        }

        int hidden = g.Int("embedding_length");
        int headCount = GgufDecoderHyperParameters.LargestOf(g, "attention.head_count");
        int headDim = g.Int("attention.key_length", hidden / headCount);
        int ssmHeads = g.Int("ssm.time_step_rank");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = virtualLayers / passes,
            [a + "pass_count"] = passes,
            [a + "hidden_size"] = hidden,
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
            [a + "ssm.conv_kernel"] = g.Int("ssm.conv_kernel"),
            [a + "ssm.state_size"] = g.Int("ssm.state_size"),
            [a + "ssm.group_count"] = g.Int("ssm.group_count"),
            [a + "ssm.head_count"] = ssmHeads,
            [a + "ssm.head_dim"] = g.Int("ssm.inner_size") / ssmHeads,
            [a + "ssm.norm_eps"] = g.Float("ssm.norm_epsilon", g.Float("attention.layer_norm_rms_epsilon")),
            [a + "attention.head_count"] = headCount,
            [a + "attention.kv_head_count"] = GgufMappingHelpers.PerLayer(g, "attention.head_count_kv", virtualLayers).Max(),
            [a + "attention.head_dim"] = headDim,
            [a + "attention.scale"] = 1f / MathF.Sqrt(headDim),
            [a + "moe.expert_count"] = g.Int("expert_count"),
            [a + "moe.expert_used_count"] = g.Int("expert_used_count"),
            [a + "moe.router_activation"] = "softmax_topk",
            [a + "moe.selection_bias_on_logits"] = true,
            [a + "continuation.logit_limit"] = g.Float("metis.continuation_logit_limit"),
        };
        GgufMappingHelpers.AddContextLength(md, a, g);
        return md;
    }

    /// <summary>Global tensors, GGUF name to canonical name.</summary>
    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["stream_embd.weight"] = "streams.embedding",
        ["depth_mem_state_write.weight"] = "depth_memory.write_state.weight",
        ["depth_mem_meta_write.weight"] = "depth_memory.write_meta.weight",
        ["depth_mem_entry_bias.weight"] = "depth_memory.entry_bias",
        ["depth_mem_query.weight"] = "depth_memory.query.weight",
        ["depth_mem_query.bias"] = "depth_memory.query.bias",
        ["depth_mem_key.weight"] = "depth_memory.key.weight",
        ["depth_mem_key.bias"] = "depth_memory.key.bias",
        ["depth_mem_value.weight"] = "depth_memory.value.weight",
        ["depth_mem_value.bias"] = "depth_memory.value.bias",
        ["depth_mem_out.weight"] = "depth_memory.out.weight",
        ["depth_mem_out.bias"] = "depth_memory.out.bias",
        ["depth_mem_route_state.weight"] = "depth_memory.route.state.weight",
        ["depth_mem_route_retr.weight"] = "depth_memory.route.retrieved.weight",
        ["depth_mem_route_diff.weight"] = "depth_memory.route.diff.weight",
        ["depth_mem_route_hist.weight"] = "depth_memory.route.history.weight",
        ["depth_mem_route.bias"] = "depth_memory.route.bias",
        ["depth_mem_stream_gate.weight"] = "depth_memory.stream_gate.weight",
        ["depth_mem_stream_gate.bias"] = "depth_memory.stream_gate.bias",
        ["cont_hidden_state.weight"] = "continuation.state.weight",
        ["cont_hidden_retr.weight"] = "continuation.retrieved.weight",
        ["cont_hidden_diff.weight"] = "continuation.diff.weight",
        ["cont_hidden_route.weight"] = "continuation.route.weight",
        ["cont_hidden.bias"] = "continuation.bias",
        ["cont_output.weight"] = "continuation.out.weight",
        ["cont_output.bias"] = "continuation.out.bias",
    };

    /// <summary>Per-layer tensors, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["ffn_norm.weight"] = "post_attention_norm.weight",
        ["hc_attn_mix.weight"] = "streams.attention.mix",
        ["hc_attn_read.weight"] = "streams.attention.read",
        ["hc_attn_write.weight"] = "streams.attention.write",
        ["hc_ffn_mix.weight"] = "streams.mlp.mix",
        ["hc_ffn_read.weight"] = "streams.mlp.read",
        ["hc_ffn_write.weight"] = "streams.mlp.write",
        ["ssm_in.weight"] = "ssm.in.weight",
        ["ssm_out.weight"] = "ssm.out.weight",
        ["ssm_conv1d.weight"] = "ssm.conv1d.weight",
        ["ssm_conv1d.bias"] = "ssm.conv1d.bias",
        ["ssm_dt.bias"] = "ssm.dt_bias",
        ["ssm_a"] = "ssm.a",
        ["ssm_d"] = "ssm.d",
        ["ssm_norm.weight"] = "ssm.norm.weight",
        ["attn_qkv.weight"] = "attention.qkv.weight",
        ["attn_output.weight"] = "attention.o.weight",
        ["attn_pass_lora_down.weight"] = "attention.pass_lora.down.weight",
        ["attn_pass_lora_up.weight"] = "attention.pass_lora.up.weight",
        ["attn_pass_gate.weight"] = "attention.pass_gate",
        ["ffn_latent_down.weight"] = "latent.down.weight",
        ["ffn_latent_up.weight"] = "latent.up.weight",
        ["ffn_gate_inp.weight"] = "mlp.router.weight",
        ["ffn_gate_inp.bias"] = "mlp.router.bias",
        ["exp_probs_b.bias"] = "mlp.router_bias",
        ["ffn_exp_embd.weight"] = "mlp.expert_embedding",
        ["ffn_gate_up_exps.weight"] = "mlp.experts.gate_up.weight",
        ["ffn_down_exps.weight"] = "mlp.experts.down.weight",
        ["ffn_gate_shexp.weight"] = "mlp.shared.gate.weight",
        ["ffn_up_shexp.weight"] = "mlp.shared.up.weight",
        ["ffn_down_shexp.weight"] = "mlp.shared.down.weight",
    };

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (GgufTensorNames.MapGlobal(name) is { } globalName)
        {
            return globalName;
        }

        if (s_global.TryGetValue(name, out var canonical))
        {
            return canonical;
        }

        var m = GgufTensorNames.Block().Match(name);
        return m.Success && s_layer.TryGetValue(m.Groups[2].Value, out var c) ? $"layers.{m.Groups[1].Value}.{c}" : null;
    }

    /// <summary>
    /// The stream embedding is stored [H, S] (stream-innermost) and becomes [S, H]; Mamba-2 A, D ([H, 1]) and the gated norm ([G, inner/G]) become flat.
    /// </summary>
    public WeightInfo MapWeight(WeightInfo w, GgufMetadataView g)
    {
        if (w.Name == "streams.embedding")
        {
            long hidden = w.Shape[0].Fixed;
            long streams = w.Shape[1].Fixed;
            return w.As(w.Name, new Shape(1, hidden * streams)).TransposeRowBlocks(w.Name, hidden, streams).As(w.Name, new Shape(streams, hidden));
        }

        if (w.Name.EndsWith(".ssm.a", StringComparison.Ordinal) || w.Name.EndsWith(".ssm.d", StringComparison.Ordinal)
            || w.Name.EndsWith(".ssm.norm.weight", StringComparison.Ordinal))
        {
            return w.As(w.Name, new Shape(w.Shape.Elements()));
        }

        return w;
    }
}
