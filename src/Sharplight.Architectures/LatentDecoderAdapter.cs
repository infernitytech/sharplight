using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Decoder-only transformer with multi-head latent attention (canonical arch <c>latent_decoder</c>): DeepSeek-V2/V2-Lite/V3/R1,
/// Kimi-K2, GLM-4.7-Flash, Mistral 4, MiniCPM3, PLM and Hunyuan v4 are this graph with different hyper-parameters. Attention is the
/// <see cref="LatentAttention"/> block (latent KV cache, absorbed queries); the feed-forward part is a gated MLP or a routed
/// mixture of experts with optional shared experts.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>latent_decoder.*</c>):
/// <list type="bullet">
/// <item><c>layer_count</c>, <c>hidden_size</c>, <c>context_length</c>, <c>norm_eps</c> (RMS norm epsilon, also of the compressed query and latent).</item>
/// <item>Attention: <c>attention.head_count</c>, <c>attention.q_rank</c> (query compression rank, 0 = direct projection), <c>attention.latent_rank</c>
/// (compressed key/value rank), <c>attention.nope_dim</c> / <c>attention.rope_dim</c> (non-rotary / rotary query-key dimensions per head; 0 rotary
/// dimensions cache the latent alone),
/// <c>attention.value_dim</c>, <c>attention.scale</c> (softmax scale, default 1/sqrt(nope + rope); importers fold yarn's mscale² in).</item>
/// <item>Rotary embedding of the rope parts: <c>rope.base</c>, <c>rope.style</c> ("gptj" interleaved pairs, the default, as DeepSeek; "neox"
/// rotate-half), <c>rope.attention_scale</c> (factor on the rotated parts, yarn cos/sin magnitude, default 1); frequency scaling arrives as
/// the weight <c>rope.freq_factors</c> (f32 [rope_dim/2], θ is divided by it), longrope additionally as <c>rope.freq_factors_long</c> with
/// <c>rope.original_context_length</c> (the long factors apply when the configured context exceeds it).</item>
/// <item>MLP: <c>mlp.activation</c> ("silu" default, "relu2", "gelu"), <c>mlp.gated</c> (default true: act(gate)·up).</item>
/// <item>Mixture of experts (a layer is MoE when it has <c>mlp.router.weight</c>; the first layers are usually dense):
/// <c>moe.expert_count</c>, <c>moe.expert_used_count</c>, <c>moe.router_activation</c> ("softmax" default or "sigmoid"),
/// <c>moe.normalize_weights</c> (renormalize the selected weights to sum 1, default false), <c>moe.weights_scale</c> (default 1),
/// <c>moe.group_count</c> / <c>moe.group_used_count</c> (group-limited routing: experts form group_count equal groups, only the
/// group_used_count best groups compete for the top-k; default no limit), <c>moe.group_score</c> ("top2_sum" (default, DeepSeek-V3:
/// sum of the two best selection scores of a group) or "max" (DeepSeek-V2)).</item>
/// <item>Query temperature (Mistral 4, Mistral Large 3): <c>attention.temperature_scale</c> (β, 0 = off) and <c>attention.temperature_length</c> (L) scale
/// every query row by <c>1 + β · ln(1 + ⌊pos / L⌋)</c>.</item>
/// <item>Mixture of experts with clamped SwiGLU experts: <c>moe.swiglu_clamp</c> (float[layer_count], 0 = plain) with <c>moe.swiglu_clamp_input</c>
/// (see <see cref="ClampedSwigluActivation"/>); dense layers and shared experts keep the plain activation unless <c>mlp.swiglu_clamp</c>
/// (float[layer_count], 0 = plain; with <c>mlp.swiglu_clamp_input</c> on the gate before the SiLU) clamps them too (experts without their own limit
/// follow it).</item>
/// <item>Sparse-attention indexer (DeepSeek-V3.2, GLM-5, Hunyuan v4, GLM-5.3 with pooled keys): <c>attention.indexer.*</c>
/// (<see cref="SparseAttentionIndexerOptions"/>) and per layer the weights <c>attention.indexer.{q_up,k,head_weights}.weight</c>,
/// <c>attention.indexer.k_norm.{weight,bias}</c> (pooled keys also <c>attention.indexer.{pool_gate,pool_position}.weight</c>; see
/// <see cref="SparseAttentionIndexer"/>); a layer without them reuses the previous indexed layer's selection. When the context does not exceed the
/// budget the indexer is left out (every row would keep every visible position).</item>
/// <item>Hyper-connections (Hunyuan v4, DeepSeek-V4 mHC): <c>hyper_connections.*</c> (<see cref="StreamHyperConnectionOptions"/>); per layer the
/// modules <c>layers.{i}.attention_hc.*</c> and <c>layers.{i}.mlp_hc.*</c> around the two sub-layers and the head <c>output_hc.*</c> before the
/// output norm (weights on <see cref="StreamHyperConnections"/>; without <c>output_hc.*</c> the streams are averaged). No draft head with
/// hyper-connections.</item>
/// <item><c>embedding_scale</c> (factor on the token embeddings), <c>residual_scale</c> (factor on each sub-layer output before its residual
/// add; MiniCPM's scale_depth/sqrt(layers)), <c>logit_scale</c> (factor on the final hidden state before the output head); all default 1.</item>
/// </list>
/// Weights: <c>token_embedding.weight</c> [V, H], <c>output_norm.weight</c>, <c>lm_head.weight</c> (optional, tied otherwise); per layer
/// <c>layers.{i}.</c>: <c>input_norm.weight</c>, the attention weights of <see cref="LatentAttention"/>, <c>post_attention_norm.weight</c>,
/// then either the dense <c>mlp.{gate,up,down}.weight</c> (no gate when not gated) or <c>mlp.router.weight</c> [E, H], optional
/// <c>mlp.router_bias</c> (f32 [E], added to the router scores for the expert choice only; the weights stay the unbiased scores),
/// rank-3 <c>mlp.experts.{gate,up,down}.weight</c> [E, N, K] and optional shared experts <c>mlp.shared.{gate,up,down}.weight</c>
/// (one MLP every token passes through, added to the routed sum).
/// </remarks>
public sealed class LatentDecoderAdapter : IModelAdapter
{
    /// <inheritdoc/>
    public string Name => "Decoder-only transformer with multi-head latent attention";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => ["latent_decoder"];

    /// <inheritdoc/>
    /// <exception cref="InvalidDataException">The hyper-parameters are inconsistent (see <c>LatentDecoderHyperParameters.Read</c>).</exception>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var hp = LatentDecoderHyperParameters.Read(src);
        var graphBuilder = new LatentDecoderGraphBuilder(src, options, hp);

        var eos = new List<int>();
        if (src.Has(CanonicalKeys.TokenizerEosId))
        {
            eos.Add(src.Get<int>(CanonicalKeys.TokenizerEosId));
        }

        if (src.Has(CanonicalKeys.TokenizerEotId))
        {
            eos.Add(src.Get<int>(CanonicalKeys.TokenizerEotId));
        }

        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = graphBuilder.BuildGraphs(),
            States = graphBuilder.States,
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Properties = new Dictionary<string, object>(MtpDraftHead.NextnProperties(graphBuilder.HasDraftHead))
            {
                [WellKnown.VocabSize] = hp.Vocab,
                [WellKnown.ContextLength] = hp.ContextLength,
                [WellKnown.EmbeddingLength] = hp.HiddenSize,
                [WellKnown.PositionAxes] = 1,
                [WellKnown.EosTokenIds] = eos.ToArray(),
                ["n_layer"] = hp.LayerCount,
                [WellKnown.CausalAttention] = true,
            },
        };
    }
}
