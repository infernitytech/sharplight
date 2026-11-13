using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Generic decoder-only transformer (canonical arch <c>decoder</c>): Llama, Mistral, SmolLM, Qwen2/Qwen3, Gemma 2/3,
/// Phi-3 and friends are all this one graph with different hyper-parameters. Importers translate their format into
/// exactly the vocabulary below.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>decoder.*</c>):
/// <list type="bullet">
/// <item><c>layer_count</c>, <c>hidden_size</c>, <c>ffn_hidden_size</c>, <c>context_length</c>, <c>norm_eps</c> (RMS norm epsilon).</item>
/// <item><c>attention.head_count</c>, <c>attention.kv_head_count</c> (GQA), <c>attention.head_dim</c>;
/// optional <c>attention.scale</c> (default 1/sqrt(head_dim)); <c>attention.logit_softcap</c> (tanh soft-capping of scores, 0 = off);
/// <c>attention.sliding_window</c> (tokens) with <c>attention.sliding_layers</c> (int[layer_count], 1 = layer uses the window).</item>
/// <item><c>rope.dim</c> (rotated dims, partial rotary when less than head_dim), <c>rope.base</c>, <c>rope.style</c>
/// ("neox" = rotate-half pairs (i, i+dim/2), the default; "gptj" = interleaved pairs), <c>rope.local_base</c> (base of sliding layers,
/// default rope.base), <c>rope.attention_scale</c> (cos/sin magnitude factor, e.g. yarn; folded into the score scale as its square).
/// Frequency scaling (llama3 / yarn / linear) is expressed as the optional weight <c>rope.freq_factors</c> (f32 [rope.dim/2], θ is divided by it)
/// which applies to the layers that are not sliding. Longrope (Phi-3) additionally carries <c>rope.freq_factors_long</c> and
/// <c>rope.original_context_length</c>: the long factors are used instead of the short ones when the configured context
/// (<see cref="AdapterOptions.ContextLength"/>, the model's <c>context_length</c> when unset) exceeds the original one. This is llama.cpp's
/// rule; HF switches on the current sequence length, which would leave already cached keys rotated with the other set, so a
/// KV-cached runtime picks once per model instance. Use a context &lt;= the original one for the short factors.
/// When <c>attention.sliding_window</c> is given without <c>attention.sliding_layers</c> (Mistral v0.1), every layer uses the window.</item>
/// <item><c>mlp.activation</c> ("silu" (default), "gelu" (erf), "gelu_tanh", "relu", "relu2" (squared ReLU), "xielu", "swiglu_oai"), <c>mlp.gated</c> (default true: act(gate)·up).
/// xIELU reads <c>mlp.xielu_beta</c> / <c>mlp.xielu_eps</c> and the per-layer scalar weights <c>layers.{i}.mlp.act.alpha_p</c> / <c>alpha_n</c> (effective values, already softplus-transformed);
/// SwiGLU-OAI (clamped, gpt-oss) reads <c>mlp.swiglu_alpha</c> (1.702) and <c>mlp.swiglu_limit</c> (7).</item>
/// <item>Building blocks of the generic layer: <c>norm.type</c> ("rms" (default) or "layer": mean-centred layer norm that also adds the sibling <c>*.bias</c> weights when present
/// and is non-parametric when the weight is absent, OLMo 1); <c>pre_norm</c> (default true; false = no norm in front of the sub-layers, OLMo 2 / EXAONE 4 normalize their outputs instead);
/// <c>residual.parallel</c> (default false: attention and MLP both read the layer input, x + attn(norm(x)) + mlp(norm'(x)); the MLP uses <c>post_attention_norm</c> when the layer has one and shares the attention norm otherwise);
/// <c>residual.scale</c> (factor on every sub-layer output before the residual add, muP); <c>attention.qk_norm_after_rope</c> (q/k norms after the rotary embedding, Hunyuan);
/// <c>attention.clip_qkv</c> (clamp of the q/k/v projections, OLMo 1); <c>attention.layer_head_counts</c> (per-layer query heads, OpenELM).
/// Optional weights: <c>position_embedding.weight</c> [positions, H] (learned absolute positions added to the token embeddings, GPT-2), <c>lm_head.bias</c>, <c>*_norm.bias</c> (layer norm).</item>
/// <item>Mixture of experts (a layer is MoE when it has <c>mlp.router.weight</c>; routing and expert weights are those of <see cref="MoeBlock"/>):
/// <c>moe.expert_count</c>, <c>moe.expert_used_count</c> (top-k per token), <c>moe.ffn_hidden_size</c> (the experts' inner size;
/// <c>ffn_hidden_size</c> then only describes dense layers) and the routing keys of <see cref="MoeOptions"/>: <c>moe.router_activation</c>
/// ("softmax" (default), "sigmoid", "softmax_topk", "sqrt_softplus"), <c>moe.normalize_weights</c> (default false), <c>moe.weights_scale</c>,
/// <c>moe.group_count</c> / <c>moe.group_used_count</c> / <c>moe.group_score</c> (group-limited top-k), <c>moe.weights_before_experts</c>;
/// optional selection bias <c>mlp.router_bias</c>, logit bias <c>mlp.router.bias</c>, per-expert biases <c>mlp.experts.*.bias</c> and a
/// shared expert <c>mlp.shared.*</c>. <c>moe.router_input</c> ("normed" (default) or "layer_input": the router reads the layer input
/// before the attention norm, SmallThinker).</item>
/// <item>Attention temperature (Llama 4, Mistral 3): <c>attention.temperature_scale</c> (β, 0 = off), <c>attention.temperature_length</c> (L),
/// <c>attention.temperature_offset</c> (o, default 0) and optional <c>attention.temperature_layers</c> (int[layer_count], default all): queries of
/// those layers are multiplied by log(floor((pos + o) / L) + 1)·β + 1. <c>attention.qk_norm_layers</c> (int[layer_count]) limits the
/// unweighted q/k norm to the flagged layers. Chunked attention <c>attention.chunk_size</c> with <c>attention.chunk_layers</c> (Llama 4)
/// limits each query to the keys of its own chunk. <c>rope.local_freq_factors</c> (bool): sliding layers use <c>rope.freq_factors</c> too (gpt-oss).</item>
/// <item><c>embedding_scale</c> (token embeddings are multiplied by it, default 1; Gemma sqrt(hidden_size)),
/// <c>embedding_norm</c> (bool: a weight-less RMS norm is applied to the (scaled) token embeddings),
/// <c>logit_scale</c> (logits are multiplied by it, default 1, before the soft cap),
/// <c>final_logit_softcap</c> (logits = cap·tanh(logits/cap), 0 = off).</item>
/// <item>Per-layer variations (all optional int[] with <c>layer_count</c> entries, defaults from the scalar keys):
/// <c>attention.layer_head_dims</c>, <c>attention.layer_kv_head_counts</c>, <c>attention.kv_source_layers</c> (layer whose K/V cache this layer
/// reads; its own index when it owns one. A shared layer has no k/v projections and no cache of its own — Gemma 4 "KV sharing"),
/// <c>rope.layers</c> (1 = rotary on, 0 = NoPE layer; default all 1), <c>rope.local_dim</c> (rotated dims of sliding layers, default rope.dim).
/// <c>attention.qk_norm_unweighted</c> (bool: q/k get a weight-less RMS norm when the layer has no q/k norm weight),
/// <c>attention.v_norm</c> (bool: v gets a weight-less RMS norm), <c>post_norm_eps</c> (epsilon of attention.out_norm / mlp.out_norm, default norm_eps).</item>
/// <item>Per-layer embeddings (Gemma 3n/4 "PLE"; present when <c>per_layer.size</c> = P &gt; 0): weights <c>per_layer.embedding.weight</c> [V, L·P],
/// <c>per_layer.model_projection.weight</c> [L·P, H], <c>per_layer.projection_norm.weight</c> [P]; per layer <c>layers.{i}.per_layer.{gate,projection}.weight</c>
/// ([P, H] / [H, P]) and <c>layers.{i}.per_layer.norm.weight</c>. Per-layer input = (norm(proj(e)·H^-½) + emb[token]·√P)·2^-½ with e the scaled token embedding;
/// each layer ends with x += norm(projection(act(gate(x)) · input_i)) (before <c>output_scale</c>).</item>
/// </list>
/// Weights: <c>token_embedding.weight</c> [V, H], <c>output_norm.weight</c>, <c>lm_head.weight</c> (optional, tied otherwise);
/// per layer <c>layers.{i}.</c>: <c>input_norm.weight</c> (in front of attention), <c>attention.{q,k,v,o}.weight</c> with optional
/// <c>.bias</c>, optional <c>attention.{q,k}_norm.weight</c> (per-head RMS norm before rope), optional <c>attention.out_norm.weight</c>
/// (norm of the attention output before the residual add), <c>post_attention_norm.weight</c> (in front of the MLP),
/// <c>mlp.{gate,up,down}.weight</c> (+ optional <c>.bias</c>; no gate when not gated), optional <c>mlp.out_norm.weight</c>,
/// optional <c>attention.sinks</c> (f32 [head_count]: a learned per-head logit that joins the softmax denominator without a value),
/// optional <c>attention.gate.weight</c> (sigmoid output gate of the attention: a separate projection of the layer input),
/// optional <c>output_scale</c> (f32 [1]: the layer's output is multiplied by it), optional BitNet sub-norms <c>attention.sub_norm.weight</c> /
/// <c>mlp.sub_norm.weight</c> (RMS norms of the inputs of the attention output and MLP down projections). A layer without <c>attention.v.weight</c> uses the
/// key projection (before its norm) as value (Gemma 4 attention_k_eq_v).
/// Semantics: RMS norm weights multiply directly (any "1 + w" convention is folded in by the importer); q/k rows are in
/// rotate-half (neox) order per head, whatever <c>rope.style</c> says about the pairing the rope op uses on them is the
/// importer's concern (a llama.cpp "normal"-rope checkpoint is un-permuted on import so canonical weights are format independent).
/// </remarks>
public sealed class DecoderAdapter : IModelAdapter
{
    /// <inheritdoc/>
    public string Name => "Decoder-only transformer";
    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => ["decoder"];

    /// <inheritdoc/>
    /// <exception cref="InvalidDataException">The hyper-parameters are inconsistent (see <see cref="DecoderHyperParameters.Read"/>).</exception>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var hp = DecoderHyperParameters.Read(src);
        var graphBuilder = new DecoderGraphBuilder(src, options, hp);
        var graphs = graphBuilder.BuildGraphs();

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
            Graphs = graphs,
            States = graphBuilder.States,
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Properties = new Dictionary<string, object>
            {
                [WellKnown.VocabSize] = hp.Vocab,
                [WellKnown.ContextLength] = hp.ContextLength,
                [WellKnown.EmbeddingLength] = hp.NEmbd,
                [WellKnown.PositionAxes] = 1,
                [WellKnown.EosTokenIds] = eos.ToArray(),
                ["n_layer"] = hp.NLayer,
                [WellKnown.EmbeddingPooling] = graphBuilder.Pooling,
                [WellKnown.EmbeddingNormalize] = graphBuilder.Normalize,
                [WellKnown.CausalAttention] = true,
                [WellKnown.ClassifierLabels] = hp.Vocab,
            },
        };
    }
}
