namespace Sharplight.Architectures;

/// <summary>How a mixture-of-experts router turns its logits into expert scores (canonical <c>moe.router_activation</c>).</summary>
internal enum MoeScoreFunction
{
    /// <summary>"softmax": softmax over all experts, then top-k (Mixtral, Qwen MoE, OLMoE, ERNIE 4.5).</summary>
    Softmax,

    /// <summary>"sigmoid": independent sigmoid per expert (DeepSeek-V3 style aux-loss-free routing, GLM-4.5, Flint, Llama 4).</summary>
    Sigmoid,

    /// <summary>
    /// "softmax_topk": top-k on the raw logits, then softmax over the k selected logits only (gpt-oss; llama.cpp
    /// <c>SOFTMAX_WEIGHT</c>). The selected weights already sum to 1.
    /// </summary>
    SoftmaxSelected,

    /// <summary>"sqrt_softplus": sqrt(softplus(logit)) per expert (DeepSeek-V4).</summary>
    SqrtSoftplus,

    /// <summary>
    /// "sparsemixer": Phi-3.5-MoE's inference-time sparse mixer. The experts are chosen by their logits (top-k); the weight of the
    /// expert in slot i is the softmax of its logit m_i over the experts not chosen in an earlier slot whose logit s lies within the
    /// jitter band (m_i − s) ≤ 2·ε·max(|s|, m_i) (<c>moe.sparsemixer_jitter</c> = ε).
    /// </summary>
    SparseMixer,
}
