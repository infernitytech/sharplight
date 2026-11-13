namespace Sharplight.HuggingFace;

/// <summary>
/// The routing of a Hugging Face MoE checkpoint in canonical terms (the <c>moe.*</c> keys documented on <c>MoeOptions</c>).
/// </summary>
/// <param name="Experts">Routed experts (HF <c>num_experts</c> / <c>n_routed_experts</c> / <c>num_local_experts</c>).</param>
/// <param name="ExpertsUsed">Experts per token (HF <c>num_experts_per_tok</c>).</param>
/// <param name="ExpertFeedForward">Inner size of one expert (HF <c>moe_intermediate_size</c>).</param>
/// <param name="Activation">Score function: "softmax", "sigmoid", "softmax_topk" or "sqrt_softplus".</param>
/// <param name="Normalize">Renormalize the selected weights (HF <c>norm_topk_prob</c>).</param>
/// <param name="Scale">Routed scaling factor (HF <c>routed_scaling_factor</c>).</param>
/// <param name="Groups">Expert groups (HF <c>n_group</c>).</param>
/// <param name="GroupsUsed">Groups kept per token (HF <c>topk_group</c>).</param>
/// <param name="WeightsBeforeExperts">Scale each expert's input by its routing weight instead of its output (Llama 4).</param>
internal sealed record MoeRouting(int Experts, int ExpertsUsed, int ExpertFeedForward, string Activation, bool Normalize, float Scale = 1f,
    int Groups = 1, int GroupsUsed = 1, bool WeightsBeforeExperts = false)
{
    /// <summary>Writes the canonical keys below <paramref name="prefix"/> (e.g. "decoder.").</summary>
    public void AddTo(Dictionary<string, object> md, string prefix)
    {
        md[prefix + "moe.expert_count"] = Experts;
        md[prefix + "moe.expert_used_count"] = ExpertsUsed;
        md[prefix + "moe.ffn_hidden_size"] = ExpertFeedForward;
        md[prefix + "moe.router_activation"] = Activation;
        md[prefix + "moe.normalize_weights"] = Normalize;
        if (Scale != 1f)
        {
            md[prefix + "moe.weights_scale"] = Scale;
        }

        if (WeightsBeforeExperts)
        {
            md[prefix + "moe.weights_before_experts"] = true;
        }

        if (Groups > 1)
        {
            md[prefix + "moe.group_count"] = Groups;
            md[prefix + "moe.group_used_count"] = GroupsUsed;
        }
    }
}
