namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>cohere2_moe</c> (Command A MoE): the Cohere2 layer (attention and MLP in parallel around one norm, layer norm without bias or RMS
/// norm when <c>rms_norm_eps</c> is set, interleaved rotary pairs, sliding layers that rotate while the global layers do not, logits scaled by
/// <c>logit_scale</c>) with leading dense layers (<c>mlp_layer_types</c>, rotary when <c>prefix_dense_sliding_window_pattern</c> is 1) and experts routed by
/// a softmax over the top-k logits (or their sigmoids, optionally renormalized). A shared expert is added to the routed sum, or averaged with it
/// (<c>shared_expert_combination_strategy</c> "average": both halves scaled by 0.5).
/// </summary>
internal sealed class Cohere2MoeHfMapping : HfMoeFamilyMapping
{
    private const float AverageScale = 0.5f;

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["cohere2_moe"];

    /// <inheritdoc/>
    protected override string RopeStyle => "gptj";

    /// <inheritdoc/>
    protected override int FeedForwardSize(HfConfig c) => c.GetInt("prefix_dense_intermediate_size") ?? c.Int("intermediate_size");

    /// <inheritdoc/>
    protected override float NormEpsilon(HfConfig c) => (float)(c.GetDouble("rms_norm_eps") ?? c.GetDouble("layer_norm_eps") ?? 1e-5);

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
    {
        string selection = c.GetString("expert_selection_fn") ?? "softmax";
        if (selection is not ("softmax" or "sigmoid"))
        {
            throw new NotSupportedException($"cohere2_moe: expert_selection_fn '{selection}' is not supported.");
        }

        bool sigmoid = selection == "sigmoid";
        return new(c.Int("num_experts"), c.Int("num_experts_per_tok"), c.Int("intermediate_size"), sigmoid ? "sigmoid" : "softmax_topk",
            Normalize: sigmoid && (c.GetBool("norm_topk_prob") ?? true), Scale: Averaged(c) ? AverageScale : 1f);
    }

    /// <inheritdoc/>
    protected override void AddFamilyHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        if (c.GetDouble("rms_norm_eps") is null)
        {
            md["decoder.norm.type"] = "layer";
        }

        md["decoder.residual.parallel"] = true;
        md["decoder.logit_scale"] = (float)(c.GetDouble("logit_scale") ?? 1.0);
        if (Averaged(c))
        {
            md["decoder.moe.shared_scale"] = AverageScale;
        }

        int layers = LayerCount(c);
        var sliding = AddSlidingLayers(md, c) ?? new int[layers];
        var mlpTypes = c.GetStringArray("mlp_layer_types");
        bool denseRotates = (c.GetInt("prefix_dense_sliding_window_pattern") ?? 1) == 1;
        md["decoder.rope.layers"] = Enumerable.Range(0, layers)
            .Select(il => sliding[il] != 0 || (denseRotates && mlpTypes is not null && mlpTypes[il] == "dense") ? 1 : 0).ToArray();
    }

    /// <summary>Whether a shared expert is averaged with the routed sum instead of added to it.</summary>
    /// <exception cref="NotSupportedException">The combination strategy is unknown.</exception>
    private static bool Averaged(HfConfig c)
    {
        if ((c.GetInt("num_shared_experts") ?? 0) == 0)
        {
            return false;
        }

        return (c.GetString("shared_expert_combination_strategy") ?? "average") switch
        {
            "average" => true,
            "sum" => false,
            var other => throw new NotSupportedException($"cohere2_moe: shared_expert_combination_strategy '{other}' is not supported."),
        };
    }
}
