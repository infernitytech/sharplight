using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Hyper-parameters of the canonical <c>latent_decoder</c> architecture, read from the <c>latent_decoder.*</c> metadata keys
/// (vocabulary on <see cref="LatentDecoderAdapter"/>).
/// </summary>
internal sealed record LatentDecoderHyperParameters(
    int LayerCount, int HiddenSize, int Vocab, int ContextLength, float Eps, LatentAttentionParameters Attention,
    float RopeBase, string RopeStyle, string Activation, bool Gated, float EmbeddingScale, float ResidualScale, float LogitScale,
    MoeOptions Moe, int ExpertGroups, int ExpertGroupsUsed, string GroupScore)
{
    /// <summary>Reads and validates the hyper-parameters of a latent decoder source.</summary>
    /// <exception cref="InvalidDataException">A required key is missing or the expert grouping is inconsistent.</exception>
    public static LatentDecoderHyperParameters Read(ModelSource src)
    {
        int nope = src.Arch<int>("attention.nope_dim");
        int rope = src.Arch<int>("attention.rope_dim");
        var attention = new LatentAttentionParameters(
            Heads: src.Arch<int>("attention.head_count"),
            QueryRank: src.Arch("attention.q_rank", 0),
            LatentRank: src.Arch<int>("attention.latent_rank"),
            NopeDims: nope,
            RopeDims: rope,
            ValueDims: src.Arch<int>("attention.value_dim"),
            Scale: src.Arch("attention.scale", 1f / MathF.Sqrt(nope + rope)),
            RopeMagnitude: src.Arch("rope.attention_scale", 1f),
            Eps: src.Arch("norm_eps", 1e-6f));
        var embedding = src.Weight("token_embedding.weight") ?? throw new InvalidDataException("latent_decoder: token_embedding.weight is missing.");
        var hp = new LatentDecoderHyperParameters(
            LayerCount: src.Arch<int>("layer_count"),
            HiddenSize: src.Arch<int>("hidden_size"),
            Vocab: embedding.Shape[0].IsSymbolic ? 0 : (int)embedding.Shape[0].Fixed,
            ContextLength: src.Arch("context_length", 4096),
            Eps: attention.Eps,
            Attention: attention,
            RopeBase: src.Arch("rope.base", 10000f),
            RopeStyle: src.Arch("rope.style", "gptj"),
            Activation: src.Arch("mlp.activation", "silu"),
            Gated: src.Arch("mlp.gated", true),
            EmbeddingScale: src.Arch("embedding_scale", 1f),
            ResidualScale: src.Arch("residual_scale", 1f),
            LogitScale: src.Arch("logit_scale", 1f),
            Moe: MoeOptions.Read(src),
            ExpertGroups: src.Arch("moe.group_count", 1),
            ExpertGroupsUsed: src.Arch("moe.group_used_count", src.Arch("moe.group_count", 1)),
            GroupScore: src.Arch("moe.group_score", "top2_sum"));

        if (hp.ExpertGroups > 1 && (hp.Moe.Experts % hp.ExpertGroups != 0 || hp.ExpertGroupsUsed < 1 || hp.ExpertGroupsUsed > hp.ExpertGroups))
        {
            throw new InvalidDataException("latent_decoder: moe.group_count must divide moe.expert_count and moe.group_used_count must be in [1, group_count].");
        }

        if (hp.GroupScore is not ("max" or "top2_sum"))
        {
            throw new InvalidDataException($"latent_decoder: unknown moe.group_score '{hp.GroupScore}'.");
        }

        return hp;
    }

    /// <summary>Whether routing is limited to the best expert groups (DeepSeek-V2/V3 group-limited top-k).</summary>
    public bool GroupLimited => ExpertGroups > 1 && ExpertGroupsUsed < ExpertGroups;
}
