using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Hyper-parameters of the canonical <c>latent_decoder</c> architecture, read from the <c>latent_decoder.*</c> metadata keys
/// (vocabulary on <see cref="LatentDecoderAdapter"/>).
/// </summary>
internal sealed record LatentDecoderHyperParameters(
    int LayerCount, int HiddenSize, int Vocab, int ContextLength, float Eps, LatentAttentionParameters Attention,
    float RopeBase, string RopeStyle, string Activation, bool Gated, float EmbeddingScale, float ResidualScale, float LogitScale,
    MoeOptions Moe, float TemperatureScale, float TemperatureLength, StreamHyperConnectionOptions? HyperConnections, float[] MoeSwigluClamp,
    bool MoeSwigluClampInput, SparseAttentionIndexerOptions? Indexer)
{
    /// <summary>Reads and validates the hyper-parameters of a latent decoder source.</summary>
    /// <exception cref="InvalidDataException">A required key is missing or a routing key is unknown.</exception>
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
            TemperatureScale: src.Arch("attention.temperature_scale", 0f),
            TemperatureLength: src.Arch("attention.temperature_length", 0f),
            HyperConnections: StreamHyperConnectionOptions.Read(src, attention.Eps),
            MoeSwigluClamp: src.Arch("moe.swiglu_clamp", Array.Empty<float>()),
            MoeSwigluClampInput: src.Arch("moe.swiglu_clamp_input", false),
            Indexer: SparseAttentionIndexerOptions.Read(src));
        if (hp.TemperatureScale != 0f && hp.TemperatureLength <= 0f)
        {
            throw new InvalidDataException("latent_decoder: attention.temperature_scale needs a positive attention.temperature_length.");
        }

        if (hp.MoeSwigluClamp.Length is not 0 && hp.MoeSwigluClamp.Length != hp.LayerCount)
        {
            throw new InvalidDataException("latent_decoder: moe.swiglu_clamp must have layer_count entries.");
        }

        return hp;
    }
}
