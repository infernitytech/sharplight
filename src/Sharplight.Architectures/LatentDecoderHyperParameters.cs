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
    bool MoeSwigluClampInput, SparseAttentionIndexerOptions? Indexer, float[] MlpSwigluClamp, bool MlpSwigluClampInput, int[] RopeLayers, int[] KvSource,
    float FinalSoftCap, LatentAttentionParameters SlidingAttention, int[] SlidingLayers, float SlidingRopeBase)
{
    /// <summary>Whether layer <paramref name="il"/> is a sliding-window layer with the <see cref="SlidingAttention"/> geometry.</summary>
    public bool IsSliding(int il) => SlidingLayers.Length != 0 && SlidingLayers[il] != 0;

    /// <summary>Whether layer <paramref name="il"/> rotates its rope dimensions (false: a NoPE layer whose whole head is non-rotary).</summary>
    public bool HasRope(int il) => RopeLayers.Length == 0 || RopeLayers[il] != 0;

    /// <summary>The layer whose latent cache layer <paramref name="il"/> attends (itself unless it shares an earlier layer's).</summary>
    public int KvSourceOf(int il) => KvSource.Length == 0 ? il : KvSource[il];

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
        int slidingNope = src.Arch("attention.sliding.nope_dim", nope);
        var sliding = attention with
        {
            Heads = src.Arch("attention.sliding.head_count", attention.Heads),
            LatentRank = src.Arch("attention.sliding.latent_rank", attention.LatentRank),
            NopeDims = slidingNope,
            ValueDims = src.Arch("attention.sliding.value_dim", attention.ValueDims),
            Scale = src.Arch("attention.sliding.scale", 1f / MathF.Sqrt(slidingNope + rope)),
            Window = src.Arch("attention.sliding_window", 0),
        };
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
            Indexer: SparseAttentionIndexerOptions.Read(src),
            MlpSwigluClamp: src.Arch("mlp.swiglu_clamp", Array.Empty<float>()),
            MlpSwigluClampInput: src.Arch("mlp.swiglu_clamp_input", false),
            RopeLayers: src.Arch("rope.layers", Array.Empty<int>()),
            KvSource: src.Arch("attention.kv_source_layers", Array.Empty<int>()),
            FinalSoftCap: src.Arch("final_logit_softcap", 0f),
            SlidingAttention: sliding,
            SlidingLayers: src.Arch("attention.sliding_layers", Array.Empty<int>()),
            SlidingRopeBase: src.Arch("rope.base_sliding", src.Arch("rope.base", 10000f)));
        if (hp.TemperatureScale != 0f && hp.TemperatureLength <= 0f)
        {
            throw new InvalidDataException("latent_decoder: attention.temperature_scale needs a positive attention.temperature_length.");
        }

        if (hp.MoeSwigluClamp.Length is not 0 && hp.MoeSwigluClamp.Length != hp.LayerCount)
        {
            throw new InvalidDataException("latent_decoder: moe.swiglu_clamp must have layer_count entries.");
        }

        if (hp.MlpSwigluClamp.Length is not 0 && hp.MlpSwigluClamp.Length != hp.LayerCount)
        {
            throw new InvalidDataException("latent_decoder: mlp.swiglu_clamp must have layer_count entries.");
        }

        if (hp.SlidingLayers.Length != 0 && hp.SlidingAttention.Window <= 0)
        {
            throw new InvalidDataException("latent_decoder: attention.sliding_layers needs a positive attention.sliding_window.");
        }

        foreach (var (values, key) in new[] { (hp.RopeLayers, "rope.layers"), (hp.KvSource, "attention.kv_source_layers"), (hp.SlidingLayers, "attention.sliding_layers") })
        {
            if (values.Length is not 0 && values.Length != hp.LayerCount)
            {
                throw new InvalidDataException($"latent_decoder: {key} must have layer_count entries.");
            }
        }

        for (int il = 0; il < hp.LayerCount; il++)
        {
            int source = hp.KvSourceOf(il);
            if (source < 0 || source > il || hp.KvSourceOf(source) != source || hp.HasRope(source) != hp.HasRope(il) || hp.IsSliding(source) != hp.IsSliding(il))
            {
                throw new InvalidDataException($"latent_decoder: layer {il} cannot attend the latent cache of layer {source}.");
            }
        }

        return hp;
    }
}
