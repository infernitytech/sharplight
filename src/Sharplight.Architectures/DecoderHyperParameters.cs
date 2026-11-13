using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Hyper-parameters of the canonical <c>decoder</c> architecture, read from the <c>decoder.*</c> metadata keys of a
/// <see cref="ModelSource"/> (see <see cref="DecoderAdapter"/> for the vocabulary). Per-layer variations are exposed through
/// the accessor methods so the graph code never indexes the optional per-layer arrays directly.
/// </summary>
internal sealed record DecoderHyperParameters(
    int NLayer, int NEmbd, int NFf, int NHead, int NHeadKv, int HeadDim, float Eps, float AttnScale, float SoftCap,
    int RopeDims, float RopeBase, float RopeLocalBase, string RopeStyle, int Window, int[] SlidingLayers,
    string Activation, bool Gated, float EmbeddingScale, float FinalSoftCap, int Vocab, int ContextLength,
    int Experts, int ExpertsUsed, bool NormalizeExpertWeights, bool SigmoidRouter, float ExpertWeightsScale,
    int[] LayerHeadDims, int[] LayerKvHeads, int[] KvSource, int[] RopeLayers, int RopeLocalDims, bool QkNormUnweighted, bool VNorm,
    float PostNormEps, bool EmbeddingNorm, float LogitScale, int PerLayerSize,
    string NormType, bool PreNorm, bool ParallelResidual, float ResidualScale, float SwigluAlpha, float SwigluLimit,
    float XieluBeta, float XieluEps, float ClipQkv, bool QkNormAfterRope, int[] LayerHeadCounts)
{
    /// <summary>Reads and validates the hyper-parameters of a decoder source.</summary>
    /// <exception cref="InvalidDataException">A per-layer array has the wrong length or the head counts are inconsistent.</exception>
    public static DecoderHyperParameters Read(ModelSource src)
    {
        int headDim = src.Arch<int>("attention.head_dim");
        float ropeAttn = src.Arch("rope.attention_scale", 1f);
        float ropeBase = src.Arch("rope.base", 10000f);
        var hp = new DecoderHyperParameters(
            NLayer: src.Arch<int>("layer_count"),
            NEmbd: src.Arch<int>("hidden_size"),
            NFf: src.Arch<int>("ffn_hidden_size"),
            NHead: src.Arch<int>("attention.head_count"),
            NHeadKv: src.Arch("attention.kv_head_count", src.Arch<int>("attention.head_count")),
            HeadDim: headDim,
            Eps: src.Arch("norm_eps", 1e-5f),
            AttnScale: src.Arch("attention.scale", 1f / MathF.Sqrt(headDim)) * ropeAttn * ropeAttn,
            SoftCap: src.Arch("attention.logit_softcap", 0f),
            RopeDims: src.Arch("rope.dim", headDim),
            RopeBase: ropeBase,
            RopeLocalBase: src.Arch("rope.local_base", ropeBase),
            RopeStyle: src.Arch("rope.style", "neox"),
            Window: src.Arch("attention.sliding_window", 0),
            SlidingLayers: src.Arch("attention.sliding_layers", Array.Empty<int>()),
            Activation: src.Arch("mlp.activation", "silu"),
            Gated: src.Arch("mlp.gated", true),
            EmbeddingScale: src.Arch("embedding_scale", 1f),
            FinalSoftCap: src.Arch("final_logit_softcap", 0f),
            Vocab: src.Weight("token_embedding.weight")!.Shape[0].IsSymbolic ? 0 : (int)src.Weight("token_embedding.weight")!.Shape[0].Fixed,
            ContextLength: src.Arch("context_length", 32768),
            Experts: src.Arch("moe.expert_count", 0),
            ExpertsUsed: src.Arch("moe.expert_used_count", 0),
            NormalizeExpertWeights: src.Arch("moe.normalize_weights", false),
            SigmoidRouter: src.Arch("moe.router_activation", "softmax") == "sigmoid",
            ExpertWeightsScale: src.Arch("moe.weights_scale", 1f),
            LayerHeadDims: src.Arch("attention.layer_head_dims", Array.Empty<int>()),
            LayerKvHeads: src.Arch("attention.layer_kv_head_counts", Array.Empty<int>()),
            KvSource: src.Arch("attention.kv_source_layers", Array.Empty<int>()),
            RopeLayers: src.Arch("rope.layers", Array.Empty<int>()),
            RopeLocalDims: src.Arch("rope.local_dim", src.Arch("rope.dim", headDim)),
            QkNormUnweighted: src.Arch("attention.qk_norm_unweighted", false),
            VNorm: src.Arch("attention.v_norm", false),
            PostNormEps: src.Arch("post_norm_eps", src.Arch("norm_eps", 1e-5f)),
            EmbeddingNorm: src.Arch("embedding_norm", false),
            LogitScale: src.Arch("logit_scale", 1f),
            PerLayerSize: src.Arch("per_layer.size", 0),
            NormType: src.Arch("norm.type", "rms"),
            PreNorm: src.Arch("pre_norm", true),
            ParallelResidual: src.Arch("residual.parallel", false),
            ResidualScale: src.Arch("residual.scale", 1f),
            SwigluAlpha: src.Arch("mlp.swiglu_alpha", 1.702f),
            SwigluLimit: src.Arch("mlp.swiglu_limit", 7f),
            XieluBeta: src.Arch("mlp.xielu_beta", 0.5f),
            XieluEps: src.Arch("mlp.xielu_eps", -1e-6f),
            ClipQkv: src.Arch("attention.clip_qkv", 0f),
            QkNormAfterRope: src.Arch("attention.qk_norm_after_rope", false),
            LayerHeadCounts: src.Arch("attention.layer_head_counts", Array.Empty<int>()));

        if (hp.NHead % hp.NHeadKv != 0)
        {
            throw new InvalidDataException("decoder: attention.head_count must be a multiple of attention.kv_head_count.");
        }

        if (hp.SlidingLayers.Length != 0 && hp.SlidingLayers.Length != hp.NLayer)
        {
            throw new InvalidDataException("decoder: attention.sliding_layers must have layer_count entries.");
        }

        foreach (var (arr, nm) in new[] { (hp.LayerHeadDims, "attention.layer_head_dims"), (hp.LayerHeadCounts, "attention.layer_head_counts"), (hp.LayerKvHeads, "attention.layer_kv_head_counts"), (hp.KvSource, "attention.kv_source_layers"), (hp.RopeLayers, "rope.layers") })
        {
            if (arr.Length != 0 && arr.Length != hp.NLayer)
            {
                throw new InvalidDataException($"decoder: {nm} must have layer_count entries.");
            }
        }

        return hp;
    }

    /// <summary>Head dimension of layer <paramref name="il"/>.</summary>
    public int HeadDimOf(int il) => LayerHeadDims.Length > 0 ? LayerHeadDims[il] : HeadDim;

    /// <summary>Number of query heads of layer <paramref name="il"/>.</summary>
    public int HeadsOf(int il) => LayerHeadCounts.Length > 0 ? LayerHeadCounts[il] : NHead;

    /// <summary>Number of key/value heads of layer <paramref name="il"/>.</summary>
    public int KvHeadsOf(int il) => LayerKvHeads.Length > 0 ? LayerKvHeads[il] : NHeadKv;

    /// <summary>The layer whose KV cache layer <paramref name="il"/> reads (its own index when it owns one).</summary>
    public int KvSourceOf(int il) => KvSource.Length > 0 ? KvSource[il] : il;

    /// <summary>Whether layer <paramref name="il"/> applies rotary embeddings (false for NoPE layers).</summary>
    public bool RopeOn(int il) => RopeLayers.Length == 0 || RopeLayers[il] != 0;

    // Local rope (base / no frequency scaling) belongs to the interleaved sliding layers only; a model that declares a window
    // without a layer list (Mistral v0.1: sliding_window) applies it to every layer and keeps the ordinary rope.
    /// <summary>Whether layer <paramref name="il"/> is one of the interleaved sliding-window layers (which use the local rope).</summary>
    public bool Sliding(int il) => Window > 0 && SlidingLayers.Length > 0 && SlidingLayers[il] != 0;

    /// <summary>Whether attention of layer <paramref name="il"/> is limited to the sliding window.</summary>
    public bool Windowed(int il) => Window > 0 && (SlidingLayers.Length == 0 || SlidingLayers[il] != 0);
}
