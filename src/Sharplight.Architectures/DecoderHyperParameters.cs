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
    MoeOptions Moe, bool RouterReadsLayerInput,
    int[] LayerHeadDims, int[] LayerKvHeads, int[] KvSource, int[] RopeLayers, int RopeLocalDims, bool QkNormUnweighted, bool VNorm,
    float PostNormEps, bool EmbeddingNorm, float LogitScale, int PerLayerSize,
    string NormType, bool PreNorm, bool ParallelResidual, float ResidualScale, float SwigluAlpha, float SwigluLimit,
    float XieluBeta, float XieluEps, float ClipQkv, bool QkNormAfterRope, int[] LayerHeadCounts,
    int[] QkNormLayers, bool RopeLocalFreqFactors, float TemperatureScale, float TemperatureLength, float TemperatureOffset, int[] TemperatureLayers,
    int ChunkSize, int[] ChunkLayers, int ValueDim, float ValueScale, float RopeAttentionScale, string GateActivation, float AlibiMaxBias,
    float[] SwigluClamp, bool SwigluClampInput, float[] MoeSwigluClamp, bool MoeSwigluClampInput, int LayerLoops, bool LoopNorm, bool Causal,
    bool TemperaturePower, bool NormPreservingResidual, bool AttentionHeadNorm, bool PostNormWeightFirst, int[] RopeSections, bool RopeSectionsInterleaved)
{
    /// <summary>Number of M-RoPE position axes (time, row, column, extra) when <see cref="RopeSections"/> is set, else 1.</summary>
    public const int MRopeAxes = 4;

    /// <summary>Position axes of the decode graphs: <see cref="MRopeAxes"/> with M-RoPE, otherwise 1.</summary>
    public int PositionAxes => RopeSections.Length > 0 ? MRopeAxes : 1;

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
            AttnScale: src.Arch("attention.scale", 1f / MathF.Sqrt(headDim)),
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
            Moe: MoeOptions.Read(src),
            RouterReadsLayerInput: src.Arch("moe.router_input", "normed") == "layer_input",
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
            LayerHeadCounts: src.Arch("attention.layer_head_counts", Array.Empty<int>()),
            QkNormLayers: src.Arch("attention.qk_norm_layers", Array.Empty<int>()),
            RopeLocalFreqFactors: src.Arch("rope.local_freq_factors", false),
            TemperatureScale: src.Arch("attention.temperature_scale", 0f),
            TemperatureLength: src.Arch("attention.temperature_length", 0f),
            TemperatureOffset: src.Arch("attention.temperature_offset", 0f),
            TemperatureLayers: src.Arch("attention.temperature_layers", Array.Empty<int>()),
            ChunkSize: src.Arch("attention.chunk_size", 0),
            ChunkLayers: src.Arch("attention.chunk_layers", Array.Empty<int>()),
            ValueDim: src.Arch("attention.value_dim", 0),
            ValueScale: src.Arch("attention.value_scale", 1f),
            RopeAttentionScale: ropeAttn,
            GateActivation: src.Arch("attention.gate_activation", "sigmoid"),
            AlibiMaxBias: src.Arch("attention.alibi_max_bias", 0f),
            SwigluClamp: src.Arch("mlp.swiglu_clamp", Array.Empty<float>()),
            SwigluClampInput: src.Arch("mlp.swiglu_clamp_input", false),
            MoeSwigluClamp: src.Arch("moe.swiglu_clamp", Array.Empty<float>()),
            MoeSwigluClampInput: src.Arch("moe.swiglu_clamp_input", false),
            LayerLoops: src.Arch("layer_loops", 1),
            LoopNorm: src.Arch("loop_norm", true),
            Causal: src.Arch("attention.causal", true),
            TemperaturePower: src.Arch("attention.temperature_power", false),
            NormPreservingResidual: src.Arch("residual.norm_preserving", false),
            AttentionHeadNorm: src.Arch("attention.out_head_norm", false),
            PostNormWeightFirst: src.Arch("post_norm.weight_first", false),
            RopeSections: src.Arch("rope.sections", Array.Empty<int>()),
            RopeSectionsInterleaved: src.Arch("rope.sections_interleaved", false));

        if (hp.NHead % hp.NHeadKv != 0)
        {
            throw new InvalidDataException("decoder: attention.head_count must be a multiple of attention.kv_head_count.");
        }

        if (hp.SlidingLayers.Length != 0 && hp.SlidingLayers.Length != hp.NLayer)
        {
            throw new InvalidDataException("decoder: attention.sliding_layers must have layer_count entries.");
        }

        foreach (var (arr, nm) in new[] { (hp.LayerHeadDims, "attention.layer_head_dims"), (hp.LayerHeadCounts, "attention.layer_head_counts"), (hp.LayerKvHeads, "attention.layer_kv_head_counts"), (hp.KvSource, "attention.kv_source_layers"), (hp.RopeLayers, "rope.layers"), (hp.QkNormLayers, "attention.qk_norm_layers"), (hp.TemperatureLayers, "attention.temperature_layers"), (hp.ChunkLayers, "attention.chunk_layers") })
        {
            if (arr.Length != 0 && arr.Length != hp.NLayer)
            {
                throw new InvalidDataException($"decoder: {nm} must have layer_count entries.");
            }
        }

        if ((hp.SwigluClamp.Length != 0 && hp.SwigluClamp.Length != hp.NLayer) || (hp.MoeSwigluClamp.Length != 0 && hp.MoeSwigluClamp.Length != hp.NLayer))
        {
            throw new InvalidDataException("decoder: mlp.swiglu_clamp and moe.swiglu_clamp must have layer_count entries.");
        }

        if (hp.LayerLoops < 1)
        {
            throw new InvalidDataException("decoder: layer_loops must be at least 1.");
        }

        if (hp.TemperatureScale != 0f && hp.TemperatureLength <= 0f)
        {
            throw new InvalidDataException("decoder: attention.temperature_scale needs a positive attention.temperature_length.");
        }

        return hp;
    }

    /// <summary>
    /// True when every layer has the same shape and attention kind (none of the per-layer arrays, no per-layer embeddings), so a
    /// layer index past <see cref="NLayer"/> (an MTP layer) has the hyper-parameters of the trunk's layers.
    /// </summary>
    public bool UniformLayers => LayerHeadDims.Length == 0 && LayerHeadCounts.Length == 0 && LayerKvHeads.Length == 0 && KvSource.Length == 0
        && RopeLayers.Length == 0 && SlidingLayers.Length == 0 && QkNormLayers.Length == 0 && TemperatureLayers.Length == 0
        && ChunkLayers.Length == 0 && PerLayerSize == 0 && SwigluClamp.Length == 0 && MoeSwigluClamp.Length == 0 && LayerLoops == 1;

    /// <summary>Head dimension of layer <paramref name="il"/>.</summary>
    public int HeadDimOf(int il) => LayerHeadDims.Length > 0 ? LayerHeadDims[il] : HeadDim;

    /// <summary>Value head dimension of layer <paramref name="il"/> (the key head dimension unless <c>attention.value_dim</c> is set).</summary>
    public int ValueDimOf(int il) => ValueDim > 0 ? ValueDim : HeadDimOf(il);

    /// <summary>
    /// ALiBi slopes of <paramref name="heads"/> query heads (the geometric sequence of the ALiBi paper, as BLOOM and llama.cpp compute it for the
    /// largest power of two of heads and the interleaved halves beyond it), or null when <c>attention.alibi_max_bias</c> is 0.
    /// </summary>
    public float[]? AlibiSlopes(int heads) => Architectures.AlibiSlopes.Of(heads, AlibiMaxBias);

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

    /// <summary>Whether layer <paramref name="il"/> applies the unweighted q/k norm (<c>attention.qk_norm_unweighted</c>, limited by <c>attention.qk_norm_layers</c>).</summary>
    public bool QkNormUnweightedOn(int il) => QkNormUnweighted && (QkNormLayers.Length == 0 || QkNormLayers[il] != 0);

    /// <summary>Whether layer <paramref name="il"/> scales its queries by the position-dependent attention temperature.</summary>
    public bool TemperatureOn(int il) => TemperatureScale != 0f && (TemperatureLayers.Length == 0 || TemperatureLayers[il] != 0);

    /// <summary>Whether layer <paramref name="il"/> attends only within its chunk of <c>attention.chunk_size</c> positions.</summary>
    public bool Chunked(int il) => ChunkSize > 0 && (ChunkLayers.Length == 0 || ChunkLayers[il] != 0);

    /// <summary>SwiGLU clamp limit of the dense MLP and shared expert of layer <paramref name="il"/> (0 = none).</summary>
    public float SwigluClampOf(int il) => SwigluClamp.Length > 0 ? SwigluClamp[il] : 0f;

    /// <summary>SwiGLU clamp limit of the routed experts of layer <paramref name="il"/> (0 = none).</summary>
    public float MoeSwigluClampOf(int il) => MoeSwigluClamp.Length > 0 ? MoeSwigluClamp[il] : 0f;

    /// <summary>Number of logical layers: the physical layers run <see cref="LayerLoops"/> times with shared weights.</summary>
    public int LogicalLayers => NLayer * LayerLoops;

    /// <summary>Whether attention of layer <paramref name="il"/> is limited to the sliding window.</summary>
    public bool Windowed(int il) => Window > 0 && (SlidingLayers.Length == 0 || SlidingLayers[il] != 0);
}
