namespace Sharplight.Import.Diffusers;

/// <summary>Hyper-parameters of a latent-diffusion UNet, written as the canonical <c>latent_diffusion.unet.*</c> keys.</summary>
internal sealed record UNetConfig
{
    /// <summary>Latent channels in.</summary>
    public required int InChannels { get; init; }
    /// <summary>Channels predicted.</summary>
    public required int OutChannels { get; init; }
    /// <summary>Channels per resolution level, finest first.</summary>
    public required int[] BlockChannels { get; init; }
    /// <summary>Residual blocks per level on the way down (the way up has one more).</summary>
    public required int LayersPerBlock { get; init; }
    /// <summary>Transformer blocks of the spatial transformer per level (0: the level has no attention).</summary>
    public required int[] TransformerDepths { get; init; }
    /// <summary>Attention heads per level.</summary>
    public required int[] HeadCounts { get; init; }
    /// <summary>Transformer blocks of the middle block's spatial transformer (0: none).</summary>
    public required int MidTransformerDepth { get; init; }
    /// <summary>Width of the text conditioning (cross-attention keys and values).</summary>
    public required int ContextDim { get; init; }
    /// <summary>GroupNorm groups.</summary>
    public int NormGroups { get; init; } = 32;
    /// <summary>GroupNorm epsilon.</summary>
    public float NormEps { get; init; } = 1e-5f;
    /// <summary>Width of the extra conditioning vector added to the time embedding (SDXL: pooled text + size embeddings), 0 for none.</summary>
    public int ClassEmbeddingInput { get; init; }
    /// <summary>Width of each sinusoidal size embedding of the extra conditioning (SDXL: 256).</summary>
    public int AdditionTimeDim { get; init; }

    /// <summary>Writes the keys under <c>{prefix}unet.</c>.</summary>
    public void WriteTo(IDictionary<string, object> metadata, string prefix)
    {
        string p = prefix + "unet.";
        metadata[p + "in_channels"] = InChannels;
        metadata[p + "out_channels"] = OutChannels;
        metadata[p + "block_channels"] = BlockChannels;
        metadata[p + "layers_per_block"] = LayersPerBlock;
        metadata[p + "transformer_depths"] = TransformerDepths;
        metadata[p + "head_counts"] = HeadCounts;
        metadata[p + "mid_transformer_depth"] = MidTransformerDepth;
        metadata[p + "context_dim"] = ContextDim;
        metadata[p + "norm_groups"] = NormGroups;
        metadata[p + "norm_eps"] = NormEps;
        metadata[p + "class_embedding_input"] = ClassEmbeddingInput;
        metadata[p + "addition_time_dim"] = AdditionTimeDim;
    }
}
