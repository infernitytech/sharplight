using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of <see cref="Sam2Adapter"/> read from the canonical metadata (<c>sam2.*</c>).</summary>
internal sealed record Sam2Parameters
{
    /// <summary>The side of the square model input.</summary>
    public required int ImageSize { get; init; }

    /// <summary>Kernel, stride and padding of the patch convolution.</summary>
    public required (int Kernel, int Stride, int Padding) Patch { get; init; }

    /// <summary>The side of the position table that is resampled to the first stage's grid.</summary>
    public required int PositionBackground { get; init; }

    /// <summary>The channel width of every stage.</summary>
    public required int[] EmbedDims { get; init; }

    /// <summary>The number of blocks of every stage.</summary>
    public required int[] Blocks { get; init; }

    /// <summary>The attention heads of every stage.</summary>
    public required int[] Heads { get; init; }

    /// <summary>The attention window of every stage (patches per side).</summary>
    public required int[] Windows { get; init; }

    /// <summary>The blocks (counted over all stages) that attend over the whole map.</summary>
    public required int[] GlobalBlocks { get; init; }

    /// <summary>The stride of the query pooling at the start of the stages that change resolution.</summary>
    public required int QueryStride { get; init; }

    /// <summary>The number of stages (after the first) that start with query pooling.</summary>
    public required int QueryPoolStages { get; init; }

    /// <summary>The MLP expansion of the backbone blocks.</summary>
    public required float MlpRatio { get; init; }

    /// <summary>The LayerNorm epsilon of the backbone and the neck.</summary>
    public required float Eps { get; init; }

    /// <summary>The channel width of the neck (the width of the image embedding).</summary>
    public required int NeckSize { get; init; }

    /// <summary>The levels (indices into the stages) whose neck output receives the upsampled coarser level.</summary>
    public required int[] TopDownLevels { get; init; }

    /// <summary>The attention heads of the mask decoder.</summary>
    public required int DecoderHeads { get; init; }

    /// <summary>The layers of the two-way transformer.</summary>
    public required int DecoderLayers { get; init; }

    /// <summary>The channel reduction of the decoder's cross attentions.</summary>
    public required int DecoderDownsample { get; init; }

    /// <summary>The number of mask tokens (one single-mask candidate and the multi-mask candidates).</summary>
    public required int MaskTokens { get; init; }

    /// <summary>The pixels per image-embedding cell of the prompt encoder.</summary>
    public required int PromptPatch { get; init; }

    /// <summary>Reads the parameters of a model source.</summary>
    /// <param name="src">The source.</param>
    public static Sam2Parameters From(ModelSource src) => new()
    {
        ImageSize = src.Arch<int>("image_size"),
        Patch = (src.Arch<int>("patch.kernel"), src.Arch<int>("patch.stride"), src.Arch<int>("patch.padding")),
        PositionBackground = src.Arch<int>("position_background"),
        EmbedDims = Ints(src, "embed_dims"),
        Blocks = Ints(src, "blocks"),
        Heads = Ints(src, "heads"),
        Windows = Ints(src, "windows"),
        GlobalBlocks = Ints(src, "global_blocks"),
        QueryStride = src.Arch<int>("query_stride"),
        QueryPoolStages = src.Arch<int>("query_pool_stages"),
        MlpRatio = src.Arch("mlp_ratio", 4f),
        Eps = src.Arch("norm_eps", 1e-6f),
        NeckSize = src.Arch<int>("neck.size"),
        TopDownLevels = Ints(src, "neck.top_down_levels"),
        DecoderHeads = src.Arch<int>("decoder.head_count"),
        DecoderLayers = src.Arch<int>("decoder.layer_count"),
        DecoderDownsample = src.Arch<int>("decoder.downsample"),
        MaskTokens = src.Arch<int>("decoder.mask_tokens"),
        PromptPatch = src.Arch<int>("prompt.patch_size"),
    };

    private static int[] Ints(ModelSource src, string key) => [.. src.Arch<string>(key).Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse)];
}
