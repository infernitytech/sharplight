using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of a latent-diffusion UNet (canonical keys <c>latent_diffusion.unet.*</c>).</summary>
internal sealed record UNetHyperParameters
{
    /// <summary>Latent channels in.</summary>
    public required int InChannels { get; init; }

    /// <summary>Channels predicted.</summary>
    public required int OutChannels { get; init; }

    /// <summary>Channels per level, finest first.</summary>
    public required int[] BlockChannels { get; init; }

    /// <summary>Residual blocks per level on the way down (one more on the way up).</summary>
    public required int LayersPerBlock { get; init; }

    /// <summary>Transformer blocks per spatial transformer on the way down, finest first (0: no attention).</summary>
    public required int[] DownTransformerDepths { get; init; }

    /// <summary>Transformer blocks per spatial transformer on the way up, coarsest first (0: no attention).</summary>
    public required int[] UpTransformerDepths { get; init; }

    /// <summary>Attention heads per level, finest first.</summary>
    public required int[] HeadCounts { get; init; }

    /// <summary>Transformer blocks of the middle block (0: none).</summary>
    public required int MidTransformerDepth { get; init; }

    /// <summary>Width of the cross-attention context.</summary>
    public required int ContextDim { get; init; }

    /// <summary>Native latent height and width.</summary>
    public required int SampleSize { get; init; }

    /// <summary>GroupNorm groups.</summary>
    public required int NormGroups { get; init; }

    /// <summary>GroupNorm epsilon of the residual blocks and the output norm.</summary>
    public required float NormEps { get; init; }

    /// <summary>Whether the sinusoidal timestep embedding puts the cosines first.</summary>
    public required bool FlipSinToCos { get; init; }

    /// <summary>Frequency shift of the timestep embedding.</summary>
    public required float FreqShift { get; init; }

    /// <summary>Width of the added conditioning vector (SDXL: pooled text embedding + size embeddings), 0 for none.</summary>
    public int ClassEmbeddingInput { get; init; }

    /// <summary>Width of the sinusoidal embedding of each size condition (SDXL: 256), 0 for none.</summary>
    public int AdditionTimeDim { get; init; }

    /// <summary>
    /// Size conditions embedded next to the pooled text embedding (<see cref="Sharplight.ImageGeneration.DiffusionGraphs.SizeConditions"/>):
    /// SDXL's six (original size, crop corner, target size) or the SDXL refiner's five (original size, crop corner, aesthetic score). Not stored:
    /// the adapter derives it from <see cref="ClassEmbeddingInput"/> and the pooled embedding's width (<see cref="WithPooledWidth"/>).
    /// </summary>
    public int SizeConditions { get; init; } = 6;

    /// <summary>
    /// The parameters with <see cref="SizeConditions"/> derived from the width of the pooled text embedding the UNet receives: the rest of
    /// <see cref="ClassEmbeddingInput"/> holds one <see cref="AdditionTimeDim"/>-wide embedding per condition.
    /// </summary>
    /// <param name="pooledWidth">The pooled embedding's width (SDXL and its refiner: OpenCLIP bigG's projection, 1280); 0 when unknown.</param>
    public UNetHyperParameters WithPooledWidth(int pooledWidth)
    {
        if (ClassEmbeddingInput <= 0 || AdditionTimeDim <= 0 || pooledWidth <= 0 || pooledWidth >= ClassEmbeddingInput
            || (ClassEmbeddingInput - pooledWidth) % AdditionTimeDim != 0)
        {
            return this;
        }

        return this with { SizeConditions = (ClassEmbeddingInput - pooledWidth) / AdditionTimeDim };
    }

    /// <summary>
    /// Input channels beyond the predicted latent channels (InstructPix2Pix: 4 image-latent channels; the inpainting UNets: 1 mask and 4
    /// masked-image channels), fed through the <see cref="Sharplight.ImageGeneration.DiffusionGraphs.ConditionLatent"/> input.
    /// </summary>
    public int ConditionChannels => Math.Max(0, InChannels - OutChannels);

    /// <summary>Reads the keys under <c>{prefix}unet.</c>.</summary>
    /// <param name="source">The canonical source.</param>
    /// <param name="prefix">Architecture prefix (<c>latent_diffusion.</c>).</param>
    public static UNetHyperParameters Read(ModelSource source, string prefix)
    {
        string p = prefix + "unet.";
        return new UNetHyperParameters
        {
            InChannels = source.Get<int>(p + "in_channels"),
            OutChannels = source.Get<int>(p + "out_channels"),
            BlockChannels = source.Get<int[]>(p + "block_channels"),
            LayersPerBlock = source.Get<int>(p + "layers_per_block"),
            DownTransformerDepths = source.Get<int[]>(p + "down_transformer_depths"),
            UpTransformerDepths = source.Get<int[]>(p + "up_transformer_depths"),
            HeadCounts = source.Get<int[]>(p + "head_counts"),
            MidTransformerDepth = source.Get<int>(p + "mid_transformer_depth"),
            ContextDim = source.Get<int>(p + "context_dim"),
            SampleSize = source.Get<int>(p + "sample_size"),
            NormGroups = source.Get(p + "norm_groups", 32),
            NormEps = source.Get(p + "norm_eps", 1e-5f),
            FlipSinToCos = source.Get(p + "flip_sin_to_cos", true),
            FreqShift = source.Get(p + "freq_shift", 0f),
            ClassEmbeddingInput = source.Get(p + "class_embedding_input", 0),
            AdditionTimeDim = source.Get(p + "addition_time_dim", 0),
        };
    }
}
