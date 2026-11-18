using Sharplight.ImageGeneration;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Hyper-parameters of a FLUX-style rectified-flow transformer (MMDiT: dual-stream blocks over separate text and image streams, then
/// single-stream blocks over the joined sequence), read from <c>flux.transformer.*</c>.
/// </summary>
internal sealed record FluxHyperParameters
{
    /// <summary>Latent channels (before patching).</summary>
    public required int LatentChannels { get; init; }

    /// <summary>Latent pixels per token along each axis.</summary>
    public required int PatchSize { get; init; }

    /// <summary>Model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Dual-stream blocks.</summary>
    public required int DoubleLayers { get; init; }

    /// <summary>Single-stream blocks.</summary>
    public required int SingleLayers { get; init; }

    /// <summary>Width of the text conditioning (the T5 hidden size).</summary>
    public required int ContextDim { get; init; }

    /// <summary>Width of the pooled text embedding (CLIP; 0 without, Chroma).</summary>
    public required int PooledDim { get; init; }

    /// <summary>Whether the guidance scale is embedded (guidance-distilled models, FLUX.1-dev).</summary>
    public required bool GuidanceEmbed { get; init; }

    /// <summary>Rotated head dimensions per position axis (text/index, row, column); they sum to the head width.</summary>
    public required int[] AxesDim { get; init; }

    /// <summary>RoPE base.</summary>
    public required float RopeTheta { get; init; }

    /// <summary>Native image size in pixels.</summary>
    public required int SampleSize { get; init; }

    /// <summary>
    /// Layers of the modulation approximator (Chroma's "distilled guidance" MLP that replaces the per-block modulation projections and
    /// the timestep / pooled-text embedders); 0 for FLUX.
    /// </summary>
    public int ApproximatorLayers { get; init; }

    /// <summary>Input width of the approximator (sinusoidal timestep, guidance and modulation-index channels; Chroma: 64).</summary>
    public int ApproximatorChannels { get; init; } = 64;

    /// <summary>Whether the text padding is masked in attention (Chroma: <see cref="DiffusionGraphs.ContextMask"/> input).</summary>
    public bool MaskedText { get; init; }

    /// <summary>
    /// Whether the RoPE positions are centred (Qwen-Image: image rows and columns around 0, the text after the largest image position)
    /// instead of FLUX's (text at the origin, image from (0, 0)).
    /// </summary>
    public bool CenteredPositions { get; init; }

    /// <summary>
    /// Whether the MLP down projections keep f32 activations in the integer-dot kernels of quantized weights (Qwen-Image, whose MLP inputs
    /// are too large for the int8 activation quantization).
    /// </summary>
    public bool F32Activations { get; init; }

    /// <summary>
    /// Whether the modulations are shared by all blocks (FLUX.2: one projection of the conditioning for the image streams, one for the
    /// text streams and one for the single-stream blocks: <c>transformer.mod_img</c> [6D, D], <c>mod_txt</c> [6D, D], <c>mod_single</c>
    /// [3D, D]) instead of one per block.
    /// </summary>
    public bool SharedModulation { get; init; }

    /// <summary>Whether the feed-forwards are SwiGLU with a fused gate / up input (FLUX.2: silu(first half) · second half) instead of GELU.</summary>
    public bool SwiGlu { get; init; }

    /// <summary>
    /// The position axis on which text tokens count up (FLUX.2's fourth axis: text j at (0, 0, 0, j), image tokens at (0, row, column, 0)),
    /// or −1 (FLUX: text at the origin).
    /// </summary>
    public int TextPositionAxis { get; init; } = -1;

    /// <summary>
    /// How many reference images an editing model takes (FLUX.1 Kontext: 1; FLUX.2: several), or 0. Reference tokens follow the image tokens
    /// with the first position axis at <see cref="ReferenceIndexBase"/> + <see cref="ReferenceIndexStep"/> · i for reference i.
    /// </summary>
    public int MaxReferences { get; init; }

    /// <summary>The first position axis of the first reference image's tokens (FLUX.1 Kontext: 1; FLUX.2: 10).</summary>
    public int ReferenceIndexBase { get; init; } = 1;

    /// <summary>The step of the first position axis between reference images (FLUX.2: 10).</summary>
    public int ReferenceIndexStep { get; init; }

    /// <summary>
    /// Whether the reference images' tokens are modulated by the conditioning at timestep 0 instead of the current one (Qwen-Image-Edit-2511,
    /// diffusers' <c>zero_cond_t</c>): their image-stream modulations (not the final one) come from a second evaluation of the timestep
    /// embedding at 0.
    /// </summary>
    public bool ZeroConditionTimestep { get; init; }

    /// <summary>Width per head.</summary>
    public int HeadDim => HiddenSize / HeadCount;

    /// <summary>Reads the keys.</summary>
    /// <param name="source">The canonical source.</param>
    /// <param name="prefix">The architecture prefix (<c>flux.</c>, <c>chroma.</c>).</param>
    public static FluxHyperParameters Read(ModelSource source, string prefix)
    {
        string p = prefix + "transformer.";
        return new FluxHyperParameters
        {
            LatentChannels = source.Get(p + "latent_channels", 16),
            PatchSize = source.Get(p + "patch_size", 2),
            HiddenSize = source.Get<int>(p + "hidden_size"),
            HeadCount = source.Get<int>(p + "head_count"),
            DoubleLayers = source.Get<int>(p + "double_layers"),
            SingleLayers = source.Get<int>(p + "single_layers"),
            ContextDim = source.Get<int>(p + "context_dim"),
            PooledDim = source.Get(p + "pooled_dim", 0),
            GuidanceEmbed = source.Get(p + "guidance_embed", false),
            AxesDim = source.Get<int[]>(p + "axes_dim", [16, 56, 56]),
            RopeTheta = source.Get(p + "rope_theta", 10000f),
            SampleSize = source.Get(p + "sample_size", 1024),
            ApproximatorLayers = source.Get(p + "approximator_layers", 0),
            ApproximatorChannels = source.Get(p + "approximator_channels", 64),
            MaskedText = source.Get(p + "masked_text", false),
            CenteredPositions = source.Get(p + "centered_positions", false),
            F32Activations = source.Get(p + "f32_activations", false),
            SharedModulation = source.Get(p + "shared_modulation", false),
            SwiGlu = source.Get(p + "swiglu", false),
            TextPositionAxis = source.Get(p + "text_position_axis", -1),
            MaxReferences = source.Get(p + "max_references", 0),
            ReferenceIndexBase = source.Get(p + "reference_index_base", 1),
            ReferenceIndexStep = source.Get(p + "reference_index_step", 0),
            ZeroConditionTimestep = source.Get(p + "zero_cond_t", false),
        };
    }
}
