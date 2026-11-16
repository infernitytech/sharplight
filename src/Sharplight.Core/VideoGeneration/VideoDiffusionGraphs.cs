namespace Sharplight.VideoGeneration;

/// <summary>
/// The graph contract between video-generation adapters and the built-in video diffusion pipeline (<see cref="VideoDiffusionGenerator"/>):
/// graph names, input and output names and the dimension names a model is rebuilt with for other sizes. Latent videos are
/// [C, T, h, w] (channels, latent frames, latent rows, latent columns), the layout of the reference implementations, so explicit
/// initial noise maps 1:1.
/// </summary>
/// <remarks>
/// Graphs and their values:
/// <list type="bullet">
/// <item><see cref="TextEncode"/> (component <c>text-encoder</c>): inputs <see cref="Tokens"/> i32 [L] (one prompt padded to the
/// context length) and <see cref="TokenMask"/> f32 [L] (1 for the prompt's tokens, 0 for the padding, which no token attends to);
/// output <see cref="Hidden"/> f32 [L, D]. The pipeline zeroes the padding rows (the conditioning of the Wan pipelines).</item>
/// <item><see cref="Denoise"/> (component <c>transformer</c>): inputs <see cref="Latent"/> f32 [C, T, h, w], <see cref="Timestep"/> f32
/// [1] (σ · 1000), or [tokens] when the denoiser takes per-token timesteps (<see cref="VideoConfigKeys.TokenTimesteps"/>: the tokens of
/// a conditioning first frame get 0), and <see cref="Context"/> f32 [L, D]; output <see cref="Prediction"/> f32 [C, T, h, w] (the
/// flow velocity). One video per run; classifier-free guidance runs the conditional and unconditional contexts one after the other.</item>
/// <item><see cref="VaeDecodeFirst"/> and <see cref="VaeDecodeNext"/> (component <c>vae</c>): the causal VAE decoder, streamed over the
/// latent frames. The first graph decodes latent frame 0 ([C, h, w] in model space; the graph undoes the latent normalization) into
/// one frame, the next graph one further latent frame into <see cref="VideoConfigKeys.TemporalDownscale"/> frames; output
/// <see cref="Frames"/> f32 [n, 3, H, W] in [0, 1]. The causal convolutions carry their last two input frames from chunk to chunk:
/// the next graph takes them as inputs <c>cache_in.{name}</c> [2, Cᵢ, Hᵢ, Wᵢ] and both graphs return the new ones as outputs
/// <c>cache_out.{name}</c> (the last one or two input frames); the pipeline keeps, per name, the last two frames it has seen
/// (zeros before the first), so a cache the first graph does not return stays zero.</item>
/// <item><see cref="VaeEncode"/> (component <c>vae</c>, optional): the causal VAE encoder of one image, input <see cref="Image"/> f32
/// [3, H, W] in [0, 1]; output <see cref="Latent"/> f32 [C, h, w] (the normalized mean of the first latent frame), for image-to-video.</item>
/// </list>
/// Sizes are fixed per graph; a model is rebuilt for other sizes with the <see cref="Frames"/> (latent frames), <see cref="Height"/> and
/// <see cref="Width"/> (pixels) dimensions (<c>TaskModel.DefinitionFor</c>).
/// </remarks>
public static class VideoDiffusionGraphs
{
    /// <summary>The text encoder graph.</summary>
    public const string TextEncode = "text_encode";

    /// <summary>The denoiser graph (one evaluation of the video transformer).</summary>
    public const string Denoise = "denoise";

    /// <summary>The VAE decoder graph of the first latent frame.</summary>
    public const string VaeDecodeFirst = "vae_decode_first";

    /// <summary>The VAE decoder graph of each further latent frame.</summary>
    public const string VaeDecodeNext = "vae_decode_next";

    /// <summary>The VAE encoder graph of one image (the first latent frame).</summary>
    public const string VaeEncode = "vae_encode";

    /// <summary>Input of <see cref="TextEncode"/>: token ids.</summary>
    public const string Tokens = "tokens";

    /// <summary>Input of <see cref="TextEncode"/>: 1 for prompt tokens, 0 for padding.</summary>
    public const string TokenMask = "token_mask";

    /// <summary>Output of <see cref="TextEncode"/>: per-token conditioning.</summary>
    public const string Hidden = "hidden";

    /// <summary>Latents: input of <see cref="Denoise"/> and the decoders, output of <see cref="VaeEncode"/>.</summary>
    public const string Latent = "latent";

    /// <summary>Input of <see cref="Denoise"/>: the model timestep (per video or per token).</summary>
    public const string Timestep = "timestep";

    /// <summary>Input of <see cref="Denoise"/>: the text conditioning.</summary>
    public const string Context = "context";

    /// <summary>Output of <see cref="Denoise"/>.</summary>
    public const string Prediction = "prediction";

    /// <summary>Output of the decoders: frames [n, 3, H, W] in [0, 1].</summary>
    public const string Frames = "frames";

    /// <summary>Input of <see cref="VaeEncode"/>: one image [3, H, W] in [0, 1].</summary>
    public const string Image = "image";

    /// <summary>Name prefix of the decoders' cache inputs.</summary>
    public const string CacheInPrefix = "cache_in.";

    /// <summary>Name prefix of the decoders' cache outputs.</summary>
    public const string CacheOutPrefix = "cache_out.";

    /// <summary>Dimension name (<c>AdapterOptions.Dimensions</c>): latent frames of the denoiser.</summary>
    public const string FramesDimension = "frames";

    /// <summary>Dimension name (<c>AdapterOptions.Dimensions</c>): video height in pixels.</summary>
    public const string Height = "height";

    /// <summary>Dimension name (<c>AdapterOptions.Dimensions</c>): video width in pixels.</summary>
    public const string Width = "width";
}
