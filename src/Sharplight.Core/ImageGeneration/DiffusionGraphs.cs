namespace Sharplight.ImageGeneration;

/// <summary>
/// The graph contract between image-generation adapters and the built-in latent-diffusion pipeline: graph names, input and output
/// names, the batch symbol and the dimension names a model is rebuilt with for other image sizes. An adapter whose model follows it
/// (any UNet or DiT denoiser with a VAE and a text encoder) is served by the pipeline without code of its own.
/// </summary>
/// <remarks>
/// Graphs and their values:
/// <list type="bullet">
/// <item><see cref="TextEncode"/> (component <c>text-encoder</c>): input <see cref="Tokens"/> i32 [L] (one padded prompt); output
/// <see cref="Hidden"/> f32 [L, D] (the conditioning), optionally <see cref="Pooled"/> f32 [P].</item>
/// <item><see cref="Denoise"/> (component <c>denoiser</c>): inputs <see cref="Latent"/> f32 [N, C, h, w] (the sampler's scaled model
/// input), <see cref="Timestep"/> f32 [N] and <see cref="Context"/> f32 [N, L, D]; output <see cref="Prediction"/> f32 [N, C, h, w]
/// (what the scheduler's prediction type says: noise, velocity or sample). N is the symbolic <see cref="BatchSymbol"/>.</item>
/// <item><see cref="VaeDecode"/> (component <c>vae</c>): input <see cref="Latent"/> f32 [N, C, h, w] in model space (the graph undoes
/// the latent scaling); output <see cref="Image"/> f32 [N, 3, H, W] in [0, 1].</item>
/// <item><see cref="VaeEncode"/> (component <c>vae</c>): input <see cref="Image"/> f32 [N, 3, H, W] in [0, 1]; output <see cref="Latent"/>
/// f32 [N, C, h, w] in model space (the scaled mean of the latent distribution).</item>
/// </list>
/// Spatial sizes are fixed per graph; a model built for its native size is rebuilt for others with the <see cref="Height"/> and
/// <see cref="Width"/> dimensions (<c>TaskModel.DefinitionFor</c>).
/// </remarks>
public static class DiffusionGraphs
{
    /// <summary>The text encoder graph.</summary>
    public const string TextEncode = "text_encode";

    /// <summary>The denoiser graph (one evaluation of the UNet / DiT).</summary>
    public const string Denoise = "denoise";

    /// <summary>The VAE decoder graph: model latents to images.</summary>
    public const string VaeDecode = "vae_decode";

    /// <summary>The VAE encoder graph: images to model latents.</summary>
    public const string VaeEncode = "vae_encode";

    /// <summary>Input of <see cref="TextEncode"/>: token ids.</summary>
    public const string Tokens = "tokens";

    /// <summary>Output of <see cref="TextEncode"/>: per-token conditioning.</summary>
    public const string Hidden = "hidden";

    /// <summary>Optional output of <see cref="TextEncode"/>: pooled (projected) text embedding.</summary>
    public const string Pooled = "pooled";

    /// <summary>Latents: input of <see cref="Denoise"/> and <see cref="VaeDecode"/>, output of <see cref="VaeEncode"/>.</summary>
    public const string Latent = "latent";

    /// <summary>Input of <see cref="Denoise"/>: the model timestep per batch row.</summary>
    public const string Timestep = "timestep";

    /// <summary>Input of <see cref="Denoise"/>: the text conditioning per batch row.</summary>
    public const string Context = "context";

    /// <summary>Output of <see cref="Denoise"/>.</summary>
    public const string Prediction = "prediction";

    /// <summary>Images: output of <see cref="VaeDecode"/>, input of <see cref="VaeEncode"/>.</summary>
    public const string Image = "image";

    /// <summary>The symbolic batch axis of the denoiser and VAE graphs.</summary>
    public const string BatchSymbol = "N";

    /// <summary>Dimension name (<c>AdapterOptions.Dimensions</c>): image height in pixels.</summary>
    public const string Height = "height";

    /// <summary>Dimension name (<c>AdapterOptions.Dimensions</c>): image width in pixels.</summary>
    public const string Width = "width";
}
