namespace Sharplight.ImageGeneration;

/// <summary>
/// The graph contract between image-generation adapters and the built-in latent-diffusion pipeline: graph names, input and output
/// names, the batch symbol and the dimension names a model is rebuilt with for other image sizes. An adapter whose model follows it
/// (any UNet or DiT denoiser with a VAE and one or two text encoders) is served by the pipeline without code of its own.
/// </summary>
/// <remarks>
/// Graphs and their values:
/// <list type="bullet">
/// <item><see cref="TextEncode"/> (component <c>text-encoder</c>): input <see cref="Tokens"/> i32 [L] (one padded prompt); outputs
/// <see cref="Hidden"/> f32 [L, D] (the per-token conditioning) and / or <see cref="Pooled"/> f32 [P].</item>
/// <item><see cref="TextEncode2"/> (optional; component <c>text-encoder</c> (SDXL's second CLIP encoder) or <c>text-encoder-2</c>
/// (FLUX's T5)): the same values. A <c>tokenizer-2</c> component, when present, tokenizes its prompts to its own
/// <see cref="DiffusionConfigKeys.ContextLength"/>; otherwise the first tokenizer's tokens are reused with the padding id
/// <see cref="DiffusionConfigKeys.SecondPadId"/> after the end token.</item>
/// <item><see cref="TextEncode3"/> (optional; component <see cref="ThirdTextEncoderRole"/> with its own <see cref="ThirdTokenizerRole"/>,
/// SD3's T5): the same values.</item>
/// <item>Conditioning: the <see cref="Hidden"/> outputs of the text graphs are joined per token ([L, D1 + D2]; graphs without one, such
/// as FLUX's CLIP encoder, only contribute their pooled embedding); a component whose config sets
/// <see cref="DiffusionConfigKeys.JoinSequence"/> (SD3's T5) appends its rows along the sequence instead, the narrower rows padded with
/// zeros to the wider width ([L1 + L3, max(D1 + D2, D3)]). The pooled embedding is the concatenation of every pooled output in encoder
/// order (SDXL: only the second encoder has one; FLUX: only the first; SD3: both CLIP encoders).</item>
/// <item><see cref="Denoise"/> (component <c>denoiser</c>): inputs <see cref="Latent"/> f32 [N, C, h, w] (the sampler's scaled model
/// input), <see cref="Timestep"/> f32 [N] (the sampler's model timestep; σ · 1000 for flow models) and <see cref="Context"/> f32
/// [N, L, D]; output <see cref="Prediction"/> f32 [N, C, h, w] (what the scheduler's prediction type says: noise, velocity, sample or
/// flow velocity). N is the symbolic <see cref="BatchSymbol"/> or a fixed batch (a denoiser built for one row is run once per row of
/// a guided step). Denoisers with added conditioning (SDXL) also take <see cref="Pooled"/> f32 [N, P] and
/// <see cref="SizeConditions"/> f32 [N, 6] (original height and width, crop top and left, target height and width, in pixels);
/// pooled-modulated transformers (FLUX) take <see cref="Pooled"/> alone, guidance-distilled ones (FLUX.1-dev) <see cref="Guidance"/>
/// f32 [N], the guidance scale, which then replaces classifier-free guidance.</item>
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

    /// <summary>The optional second text encoder graph (SDXL).</summary>
    public const string TextEncode2 = "text_encode_2";

    /// <summary>The optional third text encoder graph (SD3's T5 encoder, component <see cref="ThirdTextEncoderRole"/>).</summary>
    public const string TextEncode3 = "text_encode_3";

    /// <summary>The denoiser graph (one evaluation of the UNet / DiT).</summary>
    public const string Denoise = "denoise";

    /// <summary>The VAE decoder graph: model latents to images.</summary>
    public const string VaeDecode = "vae_decode";

    /// <summary>The VAE encoder graph: images to model latents.</summary>
    public const string VaeEncode = "vae_encode";

    /// <summary>Input of <see cref="TextEncode"/>: token ids.</summary>
    public const string Tokens = "tokens";

    /// <summary>
    /// Optional input of the text graphs: f32 [L], 1 for the prompt's tokens (with its end token) and 0 for the padding, for encoders run
    /// with the prompt's attention mask (Chroma's T5).
    /// </summary>
    public const string TokenMask = "token_mask";

    /// <summary>
    /// Optional input of <see cref="Denoise"/>: f32 [N, L], 1 for the conditioning rows the image may attend to and 0 for masked text
    /// padding (Chroma: the prompt's tokens and <see cref="DiffusionConfigKeys.ContextMaskPadding"/> padding tokens).
    /// </summary>
    public const string ContextMask = "context_mask";

    /// <summary>Output of <see cref="TextEncode"/>: per-token conditioning.</summary>
    public const string Hidden = "hidden";

    /// <summary>Optional output of <see cref="TextEncode"/> / <see cref="TextEncode2"/>: pooled (projected) text embedding; input of denoisers with added conditioning.</summary>
    public const string Pooled = "pooled";

    /// <summary>Input of guidance-distilled denoisers: the guidance scale per batch row (replaces classifier-free guidance).</summary>
    public const string Guidance = "guidance";

    /// <summary>The role of a second text-encoder component (FLUX's T5 encoder, with <see cref="TextEncode2"/>).</summary>
    public const string SecondTextEncoderRole = "text-encoder-2";

    /// <summary>The role of the tokenizer of the second text encoder when it has its own vocabulary (FLUX's T5).</summary>
    public const string SecondTokenizerRole = "tokenizer-2";

    /// <summary>The role of a third text-encoder component (SD3's T5 encoder, with <see cref="TextEncode3"/>).</summary>
    public const string ThirdTextEncoderRole = "text-encoder-3";

    /// <summary>The role of the tokenizer of the third text encoder (SD3's T5).</summary>
    public const string ThirdTokenizerRole = "tokenizer-3";

    /// <summary>Input of denoisers with added conditioning: original size, crop offset and target size per batch row.</summary>
    public const string SizeConditions = "size_conditions";

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

    /// <summary>
    /// Optional input of <see cref="Denoise"/> for editing models (FLUX.1 Kontext, FLUX.2): f32 [R, C, h, w], the model latents of R reference
    /// images (<see cref="ImageGenerationRequest.ReferenceImages"/>, resized to the output size and encoded by <see cref="VaeEncode"/>), whose
    /// tokens the transformer attends to beside the image's. The denoiser is built for R with the <see cref="References"/> dimension.
    /// </summary>
    public const string ReferenceLatent = "reference_latent";

    /// <summary>Dimension name (<c>AdapterOptions.Dimensions</c>): reference images of an editing denoiser (absent or 0: none).</summary>
    public const string References = "references";

    /// <summary>Dimension name (<c>AdapterOptions.Dimensions</c>): image height in pixels.</summary>
    public const string Height = "height";

    /// <summary>Dimension name (<c>AdapterOptions.Dimensions</c>): image width in pixels.</summary>
    public const string Width = "width";
}
