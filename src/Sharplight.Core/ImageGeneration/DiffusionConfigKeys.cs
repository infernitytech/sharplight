namespace Sharplight.ImageGeneration;

/// <summary>
/// Configuration keys of image-generation model descriptions (<see cref="Models.ModelComponent.Config"/> and
/// <see cref="Models.ModelDescription.Config"/>) that the built-in latent-diffusion pipeline reads. Adapters write them; values the
/// adapter leaves out take the defaults named here.
/// </summary>
public static class DiffusionConfigKeys
{
    /// <summary>Model: native image width in pixels.</summary>
    public const string Width = "width";

    /// <summary>Model: native image height in pixels.</summary>
    public const string Height = "height";

    /// <summary>Model: default number of denoising steps (default 25).</summary>
    public const string Steps = "steps";

    /// <summary>
    /// Model: default guidance scale (default 7.5; distilled models set 1). For denoisers that take the scale as an input
    /// (<see cref="DiffusionGraphs.Guidance"/>, guidance-distilled FLUX.1-dev) it is that input instead of classifier-free guidance.
    /// </summary>
    public const string GuidanceScale = "guidanceScale";

    /// <summary>
    /// Model: whether a guidance scale above 1 runs classifier-free guidance (default true). Distilled models that were not trained for
    /// it (FLUX.1-schnell, FLUX.1-dev) set false: the scale then only feeds a <see cref="DiffusionGraphs.Guidance"/> input, if any.
    /// </summary>
    public const string ClassifierFreeGuidance = "classifierFreeGuidance";

    /// <summary>Denoiser and VAE: latent channels.</summary>
    public const string LatentChannels = "latentChannels";

    /// <summary>VAE: pixels per latent along each axis (8 for the SD VAEs).</summary>
    public const string Downscale = "downscale";

    /// <summary>VAE: optional [latent channels × 3] row-major map from model latents to RGB in [-1, 1], for step previews.</summary>
    public const string LatentRgbFactors = "latentRgbFactors";

    /// <summary>Text encoder and tokenizer: tokens per prompt, including the start and end tokens (77 for CLIP).</summary>
    public const string ContextLength = "contextLength";

    /// <summary>
    /// Tokenizer: a template the prompt is put into before tokenizing, with <c>{prompt}</c> where the prompt goes (a language-model text
    /// encoder's chat template, Z-Image); special tokens in the template are parsed, in the prompt they are not.
    /// </summary>
    public const string PromptTemplate = "promptTemplate";

    /// <summary>Text encoder: the padding token id of the second encoder's prompts (SDXL's OpenCLIP tokenizer pads with <c>!</c>).</summary>
    public const string SecondPadId = "secondPadId";

    /// <summary>
    /// Text encoder: when no negative prompt is given, the unconditional conditioning is all zeros instead of the empty prompt's
    /// encoding (diffusers' <c>force_zeros_for_empty_prompt</c>, SDXL).
    /// </summary>
    public const string ZeroEmptyNegative = "zeroEmptyNegative";

    /// <summary>
    /// Text encoder: the component's hidden states are appended to the other encoders' along the sequence (zero-padded to the wider
    /// width) instead of being joined per token (SD3's T5 encoder; default false).
    /// </summary>
    public const string JoinSequence = "joinSequence";

    /// <summary>
    /// Text encoder: leading tokens of the templated prompt whose hidden states are dropped from the conditioning (Qwen-Image: the 34
    /// tokens of its system prompt; default 0). The encoder's context length counts them; the denoiser sees the rows after them.
    /// </summary>
    public const string PromptSkip = "promptSkip";

    /// <summary>
    /// Tokenizer: the text an empty prompt (or negative prompt) is replaced with before tokenizing (Qwen-Image 2.1: " ", its encoder has
    /// no start token to read otherwise); absent: empty prompts stay empty.
    /// </summary>
    public const string EmptyPrompt = "emptyPrompt";

    /// <summary>Scheduler: number of training timesteps (default 1000).</summary>
    public const string TrainSteps = "trainSteps";

    /// <summary>Scheduler: first beta of the training schedule.</summary>
    public const string BetaStart = "betaStart";

    /// <summary>Scheduler: last beta of the training schedule.</summary>
    public const string BetaEnd = "betaEnd";

    /// <summary>Scheduler: <c>linear</c>, <c>scaled_linear</c> or <c>squaredcos_cap_v2</c>.</summary>
    public const string BetaSchedule = "betaSchedule";

    /// <summary>Scheduler: what the denoiser predicts, <c>epsilon</c>, <c>v_prediction</c> or <c>sample</c>.</summary>
    public const string PredictionType = "predictionType";

    /// <summary>Scheduler: spacing of the inference timesteps, <c>leading</c>, <c>trailing</c> or <c>linspace</c>.</summary>
    public const string TimestepSpacing = "timestepSpacing";

    /// <summary>Scheduler: offset added to <c>leading</c> timesteps.</summary>
    public const string StepsOffset = "stepsOffset";

    /// <summary>Scheduler: whether DDIM's last step goes to alpha 1 instead of the first training alpha.</summary>
    public const string SetAlphaToOne = "setAlphaToOne";

    /// <summary>Scheduler: whether the betas are rescaled to zero terminal SNR.</summary>
    public const string ZeroTerminalSnr = "zeroTerminalSnr";

    /// <summary>Scheduler: whether DDIM clips its clean-sample estimate to [-1, 1] (absent: the sampler's default, on).</summary>
    public const string ClipSample = "clipSample";

    /// <summary>Scheduler: the default sampler (<see cref="Samplers"/> names).</summary>
    public const string Sampler = "sampler";

    /// <summary>Scheduler (flow): the static shift of the noise levels (default 1, see <see cref="NoiseSchedule.Shift"/>).</summary>
    public const string Shift = "shift";

    /// <summary>Scheduler (flow): whether the shift depends on the image size (see <see cref="NoiseSchedule.DynamicShift"/>).</summary>
    public const string DynamicShift = "dynamicShift";

    /// <summary>Scheduler (flow): μ of the dynamic shift at <see cref="BaseSequenceLength"/> image tokens (default 0.5).</summary>
    public const string BaseShift = "baseShift";

    /// <summary>Scheduler (flow): μ of the dynamic shift at <see cref="MaxSequenceLength"/> image tokens (default 1.15).</summary>
    public const string MaxShift = "maxShift";

    /// <summary>Scheduler (flow): image tokens of <see cref="BaseShift"/> (default 256).</summary>
    public const string BaseSequenceLength = "baseSequenceLength";

    /// <summary>Scheduler (flow): image tokens of <see cref="MaxShift"/> (default 4096).</summary>
    public const string MaxSequenceLength = "maxSequenceLength";

    /// <summary>
    /// Denoiser: the padding tokens after the prompt that stay visible in a <see cref="DiffusionGraphs.ContextMask"/> (Chroma: 1, the
    /// first padding token; default 0).
    /// </summary>
    public const string ContextMaskPadding = "contextMaskPadding";

    /// <summary>
    /// Scheduler (flow): the noise level the shifted schedule is stretched to end at before the final 0 (diffusers'
    /// <c>shift_terminal</c>, LTX-Video: 0.1; 0 or absent: no stretch).
    /// </summary>
    public const string ShiftTerminal = "shiftTerminal";

    /// <summary>
    /// Scheduler (flow): the noise levels the pipeline passes for its default number of steps, before the shift (diffusers'
    /// <c>sample_sigmas</c> pipeline setting, Qwen-Image 2.1 Turbo); absent: σ = linspace(1, 1 / n).
    /// </summary>
    public const string SampleSigmas = "sampleSigmas";

    /// <summary>
    /// Scheduler (flow, dynamic shift): μ from FLUX.2's empirical fit over the image token count and the step count (diffusers'
    /// <c>compute_empirical_mu</c>) instead of the linear interpolation between the base and maximum sequence lengths (default false).
    /// </summary>
    public const string EmpiricalShift = "empiricalShift";

    /// <summary>
    /// Denoiser: latent pixels per token along each axis (transformers that patchify the latents: FLUX 2; default 1). The image token
    /// count of a dynamic flow shift is the latent pixel count divided by its square.
    /// </summary>
    public const string PatchSize = "patchSize";

    /// <summary>
    /// Denoiser: classifier-free guidance rescales the guided prediction of every token to the norm of the conditional one (diffusers'
    /// Qwen-Image pipelines: the norm over the token's channels and patch pixels; default false).
    /// </summary>
    public const string GuidanceRenormalization = "guidanceRenormalization";

    /// <summary>
    /// Denoiser: how many reference images an editing model takes (<see cref="DiffusionGraphs.ReferenceLatent"/>; FLUX.1 Kontext 1, FLUX.2
    /// several); absent or 0: the model does not edit.
    /// </summary>
    public const string ReferenceImages = "referenceImages";

    /// <summary>
    /// Denoiser: what the extra input channels of a UNet built with <see cref="DiffusionGraphs.ConditionLatent"/> carry:
    /// <see cref="ConditionImage"/> (InstructPix2Pix: the input image's latents) or <see cref="ConditionInpaint"/> (the 9-channel
    /// inpainting UNets: the mask, then the masked image's latents). Absent: the denoiser has no extra channels.
    /// </summary>
    public const string ConditionMode = "conditionMode";

    /// <summary><see cref="ConditionMode"/> of InstructPix2Pix: the extra channels are the input image's (unscaled) latents.</summary>
    public const string ConditionImage = "image";

    /// <summary><see cref="ConditionMode"/> of the inpainting UNets: the extra channels are the latent-resolution mask and the masked image's latents.</summary>
    public const string ConditionInpaint = "inpaint";

    /// <summary>
    /// Model: default image guidance scale of <see cref="ConditionImage"/> models (InstructPix2Pix, default 1.5): how strongly the output
    /// follows the input image, next to <see cref="GuidanceScale"/> for the instruction.
    /// </summary>
    public const string ImageGuidanceScale = "imageGuidanceScale";

    /// <summary>VAE: the latent scaling factor (model latents are the VAE's latents times it); needed to feed unscaled latents to a <see cref="ConditionImage"/> denoiser.</summary>
    public const string ScalingFactor = "scalingFactor";

    /// <summary>
    /// Tokenizer: the text that stands for each reference image in the prompt of an editing model whose language model reads the images
    /// (Qwen-Image-Edit), with <c>{n}</c> for the image's number; the language model's image placeholder inside it is expanded to the image's
    /// rows. The text goes between the template's start and the prompt.
    /// </summary>
    public const string ImagePrompt = "imagePrompt";

    /// <summary>
    /// Text encoder: pixels of the area a reference image is resized to (keeping its aspect ratio) for the language model's vision tower
    /// (Qwen-Image-Edit-2509 / -2511: 384 · 384; the original Qwen-Image-Edit: 1024 · 1024); 0 or absent: the text encoder reads no images.
    /// </summary>
    public const string ConditionPixels = "conditionPixels";

    /// <summary>
    /// Model: pixels of the area the output is generated at when a request gives no size, the aspect ratio that of the last reference image
    /// (Qwen-Image-Edit: 1024 · 1024; sides rounded to multiples of 32); 0 or absent: the first reference image's own size.
    /// </summary>
    public const string ReferenceArea = "referenceArea";
}
