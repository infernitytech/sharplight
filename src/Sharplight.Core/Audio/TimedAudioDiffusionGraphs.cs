namespace Sharplight.Audio;

/// <summary>
/// The graph contract of the timing-conditioned latent diffusion models of the <see cref="Models.ModelKinds.AudioGeneration"/> kind (Stable Audio Open): a tokenizer, a
/// text encoder component (<see cref="Models.ModelComponentRoles.TextEncoder"/>, <see cref="TextGraph"/>), a timing conditioner component
/// (<see cref="Models.ModelComponentRoles.Encoder"/>, <see cref="TimingGraph"/>: the start and the total length of the audio as conditioning rows), a denoiser component
/// (<see cref="Models.ModelComponentRoles.Denoiser"/>, <see cref="DenoiseGraph"/>: one evaluation of a v-objective network on preconditioned latents) and an autoencoder
/// component (<see cref="Models.ModelComponentRoles.Vae"/>, <see cref="AudioGenerationGraphs.DecodeGraph"/> with the inputs and outputs of
/// <see cref="AudioGenerationGraphs"/>). The client runs the sampler (k-diffusion's DPM-Solver++(2M) SDE over an exponential sigma schedule with classifier-free
/// guidance) and decodes the latents.
/// </summary>
/// <remarks>
/// All signals are time-major. <see cref="TextGraph"/>: <see cref="TokensInput"/> i32 <c>[N]</c> and <see cref="MaskInput"/> f32 <c>[N]</c> (1 for prompt tokens, 0 for
/// padding; N the fixed prompt length <see cref="PromptLengthKey"/>) to <see cref="HiddenOutput"/> f32 <c>[N, width]</c> with the padding rows zero.
/// <see cref="TimingGraph"/>: <see cref="SecondsInput"/> f32 <c>[2]</c> (start and total in seconds) to <see cref="TimingOutput"/> f32 <c>[2, width]</c>.
/// <see cref="DenoiseGraph"/>: <see cref="LatentInput"/> f32 <c>[T, C]</c> (the latents times c_in), <see cref="TimestepInput"/> f32 <c>[1]</c> (the network's time
/// <c>atan(σ) · 2 / π</c>), <see cref="ConditionInput"/> f32 <c>[E, width]</c> (the prompt rows followed by the timing rows; zeros for the unconditional pass),
/// <see cref="GlobalInput"/> f32 <c>[1, 2 · width]</c> (the timing rows side by side) and <see cref="PositionsInput"/> i32 <c>[T + 1, 1]</c> to <see cref="PredictionOutput"/>
/// f32 <c>[T, C]</c> (the v prediction).
/// </remarks>
public static class TimedAudioDiffusionGraphs
{
    /// <summary>The text encoder graph.</summary>
    public const string TextGraph = "text_encode";

    /// <summary>The timing conditioner graph.</summary>
    public const string TimingGraph = "timing_encode";

    /// <summary>The denoiser graph.</summary>
    public const string DenoiseGraph = "denoise";

    /// <summary>Token ids of the text graph.</summary>
    public const string TokensInput = "tokens";

    /// <summary>The prompt mask of the text graph: f32 <c>[N]</c>, 1 for prompt tokens, 0 for padding.</summary>
    public const string MaskInput = "mask";

    /// <summary>The text graph's rows.</summary>
    public const string HiddenOutput = "hidden";

    /// <summary>The start and the total length of the audio in seconds (timing graph input).</summary>
    public const string SecondsInput = "seconds";

    /// <summary>The timing rows (timing graph output).</summary>
    public const string TimingOutput = "timing";

    /// <summary>The preconditioned latents (denoiser input).</summary>
    public const string LatentInput = "latent";

    /// <summary>The network's time (denoiser input).</summary>
    public const string TimestepInput = "timestep";

    /// <summary>The cross-attention rows (denoiser input).</summary>
    public const string ConditionInput = "condition";

    /// <summary>The global conditioning vector (denoiser input).</summary>
    public const string GlobalInput = "global";

    /// <summary>Row positions of the denoiser's sequence (the global row and the latent frames).</summary>
    public const string PositionsInput = "positions";

    /// <summary>The denoiser's v prediction.</summary>
    public const string PredictionOutput = "prediction";

    // Component configuration keys.

    /// <summary>Text encoder: the fixed prompt length in tokens (longer prompts are truncated).</summary>
    public const string PromptLengthKey = "promptLength";

    /// <summary>Text encoder and timing conditioner: the width of their rows.</summary>
    public const string WidthKey = "width";

    /// <summary>Timing conditioner: the largest number of seconds it encodes.</summary>
    public const string MaxSecondsKey = "maxSeconds";

    /// <summary>Denoiser: the latent frames the network generates (the audio is cut to the requested duration).</summary>
    public const string FramesKey = "frames";

    /// <summary>Denoiser: the smallest sigma of the schedule.</summary>
    public const string SigmaMinKey = "sigmaMin";

    /// <summary>Denoiser: the largest sigma of the schedule.</summary>
    public const string SigmaMaxKey = "sigmaMax";

    /// <summary>Denoiser: the sigma of the data.</summary>
    public const string SigmaDataKey = "sigmaData";

    /// <summary>Denoiser: the default classifier-free guidance scale.</summary>
    public const string DefaultGuidanceKey = "defaultGuidance";
}
