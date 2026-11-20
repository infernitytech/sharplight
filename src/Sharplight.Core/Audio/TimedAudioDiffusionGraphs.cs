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
/// <para>
/// Rectified-flow form (<see cref="ObjectiveKey"/> <c>rectified_flow</c> or <c>rf_denoiser</c> on the denoiser: Stable Audio 3). The latent length follows the requested
/// duration (the symbol <see cref="Models.WellKnown.TokenSymbol"/> = T frames, a multiple of <see cref="FrameAlignKey"/>, the duration plus <see cref="HeadroomSecondsKey"/>,
/// at most <see cref="FramesKey"/>). <see cref="TextGraph"/> as above, but its padding rows are the model's learned padding row. <see cref="TimingGraph"/>:
/// <see cref="SecondsInput"/> f32 <c>[1]</c> (the total seconds) to <see cref="TimingOutput"/> f32 <c>[1, width]</c>. <see cref="DenoiseGraph"/>: <see cref="LatentInput"/>
/// f32 <c>[T, C]</c> (x at time t), <see cref="TimestepInput"/> f32 <c>[1]</c> (t; the normalized log-SNR <c>(5 − clamp(log((1 − t) / t), −12, 5)) / 17</c> with
/// <see cref="TimeLogSnrKey"/>), <see cref="ConditionInput"/> f32 <c>[N + 1, width]</c> (the prompt rows and the seconds row; zeros for the unconditional pass without a
/// negative prompt), <see cref="GlobalInput"/> f32 <c>[1, width]</c> (the seconds row), <see cref="LocalInput"/> f32 <c>[T, L]</c> (the local additive conditioning,
/// zeros without inpainting), <see cref="PositionsInput"/> i32 <c>[T + M, 1]</c> and <see cref="ValidInput"/> f32 <c>[T + M]</c> (rows: the T frames followed by the
/// M = <see cref="MemoryTokensKey"/> memory rows, at positions M … M + T − 1 and 0 … M − 1; valid 1 for the memory rows and the frames before the padding) to
/// <see cref="PredictionOutput"/> f32 <c>[T + M, C]</c> whose first T rows are the velocity. <see cref="AudioGenerationGraphs.DecodeGraph"/> has a fixed window of
/// <see cref="AudioGenerationGraphs.DecodeFramesKey"/> frames: <see cref="AudioGenerationGraphs.LatentInput"/> f32 <c>[F, C]</c> and <see cref="NoiseInput"/> f32
/// <c>[F · <see cref="NoiseRowsKey"/>, <see cref="NoiseWidthKey"/>]</c> (standard normal noise the decoder scales by its own amount; zeros decode deterministically) to
/// <see cref="AudioGenerationGraphs.AudioOutput"/>; the client adds <see cref="LatentNoiseKey"/> times standard normal noise to the latents before.
/// </para>
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

    /// <summary>The denoiser's v prediction (the velocity in the rectified-flow form).</summary>
    public const string PredictionOutput = "prediction";

    /// <summary>The local additive conditioning of the rectified-flow denoiser (the inpainting mask and the masked latents).</summary>
    public const string LocalInput = "local";

    /// <summary>The rectified-flow denoiser's valid rows (1) and padding frames (0).</summary>
    public const string ValidInput = "valid";

    /// <summary>The rectified-flow decoder's noise.</summary>
    public const string NoiseInput = "noise";

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

    /// <summary>Denoiser: the training objective, <c>v</c> (default), <c>rectified_flow</c> (Euler steps with guidance) or <c>rf_denoiser</c> (ping-pong steps).</summary>
    public const string ObjectiveKey = "objective";

    /// <summary>Rectified-flow denoiser: the memory rows after the frames.</summary>
    public const string MemoryTokensKey = "memoryTokens";

    /// <summary>Rectified-flow denoiser: the latent frames are a multiple of this.</summary>
    public const string FrameAlignKey = "frameAlign";

    /// <summary>Rectified-flow denoiser: the seconds generated after the requested duration (they stay valid rows).</summary>
    public const string HeadroomSecondsKey = "headroomSeconds";

    /// <summary>Rectified-flow denoiser: the network's time is the normalized log-SNR of t.</summary>
    public const string TimeLogSnrKey = "timeLogSnr";

    /// <summary>Rectified-flow denoiser: the width of the local additive conditioning.</summary>
    public const string LocalChannelsKey = "localChannels";

    /// <summary>
    /// Rectified-flow denoiser: the shift of the linear time schedule (<c>logsnr</c>: log-SNR spacing from <see cref="AnchorLogSnrKey"/> to <see cref="LogSnrEndKey"/>,
    /// moved by <see cref="ShiftRateKey"/> per doubling of the frames over <see cref="AnchorLengthKey"/>; <c>flux</c>: <c>α t / (1 + (α − 1) t)</c> with α interpolated
    /// log-linearly between <see cref="AlphaMinKey"/> and <see cref="AlphaMaxKey"/> over <see cref="MinLengthKey"/> … <see cref="MaxLengthKey"/>; <c>full</c>: the
    /// exponential shift from <see cref="BaseShiftKey"/> to <see cref="MaxShiftKey"/>; <c>none</c>).
    /// </summary>
    public const string ScheduleKey = "schedule";

    /// <summary>Rectified-flow schedule: the log-SNR at t = 1 for the anchor length.</summary>
    public const string AnchorLogSnrKey = "anchorLogSnr";

    /// <summary>Rectified-flow schedule: the log-SNR at t = 0.</summary>
    public const string LogSnrEndKey = "logSnrEnd";

    /// <summary>Rectified-flow schedule: the log-SNR change per doubling of the frames.</summary>
    public const string ShiftRateKey = "shiftRate";

    /// <summary>Rectified-flow schedule: the frames of the anchor log-SNR.</summary>
    public const string AnchorLengthKey = "anchorLength";

    /// <summary>Rectified-flow schedule: the frames of the smallest shift.</summary>
    public const string MinLengthKey = "minLength";

    /// <summary>Rectified-flow schedule: the frames of the largest shift.</summary>
    public const string MaxLengthKey = "maxLength";

    /// <summary>Rectified-flow schedule (<c>flux</c>): α at the smallest length.</summary>
    public const string AlphaMinKey = "alphaMin";

    /// <summary>Rectified-flow schedule (<c>flux</c>): α at the largest length.</summary>
    public const string AlphaMaxKey = "alphaMax";

    /// <summary>Rectified-flow schedule (<c>full</c>): the shift at the smallest length.</summary>
    public const string BaseShiftKey = "baseShift";

    /// <summary>Rectified-flow schedule (<c>full</c>): the shift at the largest length.</summary>
    public const string MaxShiftKey = "maxShift";

    /// <summary>Rectified-flow decoder: the noise rows per latent frame.</summary>
    public const string NoiseRowsKey = "noiseRows";

    /// <summary>Rectified-flow decoder: the width of a noise row.</summary>
    public const string NoiseWidthKey = "noiseWidth";

    /// <summary>Rectified-flow decoder: the standard deviation of the noise the client adds to the latents (0 for none).</summary>
    public const string LatentNoiseKey = "latentNoise";
}
