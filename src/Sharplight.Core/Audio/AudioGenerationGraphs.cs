namespace Sharplight.Audio;

/// <summary>
/// The graph contract of the <see cref="Models.ModelKinds.AudioGeneration"/> kind (a latent-diffusion audio model whose conditioning has a text encoder, an optional
/// lyric path and a condition encoder; ACE-Step): a tokenizer, a text encoder component (<see cref="Models.ModelComponentRoles.TextEncoder"/>) with
/// <see cref="TextGraph"/> (the caption's rows, causal) and <see cref="LyricGraph"/> (the lyric tokens' embeddings), a condition encoder component
/// (<see cref="Models.ModelComponentRoles.Encoder"/>, <see cref="ConditionGraph"/>) that packs lyrics, timbre reference and caption into the rows the denoiser attends
/// to, a denoiser component (<see cref="Models.ModelComponentRoles.Denoiser"/>, <see cref="DenoiseGraph"/>: one evaluation, the flow velocity) and an autoencoder
/// component (<see cref="Models.ModelComponentRoles.Vae"/>, <see cref="DecodeGraph"/>) that turns latents into audio.
/// </summary>
/// <remarks>
/// All signals are time-major. <see cref="TextGraph"/> / <see cref="LyricGraph"/>: <see cref="TokensInput"/> i32 <c>[T]</c> (and <see cref="PositionsInput"/> i32
/// <c>[T, 1]</c> for the text graph) to <see cref="HiddenOutput"/> f32 <c>[T, width]</c>. <see cref="ConditionGraph"/>: <see cref="TextInput"/> f32 <c>[S, width]</c>
/// (symbol <c>S</c>), <see cref="LyricInput"/> f32 <c>[L, width]</c> (symbol <c>L</c>), <see cref="ReferenceInput"/> f32 <c>[R, latent]</c> (the timbre reference, the
/// silence latent for no reference), the positions of the three (<see cref="LyricPositionsInput"/>, <see cref="ReferencePositionsInput"/>) to
/// <see cref="ConditionOutput"/> f32 <c>[L + 1 + S, D]</c>. <see cref="DenoiseGraph"/>: <see cref="LatentInput"/> f32 <c>[T, C]</c>, <see cref="ContextInput"/> f32
/// <c>[T, K]</c>, <see cref="TimestepInput"/> and <see cref="TimestepRInput"/> f32 <c>[1]</c>, <see cref="ConditionInput"/> f32 <c>[E, D]</c> and
/// <see cref="PositionsInput"/> i32 <c>[T / patch, 1]</c> to <see cref="VelocityOutput"/> f32 <c>[T, C]</c> (T a multiple of the patch size). <see cref="DecodeGraph"/>:
/// <see cref="LatentInput"/> f32 <c>[T, C]</c> to <see cref="AudioOutput"/> f32 <c>[T · samples per frame, channels]</c>.
/// </remarks>
public static class AudioGenerationGraphs
{
    /// <summary>The caption encoder graph.</summary>
    public const string TextGraph = "text_encode";

    /// <summary>The lyric embedding graph.</summary>
    public const string LyricGraph = "lyric_embed";

    /// <summary>The condition encoder graph.</summary>
    public const string ConditionGraph = "condition_encode";

    /// <summary>The denoiser graph.</summary>
    public const string DenoiseGraph = "denoise";

    /// <summary>The autoencoder decoder graph.</summary>
    public const string DecodeGraph = "audio_decode";

    /// <summary>Token ids of the text and lyric graphs.</summary>
    public const string TokensInput = "tokens";

    /// <summary>Row positions (0, 1, 2, ...) of a graph with rotary attention: i32 <c>[rows, 1]</c>.</summary>
    public const string PositionsInput = "positions";

    /// <summary>Row positions of the lyric rows of the condition graph.</summary>
    public const string LyricPositionsInput = "lyric_positions";

    /// <summary>Row positions of the reference rows of the condition graph.</summary>
    public const string ReferencePositionsInput = "reference_positions";

    /// <summary>The caption rows (condition graph input).</summary>
    public const string TextInput = "text";

    /// <summary>The lyric embeddings (condition graph input).</summary>
    public const string LyricInput = "lyric";

    /// <summary>The timbre reference latents (condition graph input).</summary>
    public const string ReferenceInput = "reference";

    /// <summary>The rows of the text and lyric graphs.</summary>
    public const string HiddenOutput = "hidden";

    /// <summary>The packed conditioning rows (condition graph output).</summary>
    public const string ConditionOutput = "condition";

    /// <summary>The conditioning rows (denoiser input).</summary>
    public const string ConditionInput = "condition";

    /// <summary>The noisy latents (denoiser and decoder input).</summary>
    public const string LatentInput = "latent";

    /// <summary>The source latents and chunk mask concatenated per frame (denoiser input).</summary>
    public const string ContextInput = "context";

    /// <summary>The time of the flow, 1 at noise and 0 at the data (denoiser input, f32 <c>[1]</c>).</summary>
    public const string TimestepInput = "timestep";

    /// <summary>The second time of the mean-flow denoiser (equal to <see cref="TimestepInput"/> for a plain flow).</summary>
    public const string TimestepRInput = "timestep_r";

    /// <summary>The denoiser's flow velocity.</summary>
    public const string VelocityOutput = "velocity";

    /// <summary>The decoder's audio.</summary>
    public const string AudioOutput = "audio";

    /// <summary>The symbol of the lyric rows.</summary>
    public const string LyricSymbol = "L";

    /// <summary>The symbol of the caption rows.</summary>
    public const string TextSymbol = "S";

    /// <summary>The attachment of the denoiser with the silence latent, f32 <c>[frames, channels]</c> (the source of audio without reference and the timbre reference).</summary>
    public const string SilenceAttachment = "silence-latent";

    // Component configuration keys.

    /// <summary>Denoiser: the sample rate of the decoded audio.</summary>
    public const string SampleRateKey = "sampleRate";

    /// <summary>Denoiser: the output channels of the decoder.</summary>
    public const string ChannelsKey = "channels";

    /// <summary>Denoiser: the audio samples per latent frame.</summary>
    public const string SamplesPerFrameKey = "samplesPerFrame";

    /// <summary>Denoiser: the values per latent frame.</summary>
    public const string LatentChannelsKey = "latentChannels";

    /// <summary>Denoiser: the values of the context per frame.</summary>
    public const string ContextChannelsKey = "contextChannels";

    /// <summary>Denoiser: the frames covered by the silence latent attachment.</summary>
    public const string SilenceFramesKey = "silenceFrames";

    /// <summary>Denoiser: the value of the chunk mask channels of a generation without source audio.</summary>
    public const string ChunkMaskValueKey = "chunkMaskValue";

    /// <summary>Denoiser: the patch size (the latent frames are padded to a multiple of it).</summary>
    public const string PatchSizeKey = "patchSize";

    /// <summary>Denoiser: the frames of the longest audio.</summary>
    public const string MaxFramesKey = "maxFrames";

    /// <summary>Denoiser: the default number of steps.</summary>
    public const string DefaultStepsKey = "defaultSteps";

    /// <summary>Denoiser: the default timestep shift.</summary>
    public const string DefaultShiftKey = "defaultShift";

    /// <summary>Denoiser: the frames of the timbre reference of audio without reference.</summary>
    public const string ReferenceFramesKey = "referenceFrames";

    /// <summary>Text encoder: the most caption tokens.</summary>
    public const string MaxTextTokensKey = "maxTextTokens";

    /// <summary>Text encoder: the most lyric tokens.</summary>
    public const string MaxLyricTokensKey = "maxLyricTokens";

    /// <summary>Text encoder: the width of the rows of the text and lyric graphs.</summary>
    public const string TextWidthKey = "textWidth";

    /// <summary>Condition encoder: the width of the condition rows.</summary>
    public const string ConditionWidthKey = "conditionWidth";

    /// <summary>Text encoder: the instruction of the caption prompt.</summary>
    public const string InstructionKey = "instruction";

    /// <summary>Autoencoder: the frames one decoder run covers (longer audio is decoded in overlapping windows).</summary>
    public const string DecodeFramesKey = "decodeFrames";

    /// <summary>Autoencoder: the frames of context on either side of a window.</summary>
    public const string DecodeOverlapKey = "decodeOverlap";
}
