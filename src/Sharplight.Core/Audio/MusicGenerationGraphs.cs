namespace Sharplight.Audio;

/// <summary>
/// The graph contract of the <see cref="Models.ModelKinds.MusicGeneration"/> kind (MusicGen): a text encoder component (<see cref="Models.ModelComponentRoles.TextEncoder"/>,
/// graph <see cref="TextGraph"/>: tokens in, the conditioning rows out), a tokenizer, a decoder component (<see cref="Models.ModelComponentRoles.Decoder"/>) with the
/// cached one-step graphs <see cref="StepGraph"/> (conditioned) and <see cref="UnconditionalStepGraph"/> (the unconditional branch of classifier-free guidance), and a
/// codec decoder component (<see cref="Models.ModelComponentRoles.Vocoder"/>) with <see cref="AudioCodecGraphs.DecodeGraph"/>. A step takes one code per codebook
/// (<see cref="CodeInput"/><c>(q)</c> i32 <c>[1]</c>) and returns <see cref="LogitsOutput"/> f32 <c>[codebooks, vocabulary]</c>.
/// </summary>
public static class MusicGenerationGraphs
{
    /// <summary>Text graph: <see cref="TokensInput"/> i32 <c>[T]</c> in, <see cref="ConditionOutput"/> f32 <c>[T, width]</c> out.</summary>
    public const string TextGraph = "text_encode";

    /// <summary>The conditioned step graph (cross-attention over <see cref="ConditionInput"/>).</summary>
    public const string StepGraph = "decode";

    /// <summary>The unconditioned step graph.</summary>
    public const string UnconditionalStepGraph = "decode_unconditional";

    /// <summary>The token input of the text graph.</summary>
    public const string TokensInput = "tokens";

    /// <summary>The conditioning output of the text graph.</summary>
    public const string ConditionOutput = "condition";

    /// <summary>The conditioning input of the conditioned step graph, f32 <c>[E, width]</c> over the encoder symbol.</summary>
    public const string ConditionInput = "condition";

    /// <summary>The logits output of the step graphs.</summary>
    public const string LogitsOutput = "logits";

    /// <summary>The name of the step graphs' code input of codebook <paramref name="q"/>.</summary>
    /// <param name="q">The codebook.</param>
    public static string CodeInput(int q) => $"code_{q}";

    /// <summary>The decoder's configuration key of the number of codebooks.</summary>
    public const string CodebooksKey = "codebooks";

    /// <summary>The decoder's configuration key of the number of codes of a codebook (the start and padding code is that value).</summary>
    public const string VocabularyKey = "vocabulary";

    /// <summary>The decoder's configuration key of the most steps.</summary>
    public const string MaxStepsKey = "maxSteps";

    /// <summary>The decoder's configuration key of the default guidance scale.</summary>
    public const string GuidanceKey = "guidance";

    /// <summary>The decoder's configuration key of the sample rate of the generated audio.</summary>
    public const string SampleRateKey = "sampleRate";

    /// <summary>The decoder's configuration key of the code frames per second.</summary>
    public const string FrameRateKey = "frameRate";

    /// <summary>The text encoder's configuration key of the most tokens.</summary>
    public const string MaxTokensKey = "maxTokens";

    /// <summary>The text encoder's configuration key of the width of the conditioning rows.</summary>
    public const string ConditionWidthKey = "conditionWidth";
}
