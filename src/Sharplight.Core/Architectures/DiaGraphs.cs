namespace Sharplight.Architectures;

/// <summary>The graph contract of Dia (<see cref="DiaAdapter"/>): the text encoder, the cached decoder step and the DAC codec graphs.</summary>
internal static class DiaGraphs
{
    /// <summary>The text encoder graph: <see cref="TokensInput"/> and <see cref="PositionsInput"/> in, <see cref="ConditionOutput"/> [T, width] out.</summary>
    public const string TextGraph = "dia_text";

    /// <summary>The decoder step graph (any number of rows; <see cref="CodeInput"/> per codebook).</summary>
    public const string StepGraph = "dia_step";

    /// <summary>The text token input of the text graph, i32 [T].</summary>
    public const string TokensInput = "tokens";

    /// <summary>The positions input of the text graph and the step graph, i32 [T, 1].</summary>
    public const string PositionsInput = "positions";

    /// <summary>The output of the text graph and the encoder input of the step graph.</summary>
    public const string ConditionOutput = "condition";

    /// <summary>The span input of the step graph, i32 [T, 2]: the rows of the cached source each row attends to.</summary>
    public const string SpansInput = "source_spans";

    /// <summary>The logits output of the step graph, f32 [T, channels · vocabulary].</summary>
    public const string LogitsOutput = "logits";

    /// <summary>The name of the step graph's input of codebook <paramref name="channel"/> (ids already offset by <c>channel · vocabulary</c>).</summary>
    /// <param name="channel">The codebook.</param>
    public static string CodeInput(int channel) => $"code_{channel}";

    /// <summary>The role of the text encoder component.</summary>
    public const string TextRole = "text-encoder";

    /// <summary>The role of the decoder component.</summary>
    public const string DecoderRole = "decoder";

    /// <summary>The role of the codec encoder component (audio prompts).</summary>
    public const string CodecEncoderRole = "codec-encoder";

    /// <summary>The role of the codec decoder component.</summary>
    public const string CodecDecoderRole = "codec-decoder";

    /// <summary>The prefix of the codec's weights in the model.</summary>
    public const string CodecPrefix = "codec.";
}
