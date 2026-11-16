namespace Sharplight.Audio;

/// <summary>
/// The graph contract of the <see cref="Models.ModelKinds.AudioCodec"/> kind: an encoder component (<see cref="Models.ModelComponentRoles.Encoder"/>) with the graph
/// <see cref="EncodeGraph"/> and a decoder component (<see cref="Models.ModelComponentRoles.Decoder"/>) with <see cref="DecodeGraph"/>, both over the row symbol
/// <c>T</c> (samples for the encoder, frames for the decoder). Which codec produced the graphs is irrelevant to the client.
/// </summary>
public static class AudioCodecGraphs
{
    /// <summary>Encoder graph: input <see cref="AudioInput"/> f32 <c>[T, 1]</c> (mono samples, a multiple of the samples per frame), outputs <see cref="CodesName"/><c>(q)</c> i32 <c>[T / samples per frame]</c> per codebook.</summary>
    public const string EncodeGraph = "audio_encode";

    /// <summary>Decoder graph: inputs <see cref="CodesName"/><c>(q)</c> i32 <c>[T]</c> per codebook and <see cref="MaskInput"/> f32 <c>[codebooks]</c> (1 for a codebook in use, 0 for one left out), output <see cref="AudioOutput"/> f32 <c>[T · samples per frame, 1]</c>.</summary>
    public const string DecodeGraph = "audio_decode";

    /// <summary>The audio input of the encoder and the audio output of the decoder.</summary>
    public const string AudioInput = "audio";

    /// <summary>The name of the decoder's audio output.</summary>
    public const string AudioOutput = "audio";

    /// <summary>The decoder's input of the codebook mask.</summary>
    public const string MaskInput = "codebook_mask";

    /// <summary>The configuration key of the sample rate.</summary>
    public const string SampleRateKey = "sampleRate";

    /// <summary>The configuration key of the samples per frame.</summary>
    public const string SamplesPerFrameKey = "samplesPerFrame";

    /// <summary>The configuration key of the codebook count.</summary>
    public const string CodebooksKey = "codebooks";

    /// <summary>The configuration key of the default number of codebooks.</summary>
    public const string DefaultCodebooksKey = "defaultCodebooks";

    /// <summary>The configuration key of the codebook size.</summary>
    public const string CodebookSizeKey = "codebookSize";

    /// <summary>
    /// The optional input of a graph with a rotary transformer: i32 <c>[rows, 1]</c> with the values 0, 1, 2, ...; the component configuration says how many rows a run has
    /// (<c>T · <see cref="PositionsNumeratorKey"/> / <see cref="PositionsDenominatorKey"/></c>).
    /// </summary>
    public const string PositionsInput = "positions";

    /// <summary>The optional input of a graph with causal attention: i32 <c>[rows, 2]</c>, every row <c>(0, rows)</c> (one sequence); the row count is that of <see cref="PositionsInput"/>.</summary>
    public const string SpansInput = "spans";

    /// <summary>The configuration key of the numerator of the row count of <see cref="PositionsInput"/> (default 1).</summary>
    public const string PositionsNumeratorKey = "positionsNumerator";

    /// <summary>The configuration key of the denominator of the row count of <see cref="PositionsInput"/> (default 1).</summary>
    public const string PositionsDenominatorKey = "positionsDenominator";

    /// <summary>The name of the codes input or output of codebook <paramref name="codebook"/>.</summary>
    /// <param name="codebook">The codebook.</param>
    public static string CodesName(int codebook) => $"codes.{codebook}";
}
