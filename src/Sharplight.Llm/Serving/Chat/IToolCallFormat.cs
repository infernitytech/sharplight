namespace Sharplight.Llm.Serving.Chat;

/// <summary>
/// How a model family writes tool calls into its output. The output parser watches for <see cref="StartMarker"/>, collects text up to
/// <see cref="EndMarker"/> (or the end of the output) and hands the block to <see cref="TryParse"/>. Implement this to add a format.
/// </summary>
public interface IToolCallFormat
{
    public string Name { get; }
    /// <summary>Text that opens a tool-call block (null when the format has none).</summary>
    public string? StartMarker { get; }
    /// <summary>Text that closes the block; null when the block runs to the end of the output.</summary>
    public string? EndMarker { get; }
    /// <summary>The call may also appear as bare JSON at the very beginning of the output (Llama 3).</summary>
    public bool AllowLeadingJson { get; }
    /// <summary>Parses the block text (without markers). False when it is not a valid call.</summary>
    public bool TryParse(string block, out IReadOnlyList<ParsedCall> calls);
}
