namespace Sharplight.Llm.Serving.Chat;

/// <summary>Configuration of a <see cref="ChatOutputParser"/>: reasoning markers, tool-call formats and stop strings.</summary>
public sealed class ChatOutputParserOptions
{
    /// <summary>The default marker that opens a reasoning block.</summary>
    public const string DefaultReasoningStart = "<think>";

    /// <summary>The default marker that closes a reasoning block.</summary>
    public const string DefaultReasoningEnd = "</think>";

    /// <summary>Tool-call formats to recognise; empty disables tool-call parsing.</summary>
    public IReadOnlyList<IToolCallFormat> ToolCallFormats { get; init; } = [];

    /// <summary>Names of the offered tools: calls to other names stay text. Empty disables tool-call parsing.</summary>
    public IReadOnlyCollection<string> ToolNames { get; init; } = [];

    /// <summary>
    /// The prompt already opened a reasoning block, so the output starts with reasoning. Compute it with
    /// <see cref="ChatOutputParser.PromptEndsInReasoning"/>.
    /// </summary>
    public bool StartsInReasoning { get; init; }

    /// <summary>Marker that opens a reasoning block.</summary>
    public string ReasoningStart { get; init; } = DefaultReasoningStart;

    /// <summary>Marker that closes a reasoning block.</summary>
    public string ReasoningEnd { get; init; } = DefaultReasoningEnd;

    /// <summary>
    /// The output is OpenAI harmony (gpt-oss): analysis messages are reasoning, messages to <c>functions.*</c> are tool calls and
    /// the final message is content. Detect it with <see cref="ChatOutputParser.IsHarmonyTemplate"/>.
    /// </summary>
    public bool Harmony { get; init; }

    /// <summary>
    /// The output ends before the first occurrence of any of these (for generators that do not stop on strings themselves).
    /// Empty when the generator already handles stop strings.
    /// </summary>
    public IReadOnlyList<string> StopStrings { get; init; } = [];
}
