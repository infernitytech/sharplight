namespace Sharplight.Llm.Serving.Chat;

/// <summary>An image attached to a message: encoded file bytes (PNG/JPEG/...) or a URL.</summary>
public sealed record ImageData(byte[]? Bytes, string? MediaType, string? Url);

/// <summary>A content part of a message: <c>text</c> or <c>image</c>.</summary>
public sealed record ChatPart(string Type, string? Text = null, ImageData? Image = null)
{
    public static ChatPart FromText(string text) => new("text", text);
    public static ChatPart FromImage(ImageData image) => new("image", null, image);
}

/// <summary>A tool/function call made by the model (or replayed in the history). Arguments are a JSON object text.</summary>
public sealed record ToolCall(string Id, string Name, string ArgumentsJson);

/// <summary>A function offered to the model; <see cref="ParametersJson"/> is its JSON-schema object.</summary>
public sealed record ToolSpec(string Name, string? Description, string? ParametersJson);

/// <summary>One message. Roles: system, user, assistant, tool. Give <see cref="Content"/> or <see cref="Parts"/>.</summary>
public sealed record ChatTurn(
    string Role,
    string? Content = null,
    IReadOnlyList<ChatPart>? Parts = null,
    string? Reasoning = null,
    IReadOnlyList<ToolCall>? ToolCalls = null,
    string? ToolCallId = null,
    string? Name = null);

public enum ToolChoiceMode { Auto, None, Required, Function }

public sealed class ChatRequest
{
    public required IReadOnlyList<ChatTurn> Messages { get; init; }
    public IReadOnlyList<ToolSpec> Tools { get; init; } = [];
    public ToolChoiceMode ToolChoice { get; init; }
    /// <summary>Function name for <see cref="ToolChoiceMode.Function"/>.</summary>
    public string? ToolChoiceFunction { get; init; }
    public SamplingParams Sampling { get; init; } = new();
    /// <summary>0 = as many as fit into the context.</summary>
    public int MaxTokens { get; init; }
    public IReadOnlyList<string> StopStrings { get; init; } = [];
    public IReadOnlyList<int> StopTokenIds { get; init; } = [];
    /// <summary>Number of independent completions (each is its own engine request sharing the prompt prefix).</summary>
    public int N { get; init; } = 1;
    public int Priority { get; init; }
    public TimeSpan? Timeout { get; init; }
    /// <summary>Passed to the chat template as <c>enable_thinking</c> when set.</summary>
    public bool? EnableThinking { get; init; }
    /// <summary>Best effort JSON mode: instructs the model to answer with a JSON object only.</summary>
    public bool JsonMode { get; init; }
    public bool Logprobs { get; init; }
    public int TopLogprobs { get; init; }
    /// <summary>Extra variables for the chat template (<c>chat_template_kwargs</c>).</summary>
    public IReadOnlyDictionary<string, object?>? TemplateContext { get; init; }
}

/// <summary>A plain text completion request (no chat template).</summary>
public sealed class CompletionRequest
{
    public string? Prompt { get; init; }
    public int[]? PromptTokens { get; init; }
    public SamplingParams Sampling { get; init; } = new();
    public int MaxTokens { get; init; }
    public IReadOnlyList<string> StopStrings { get; init; } = [];
    public IReadOnlyList<int> StopTokenIds { get; init; } = [];
    public int N { get; init; } = 1;
    public int Priority { get; init; }
    public TimeSpan? Timeout { get; init; }
    public bool Logprobs { get; init; }
    public int TopLogprobs { get; init; }
}

public enum ChatFinishReason { Stop, Length, ToolCalls, Cancelled, Error, Timeout }

/// <summary>
/// An event of a chat stream. Exactly one of <see cref="Content"/>, <see cref="Reasoning"/>, <see cref="ToolCall"/>, <see cref="Logprobs"/>
/// or (final) <see cref="Finish"/> is set per event.
/// </summary>
public sealed record ChatEvent(
    int Index,
    string? Content = null,
    string? Reasoning = null,
    ToolCall? ToolCall = null,
    IReadOnlyList<TokenLogprob>? Logprobs = null,
    ChatFinishReason? Finish = null,
    GenerationUsage? Usage = null,
    string? StopSequence = null,
    string? Error = null);

public sealed class ChatChoice
{
    public int Index { get; init; }
    public string Content { get; init; } = "";
    public string? Reasoning { get; init; }
    public IReadOnlyList<ToolCall> ToolCalls { get; init; } = [];
    public ChatFinishReason Finish { get; init; }
    public string? StopSequence { get; init; }
    public string? Error { get; init; }
    public IReadOnlyList<TokenLogprob> Logprobs { get; init; } = [];
    public GenerationUsage Usage { get; init; }
}

public sealed class ChatResult
{
    public IReadOnlyList<ChatChoice> Choices { get; init; } = [];
    /// <summary>Prompt tokens counted once, completion tokens summed over the choices.</summary>
    public GenerationUsage Usage { get; init; }
}
