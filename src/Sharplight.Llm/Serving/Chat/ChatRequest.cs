namespace Sharplight.Llm.Serving.Chat;

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
