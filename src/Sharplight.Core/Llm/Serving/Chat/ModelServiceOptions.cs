namespace Sharplight.Llm.Serving.Chat;

/// <summary>Settings of a <see cref="ModelService"/>: reported model name, defaults and chat-template / tool-format overrides.</summary>
public sealed record ModelServiceOptions
{
    /// <summary>Name reported by the APIs.</summary>
    public string ModelName { get; set; } = "sharplight";
    /// <summary>Completion length when a request gives none (0 = until the context is full).</summary>
    public int DefaultMaxTokens { get; set; } = 1024;
    /// <summary>Tool-call formats to parse; null = detected from the chat template.</summary>
    public IReadOnlyList<IToolCallFormat>? ToolFormats { get; set; }
    /// <summary>Overrides the model's own chat template (Jinja source).</summary>
    public string? ChatTemplate { get; set; }
}
