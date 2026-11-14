namespace Sharplight.Studio.Models;

/// <summary>A persisted chat: its settings and the tree of messages.</summary>
internal sealed record Conversation
{
    /// <summary>Unique id.</summary>
    public string Id { get; init; } = Guid.NewGuid().ToString("N");

    /// <summary>Title shown in the list (derived from the first message until renamed).</summary>
    public string Title { get; init; } = "New chat";

    /// <summary>Creation time.</summary>
    public DateTimeOffset CreatedAt { get; init; } = DateTimeOffset.UtcNow;

    /// <summary>Time of the last change.</summary>
    public DateTimeOffset UpdatedAt { get; init; } = DateTimeOffset.UtcNow;

    /// <summary>The model last used in this chat.</summary>
    public string? ModelId { get; init; }

    /// <summary>Instructions sent as the system message.</summary>
    public string SystemPrompt { get; init; } = "";

    /// <summary>Sampling settings of this chat.</summary>
    public SamplingSettings Sampling { get; init; } = new();

    /// <summary>Let reasoning models think first.</summary>
    public bool EnableThinking { get; init; } = true;

    /// <summary>The last message of the branch being shown; null for an empty chat.</summary>
    public string? CurrentLeafId { get; init; }

    /// <summary>Every message of every branch.</summary>
    public IReadOnlyList<ChatMessageRecord> Messages { get; init; } = [];
}
