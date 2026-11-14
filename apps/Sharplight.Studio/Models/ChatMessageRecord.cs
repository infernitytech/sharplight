namespace Sharplight.Studio.Models;

/// <summary>
/// One message of a conversation. Messages form a tree through <see cref="ParentId"/>: editing a message or regenerating an
/// answer adds a sibling, so earlier versions stay reachable as branches.
/// </summary>
internal sealed record ChatMessageRecord
{
    /// <summary>Unique id.</summary>
    public string Id { get; init; } = Guid.NewGuid().ToString("N");

    /// <summary>The message this one answers or follows; null for the first message.</summary>
    public string? ParentId { get; init; }

    /// <summary>Author.</summary>
    public ChatRole Role { get; init; }

    /// <summary>Text (Markdown for answers).</summary>
    public string Content { get; init; } = "";

    /// <summary>The model's reasoning before the answer, if it produced any.</summary>
    public string? Reasoning { get; init; }

    /// <summary>Images and documents (user messages).</summary>
    public IReadOnlyList<AttachmentRecord> Attachments { get; init; } = [];

    /// <summary>Generation statistics (answers).</summary>
    public MessageStats? Stats { get; init; }

    /// <summary>The model that wrote an answer.</summary>
    public string? ModelId { get; init; }

    /// <summary>Failure message when generation failed.</summary>
    public string? Error { get; init; }

    /// <summary>When the message was created.</summary>
    public DateTimeOffset CreatedAt { get; init; } = DateTimeOffset.UtcNow;
}
