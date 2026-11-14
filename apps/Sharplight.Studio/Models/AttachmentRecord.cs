namespace Sharplight.Studio.Models;

/// <summary>A file attached to a user message.</summary>
internal sealed record AttachmentRecord
{
    /// <summary>Unique id.</summary>
    public string Id { get; init; } = Guid.NewGuid().ToString("N");

    /// <summary>Image or document.</summary>
    public AttachmentKind Kind { get; init; }

    /// <summary>File name shown in the chat.</summary>
    public required string Name { get; init; }

    /// <summary>MIME type (<c>image/png</c>, <c>application/pdf</c>, ...).</summary>
    public string MediaType { get; init; } = "application/octet-stream";

    /// <summary>Encoded image bytes (images only).</summary>
    public byte[]? Data { get; init; }

    /// <summary>Extracted text as inserted into the prompt (documents only).</summary>
    public string? Text { get; init; }

    /// <summary>Tokens the text occupies in the prompt (documents only).</summary>
    public int TokenCount { get; init; }

    /// <summary>The text was shortened to fit the context window.</summary>
    public bool Truncated { get; init; }
}
