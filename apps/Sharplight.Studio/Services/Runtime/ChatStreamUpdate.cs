using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>One streamed event of an answer.</summary>
/// <param name="Kind">What it carries.</param>
/// <param name="Text">Content, reasoning or tool-call text.</param>
/// <param name="Stats">Statistics (final event).</param>
/// <param name="Error">Failure message (final event).</param>
internal sealed record ChatStreamUpdate(ChatStreamKind Kind, string Text = "", MessageStats? Stats = null, string? Error = null)
{
    /// <summary>A piece of the answer.</summary>
    public static ChatStreamUpdate Content(string text) => new(ChatStreamKind.Content, text);

    /// <summary>A piece of reasoning.</summary>
    public static ChatStreamUpdate Reasoning(string text) => new(ChatStreamKind.Reasoning, text);

    /// <summary>The final event.</summary>
    public static ChatStreamUpdate Finished(MessageStats stats, string? error = null) => new(ChatStreamKind.Finished, "", stats, error);
}
