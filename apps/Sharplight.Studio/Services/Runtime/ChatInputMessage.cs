using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>One message as sent to a model: the final text (documents already inlined) and images.</summary>
/// <param name="Role">Author.</param>
/// <param name="Text">Text content.</param>
/// <param name="Images">Images, in order.</param>
/// <param name="Reasoning">
/// The reasoning of an earlier answer. Templates that replay earlier reasoning (Qwen3.6 and later keep it by default) render it;
/// without it they show the model an empty reasoning block for every earlier answer, which it never saw in training.
/// </param>
internal sealed record ChatInputMessage(ChatRole Role, string Text, IReadOnlyList<ImageContent> Images, string? Reasoning = null)
{
    /// <summary>A text-only message.</summary>
    public static ChatInputMessage FromText(ChatRole role, string text) => new(role, text, []);
}
