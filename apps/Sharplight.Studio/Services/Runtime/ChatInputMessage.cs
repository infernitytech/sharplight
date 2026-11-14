using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>One message as sent to a model: the final text (documents already inlined) and images.</summary>
/// <param name="Role">Author.</param>
/// <param name="Text">Text content.</param>
/// <param name="Images">Images, in order.</param>
internal sealed record ChatInputMessage(ChatRole Role, string Text, IReadOnlyList<ImageContent> Images)
{
    /// <summary>A text-only message.</summary>
    public static ChatInputMessage FromText(ChatRole role, string text) => new(role, text, []);
}
