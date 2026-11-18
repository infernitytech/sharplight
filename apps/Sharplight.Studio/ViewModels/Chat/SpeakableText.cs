using System.Text.RegularExpressions;

namespace Sharplight.Studio.ViewModels.Chat;

/// <summary>Turns an answer written in Markdown into the text a speech model should read: no code blocks, link addresses, markup characters or tables.</summary>
internal static partial class SpeakableText
{
    /// <summary>The words to speak for <paramref name="markdown"/>; empty when nothing speakable is left.</summary>
    /// <param name="markdown">An answer as shown in the chat.</param>
    public static string From(string markdown)
    {
        var text = FencedCode().Replace(markdown, " ");
        text = Image().Replace(text, "$1");
        text = Link().Replace(text, "$1");
        text = InlineCode().Replace(text, "$1");
        text = TableRule().Replace(text, "");
        text = LinePrefix().Replace(text, "");
        text = text.Replace("|", ", ", StringComparison.Ordinal);
        text = Emphasis().Replace(text, "$2");
        text = Whitespace().Replace(text, " ");
        return text.Trim();
    }

    [GeneratedRegex(@"```.*?(```|\z)", RegexOptions.Singleline)]
    private static partial Regex FencedCode();

    [GeneratedRegex(@"!\[([^\]]*)\]\([^)]*\)")]
    private static partial Regex Image();

    [GeneratedRegex(@"\[([^\]]+)\]\([^)]*\)")]
    private static partial Regex Link();

    [GeneratedRegex(@"`([^`]*)`")]
    private static partial Regex InlineCode();

    [GeneratedRegex(@"^\s*\|?[\s:|-]{3,}\|?\s*$", RegexOptions.Multiline)]
    private static partial Regex TableRule();

    [GeneratedRegex(@"^\s*(#{1,6}\s+|>\s?|[-*+]\s+|\d+[.)]\s+)", RegexOptions.Multiline)]
    private static partial Regex LinePrefix();

    [GeneratedRegex(@"(\*{1,3}|_{1,3})(\S(?:.*?\S)?)\1")]
    private static partial Regex Emphasis();

    [GeneratedRegex(@"\s+")]
    private static partial Regex Whitespace();
}
