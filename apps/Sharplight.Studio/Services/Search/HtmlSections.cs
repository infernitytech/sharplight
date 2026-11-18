using System.Text;
using System.Text.RegularExpressions;

namespace Sharplight.Studio.Services.Search;

/// <summary>Reduces HTML to its text and cuts it into the sections under its headings (scripts, styles and markup are dropped).</summary>
internal static partial class HtmlSections
{
    /// <summary>The sections of <paramref name="html"/>.</summary>
    /// <param name="html">The document.</param>
    public static IReadOnlyList<DocumentSection> Read(string html)
    {
        string clean = Comments().Replace(ScriptOrStyle().Replace(html, " "), " ");
        var sections = new List<DocumentSection>();
        string heading = "";
        int last = 0;
        foreach (Match match in Heading().Matches(clean))
        {
            Add(sections, clean[last..match.Index], heading);
            heading = ToText(match.Groups[2].Value).Replace('\n', ' ').Trim();
            last = match.Index;
        }

        Add(sections, clean[last..], heading);
        return sections;
    }

    /// <summary>The visible text of <paramref name="html"/>.</summary>
    /// <param name="html">A piece of HTML.</param>
    public static string ToText(string html)
    {
        string breaks = BlockTag().Replace(html, "\n");
        var builder = new StringBuilder(System.Net.WebUtility.HtmlDecode(AnyTag().Replace(breaks, "")));
        return BlankLines().Replace(builder.ToString(), "\n\n").Trim();
    }

    private static void Add(List<DocumentSection> sections, string part, string heading)
    {
        string text = ToText(part);
        if (text.Length > 0)
        {
            sections.Add(new DocumentSection(text, heading));
        }
    }

    [GeneratedRegex(@"<(script|style|head)\b[^>]*>.*?</\1>", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex ScriptOrStyle();

    [GeneratedRegex(@"<!--.*?-->", RegexOptions.Singleline)]
    private static partial Regex Comments();

    [GeneratedRegex(@"<h([1-6])[^>]*>(.*?)</h\1>", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex Heading();

    [GeneratedRegex(@"<(br|/p|/div|/h[1-6]|/li|/tr|/section|/article)[^>]*>", RegexOptions.IgnoreCase)]
    private static partial Regex BlockTag();

    [GeneratedRegex(@"<[^>]+>")]
    private static partial Regex AnyTag();

    [GeneratedRegex(@"[ \t]*\n\s*\n\s*\n+")]
    private static partial Regex BlankLines();
}
