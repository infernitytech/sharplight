using System.Text;

namespace Sharplight.Studio.Services.Search;

/// <summary>Cuts Markdown into the sections under its headings (lines inside code fences are never headings).</summary>
internal static class MarkdownSections
{
    /// <summary>The sections of <paramref name="markdown"/>, each with its heading and first line.</summary>
    /// <param name="markdown">The document.</param>
    public static IReadOnlyList<DocumentSection> Read(string markdown)
    {
        var sections = new List<DocumentSection>();
        var body = new StringBuilder();
        string heading = "";
        int start = 1;
        bool fenced = false;
        string[] lines = markdown.ReplaceLineEndings("\n").Split('\n');
        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i];
            if (line.TrimStart().StartsWith("```", StringComparison.Ordinal) || line.TrimStart().StartsWith("~~~", StringComparison.Ordinal))
            {
                fenced = !fenced;
            }

            if (!fenced && HeadingOf(line) is { } title)
            {
                Flush(sections, body, heading, start);
                heading = title;
                start = i + 1;
            }

            body.Append(line).Append('\n');
        }

        Flush(sections, body, heading, start);
        return sections;
    }

    private static void Flush(List<DocumentSection> sections, StringBuilder body, string heading, int start)
    {
        string text = body.ToString().TrimEnd('\n');
        body.Clear();
        if (text.Trim().Length > 0)
        {
            sections.Add(new DocumentSection(text, heading, 0, start));
        }
    }

    private static string? HeadingOf(string line)
    {
        int level = 0;
        while (level < line.Length && level < 7 && line[level] == '#')
        {
            level++;
        }

        if (level is 0 or > 6 || level >= line.Length || line[level] != ' ')
        {
            return null;
        }

        return line[(level + 1)..].Trim().TrimEnd('#').Trim();
    }
}
