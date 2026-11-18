using System.Text;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;

namespace Sharplight.Studio.Services.Search;

/// <summary>Reads a DOCX into the sections under its heading paragraphs (styles Title and Heading n).</summary>
internal static class WordSections
{
    /// <summary>The sections of the document body; tables become rows of <c>| cell | cell |</c>.</summary>
    /// <param name="path">The DOCX.</param>
    /// <param name="cancellationToken">Cancels reading.</param>
    public static IReadOnlyList<DocumentSection> Read(string path, CancellationToken cancellationToken)
    {
        using var document = WordprocessingDocument.Open(path, isEditable: false);
        var sections = new List<DocumentSection>();
        if (document.MainDocumentPart?.Document?.Body is not { } body)
        {
            return sections;
        }

        var text = new StringBuilder();
        string heading = "";
        foreach (var element in body.ChildElements)
        {
            cancellationToken.ThrowIfCancellationRequested();
            switch (element)
            {
                case Paragraph paragraph when IsHeading(paragraph) && paragraph.InnerText.Trim().Length > 0:
                    Flush(sections, text, heading);
                    heading = paragraph.InnerText.Trim();
                    text.AppendLine(heading);
                    break;
                case Paragraph paragraph:
                    text.AppendLine(paragraph.InnerText);
                    break;
                case Table table:
                    foreach (var row in table.Elements<TableRow>())
                    {
                        text.AppendLine("| " + string.Join(" | ", row.Elements<TableCell>().Select(cell => cell.InnerText.Trim())) + " |");
                    }

                    text.AppendLine();
                    break;
            }
        }

        Flush(sections, text, heading);
        return sections;
    }

    private static bool IsHeading(Paragraph paragraph)
    {
        string? style = paragraph.ParagraphProperties?.ParagraphStyleId?.Val?.Value;
        return style is not null && (style.StartsWith("Heading", StringComparison.OrdinalIgnoreCase) || style.Equals("Title", StringComparison.OrdinalIgnoreCase));
    }

    private static void Flush(List<DocumentSection> sections, StringBuilder text, string heading)
    {
        string body = text.ToString().Trim();
        text.Clear();
        if (body.Length > 0)
        {
            sections.Add(new DocumentSection(body, heading));
        }
    }
}
