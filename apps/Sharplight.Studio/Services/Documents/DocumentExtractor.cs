using System.Text;
using System.Text.RegularExpressions;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;
using UglyToad.PdfPig;
using UglyToad.PdfPig.DocumentLayoutAnalysis.TextExtractor;

namespace Sharplight.Studio.Services.Documents;

/// <summary>
/// <see cref="IDocumentExtractor"/> with PdfPig (PDF, Apache-2.0), the Open XML SDK (DOCX, MIT) and plain reads for text,
/// Markdown and source code. HTML is reduced to its text.
/// </summary>
internal sealed partial class DocumentExtractor : IDocumentExtractor
{
    /// <summary>Largest text file read (bigger files would not fit any context anyway).</summary>
    private const long MaxTextBytes = 16L << 20;

    private static readonly HashSet<string> s_textExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".log",
        ".cs", ".fs", ".vb", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cc", ".m", ".swift",
        ".rb", ".php", ".pl", ".lua", ".r", ".jl", ".scala", ".sh", ".bash", ".ps1", ".bat", ".sql", ".css", ".scss", ".slang", ".hlsl", ".glsl",
        ".axaml", ".xaml", ".csproj", ".props", ".targets", ".sln", ".slnx", ".gradle", ".cmake", ".dockerfile", ".tex",
    };

    /// <summary>Extensions offered in the file picker.</summary>
    public static IReadOnlyList<string> SupportedExtensions { get; } = [".pdf", ".docx", ".html", ".htm", .. s_textExtensions];

    /// <inheritdoc/>
    public bool CanExtract(string path)
    {
        var extension = Path.GetExtension(path);
        return extension is ".pdf" or ".docx" or ".html" or ".htm" || s_textExtensions.Contains(extension)
            || Path.GetFileName(path) is "Dockerfile" or "Makefile" or "README" or "LICENSE";
    }

    /// <inheritdoc/>
    public Task<ExtractedDocument> ExtractAsync(string path, CancellationToken cancellationToken)
    {
        if (!CanExtract(path))
        {
            throw new NotSupportedException($"'{Path.GetFileName(path)}' is not a supported document type.");
        }

        return Task.Run(() => Extract(path), cancellationToken);
    }

    private static ExtractedDocument Extract(string path)
    {
        var name = Path.GetFileName(path);
        return Path.GetExtension(path).ToLowerInvariant() switch
        {
            ".pdf" => ExtractPdf(path, name),
            ".docx" => new ExtractedDocument(name, "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ExtractDocx(path), null),
            ".html" or ".htm" => new ExtractedDocument(name, "text/html", HtmlToText(ReadText(path)), null),
            ".md" or ".markdown" => new ExtractedDocument(name, "text/markdown", ReadText(path), null),
            _ => new ExtractedDocument(name, "text/plain", ReadText(path), null),
        };
    }

    private static ExtractedDocument ExtractPdf(string path, string name)
    {
        using var document = PdfDocument.Open(path);
        var text = new StringBuilder();
        foreach (var page in document.GetPages())
        {
            text.Append("--- page ").Append(page.Number).AppendLine(" ---");
            text.AppendLine(ContentOrderTextExtractor.GetText(page).Trim());
        }

        return new ExtractedDocument(name, "application/pdf", text.ToString().Trim(), document.NumberOfPages);
    }

    private static string ExtractDocx(string path)
    {
        using var document = WordprocessingDocument.Open(path, isEditable: false);
        var body = document.MainDocumentPart?.Document?.Body;
        if (body is null)
        {
            return "";
        }

        var text = new StringBuilder();
        foreach (var element in body.ChildElements)
        {
            switch (element)
            {
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

        return text.ToString().Trim();
    }

    private static string ReadText(string path)
    {
        if (new FileInfo(path).Length > MaxTextBytes)
        {
            throw new NotSupportedException($"'{Path.GetFileName(path)}' is larger than {MaxTextBytes >> 20} MB.");
        }

        return File.ReadAllText(path);
    }

    private static string HtmlToText(string html)
    {
        var withoutScripts = ScriptOrStyle().Replace(html, " ");
        var withBreaks = BlockTag().Replace(withoutScripts, "\n");
        var text = System.Net.WebUtility.HtmlDecode(AnyTag().Replace(withBreaks, ""));
        return BlankLines().Replace(text, "\n\n").Trim();
    }

    [GeneratedRegex(@"<(script|style)[^>]*>.*?</\1>", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex ScriptOrStyle();

    [GeneratedRegex(@"<(br|/p|/div|/h[1-6]|/li|/tr)[^>]*>", RegexOptions.IgnoreCase)]
    private static partial Regex BlockTag();

    [GeneratedRegex(@"<[^>]+>")]
    private static partial Regex AnyTag();

    [GeneratedRegex(@"\n\s*\n\s*\n+")]
    private static partial Regex BlankLines();
}
