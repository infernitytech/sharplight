namespace Sharplight.Studio.Services.Search;

/// <summary>
/// <see cref="IDocumentParser"/> for the formats the Search page indexes: Markdown (sections under headings), HTML (text only, sections
/// under headings), PDF (pages; PdfPig, Apache-2.0), DOCX (sections under headings), PPTX (slides) and XLSX (sheets) through the Open XML
/// SDK (MIT), and plain text and source code (lines). Size and unpacking limits come from <see cref="DocumentLimits"/>.
/// </summary>
internal sealed class StructuredDocumentParser(DocumentLimits? limits = null) : IDocumentParser
{
    private static readonly HashSet<string> s_textExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".txt", ".rst", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".log",
        ".cs", ".fs", ".vb", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cc", ".m", ".swift",
        ".rb", ".php", ".pl", ".lua", ".r", ".jl", ".scala", ".sh", ".bash", ".ps1", ".bat", ".sql", ".css", ".scss", ".slang", ".hlsl", ".glsl",
        ".axaml", ".xaml", ".csproj", ".props", ".targets", ".sln", ".slnx", ".gradle", ".cmake", ".dockerfile", ".tex",
    };

    private readonly DocumentLimits _limits = limits ?? new DocumentLimits();

    /// <inheritdoc/>
    public bool CanParse(string path)
    {
        string extension = Path.GetExtension(path);
        return extension.ToLowerInvariant() is ".pdf" or ".docx" or ".pptx" or ".xlsx" or ".html" or ".htm" or ".md" or ".markdown"
            || s_textExtensions.Contains(extension) || Path.GetFileName(path) is "Dockerfile" or "Makefile" or "README" or "LICENSE";
    }

    /// <inheritdoc/>
    public long MaxBytes(string path) => IsPackage(path) ? _limits.MaxPackageBytes : _limits.MaxTextBytes;

    /// <inheritdoc/>
    public IReadOnlyList<DocumentSection> Parse(string path, CancellationToken cancellationToken)
    {
        long size = new FileInfo(path).Length;
        if (size > MaxBytes(path))
        {
            throw new InvalidDataException($"The file is larger than {MaxBytes(path) >> 20} MB.");
        }

        try
        {
            var sections = Dispatch(path, cancellationToken);
            return Capped(sections);
        }
        catch (Exception failure) when (failure is not (InvalidDataException or OperationCanceledException or IOException or UnauthorizedAccessException))
        {
            throw new InvalidDataException($"The file cannot be read: {failure.Message}", failure);
        }
    }

    private static bool IsPackage(string path) => Path.GetExtension(path).ToLowerInvariant() is ".pdf" or ".docx" or ".pptx" or ".xlsx";

    private IReadOnlyList<DocumentSection> Dispatch(string path, CancellationToken cancellationToken)
    {
        switch (Path.GetExtension(path).ToLowerInvariant())
        {
            case ".pdf":
                return PdfSections.Read(path, _limits, cancellationToken);
            case ".docx":
                ArchiveGuard.Check(path, _limits);
                return WordSections.Read(path, cancellationToken);
            case ".pptx":
                ArchiveGuard.Check(path, _limits);
                return SlideSections.Read(path, _limits, cancellationToken);
            case ".xlsx":
                ArchiveGuard.Check(path, _limits);
                return SheetSections.Read(path, _limits, cancellationToken);
            case ".html" or ".htm":
                return HtmlSections.Read(File.ReadAllText(path));
            case ".md" or ".markdown":
                return MarkdownSections.Read(File.ReadAllText(path));
            default:
                string text = File.ReadAllText(path);
                return text.Trim().Length == 0 ? [] : [new DocumentSection(text, "", 0, 1)];
        }
    }

    private List<DocumentSection> Capped(IReadOnlyList<DocumentSection> sections)
    {
        var kept = new List<DocumentSection>();
        long characters = 0;
        foreach (var section in sections)
        {
            if (section.Text.Trim().Length == 0)
            {
                continue;
            }

            characters += section.Text.Length;
            if (characters > _limits.MaxCharacters)
            {
                break;
            }

            kept.Add(section);
        }

        return kept;
    }
}
