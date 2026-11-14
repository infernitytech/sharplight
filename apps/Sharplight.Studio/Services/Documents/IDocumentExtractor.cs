namespace Sharplight.Studio.Services.Documents;

/// <summary>Extracts plain text from documents (PDF, DOCX, Markdown, plain text, source code, HTML).</summary>
internal interface IDocumentExtractor
{
    /// <summary>True when the file's extension is a supported document type.</summary>
    public bool CanExtract(string path);

    /// <summary>Extracts the text.</summary>
    /// <exception cref="NotSupportedException">The file type is not supported.</exception>
    public Task<ExtractedDocument> ExtractAsync(string path, CancellationToken cancellationToken);
}
