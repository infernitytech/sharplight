namespace Sharplight.Studio.Services.Documents;

/// <summary>Text extracted from a document.</summary>
/// <param name="Name">File name.</param>
/// <param name="MediaType">MIME type.</param>
/// <param name="Text">The extracted text.</param>
/// <param name="Pages">Page count for paged formats, else null.</param>
internal sealed record ExtractedDocument(string Name, string MediaType, string Text, int? Pages);
