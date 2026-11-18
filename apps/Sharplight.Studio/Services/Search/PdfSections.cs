using UglyToad.PdfPig;
using UglyToad.PdfPig.DocumentLayoutAnalysis.TextExtractor;

namespace Sharplight.Studio.Services.Search;

/// <summary>Reads a PDF page by page.</summary>
internal static class PdfSections
{
    /// <summary>One section per page that has text.</summary>
    /// <param name="path">The PDF.</param>
    /// <param name="limits">The page limit.</param>
    /// <param name="cancellationToken">Cancels reading.</param>
    public static IReadOnlyList<DocumentSection> Read(string path, DocumentLimits limits, CancellationToken cancellationToken)
    {
        using var document = PdfDocument.Open(path);
        if (document.NumberOfPages > limits.MaxPages)
        {
            throw new InvalidDataException($"The PDF has {document.NumberOfPages} pages (limit {limits.MaxPages}).");
        }

        var sections = new List<DocumentSection>();
        foreach (var page in document.GetPages())
        {
            cancellationToken.ThrowIfCancellationRequested();
            string text = ContentOrderTextExtractor.GetText(page).Trim();
            if (text.Length > 0)
            {
                sections.Add(new DocumentSection(text, "", page.Number));
            }
        }

        return sections;
    }
}
