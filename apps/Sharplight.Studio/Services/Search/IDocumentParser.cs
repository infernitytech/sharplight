namespace Sharplight.Studio.Services.Search;

/// <summary>Reads a document into its structural sections for indexing.</summary>
internal interface IDocumentParser
{
    /// <summary>True when the file's type is one the parser reads.</summary>
    /// <param name="path">The file.</param>
    public bool CanParse(string path);

    /// <summary>The largest size in bytes the parser reads for this file's type.</summary>
    /// <param name="path">The file.</param>
    public long MaxBytes(string path);

    /// <summary>Reads the file.</summary>
    /// <param name="path">The file.</param>
    /// <param name="cancellationToken">Cancels reading.</param>
    /// <exception cref="InvalidDataException">The file is damaged, too large or unpacks to too much.</exception>
    public IReadOnlyList<DocumentSection> Parse(string path, CancellationToken cancellationToken);
}
