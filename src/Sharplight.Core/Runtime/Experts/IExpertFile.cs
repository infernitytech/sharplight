namespace Sharplight.Runtime.Experts;

/// <summary>A file opened by an <see cref="IExpertReader"/>.</summary>
public interface IExpertFile : IDisposable
{
    /// <summary>The path the file was opened with.</summary>
    public string Path { get; }

    /// <summary>Length of the file in bytes.</summary>
    public long Length { get; }

    /// <summary>
    /// The alignment, in bytes, of the offsets, lengths and host addresses of reads of this file: above 1 when the reader bypasses the page cache
    /// (O_DIRECT, unbuffered handles), else 1.
    /// </summary>
    public int Alignment { get; }
}
