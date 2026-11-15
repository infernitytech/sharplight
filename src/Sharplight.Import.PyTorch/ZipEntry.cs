namespace Sharplight.Import.PyTorch;

/// <summary>An uncompressed zip entry: where its bytes lie in the archive file.</summary>
/// <param name="Name">The entry name.</param>
/// <param name="DataOffset">Absolute offset of the entry data.</param>
/// <param name="Size">Size of the entry data in bytes.</param>
internal sealed record ZipEntry(string Name, long DataOffset, long Size);
