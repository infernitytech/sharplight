namespace Sharplight.Studio.Services.Maintenance;

/// <summary>A failed, cancelled or abandoned download left in the model store: a partial file or a model folder without a manifest.</summary>
/// <param name="Path">The file or folder.</param>
/// <param name="Bytes">Its size.</param>
/// <param name="IsDirectory">A whole folder (an interrupted pull) rather than one file.</param>
internal sealed record PartialDownload(string Path, long Bytes, bool IsDirectory);
