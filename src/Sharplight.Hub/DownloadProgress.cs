namespace Sharplight.Hub;

/// <summary>Progress of a download: the current file and the totals of the whole operation (one file or a whole pull).</summary>
/// <param name="File">Name of the file being downloaded.</param>
/// <param name="FileReceived">Bytes of the file on disk (including bytes of an earlier, resumed attempt).</param>
/// <param name="FileSize">Size of the file.</param>
/// <param name="Received">Bytes of all files of the operation on disk.</param>
/// <param name="Total">Size of all files of the operation.</param>
/// <param name="Resumed">Bytes that were already on disk when the operation started (not downloaded now; exclude them from speed estimates).</param>
public readonly record struct DownloadProgress(string File, long FileReceived, long FileSize, long Received, long Total, long Resumed);
