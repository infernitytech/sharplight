namespace Sharplight.Hub;

/// <summary>The outcome of <see cref="ModelPuller.PullAsync"/>.</summary>
/// <param name="Manifest">The stored model.</param>
/// <param name="UpToDate">The model was already stored at the resolved commit; nothing was downloaded or converted.</param>
/// <param name="DownloadedBytes">Bytes transferred by this pull (excluding resumed and already complete files).</param>
/// <param name="DownloadTime">Time spent downloading.</param>
/// <param name="ConvertTime">Time spent converting and verifying the package.</param>
/// <param name="PrecompileTime">Time spent precompiling (zero when skipped).</param>
public sealed record PullResult(
    ModelManifest Manifest,
    bool UpToDate,
    long DownloadedBytes,
    TimeSpan DownloadTime,
    TimeSpan ConvertTime,
    TimeSpan PrecompileTime);
