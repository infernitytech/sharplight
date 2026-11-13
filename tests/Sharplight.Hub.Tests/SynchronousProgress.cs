namespace Sharplight.Hub.Tests;

/// <summary>An <see cref="IProgress{T}"/> that calls back immediately (unlike <see cref="Progress{T}"/>, which posts).</summary>
internal sealed class SynchronousProgress(Action<DownloadProgress> report) : IProgress<DownloadProgress>
{
    public void Report(DownloadProgress value) => report(value);
}
