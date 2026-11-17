namespace Sharplight.Studio.Services.Updates;

/// <summary>
/// Checks for, downloads and applies new versions of Studio. The real implementation wraps Velopack; the view models only see this
/// interface, so tests run them against a fake.
/// </summary>
internal interface IUpdateService
{
    /// <summary>Whether this run can update itself (installed with Velopack, update URL configured).</summary>
    public UpdateSupport Support { get; }

    /// <summary>The running version.</summary>
    public string CurrentVersion { get; }

    /// <summary>The update channel this install follows, e.g. <c>win-x64-stable</c>; null when not installed.</summary>
    public string? Channel { get; }

    /// <summary>The update feed in use (base URL); null when none is configured.</summary>
    public string? UpdateUrl { get; }

    /// <summary>
    /// Asks the update feed of <paramref name="channel"/> (null: <see cref="Channel"/>) for a different version. An offer older than
    /// the running version has <see cref="UpdateOffer.IsDowngrade"/> set; a channel without releases answers
    /// <see cref="UpdateCheckResult.NoReleaseYet"/>.
    /// </summary>
    public Task<UpdateCheckResult> CheckAsync(string? channel, CancellationToken cancellationToken);

    /// <summary>Downloads <paramref name="offer"/> (full or delta packages), reporting percent complete.</summary>
    public Task DownloadAsync(UpdateOffer offer, IProgress<int> progress, CancellationToken cancellationToken);

    /// <summary>Quits, installs the downloaded <paramref name="offer"/> and starts the new version.</summary>
    public void ApplyAndRestart(UpdateOffer offer);
}
