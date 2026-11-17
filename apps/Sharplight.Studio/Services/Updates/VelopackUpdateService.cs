using System.Reflection;
using Velopack;
using Velopack.Locators;
using Velopack.Sources;

namespace Sharplight.Studio.Services.Updates;

/// <summary>
/// Update checks and installs through Velopack, against the feed at <see cref="UpdateEndpoint"/>. The channel is the one the
/// installed package was built for (<c>vpk pack --channel</c>) unless the user picked another one, which is passed to Velopack as the
/// explicit channel (moving to an older release is allowed; the view model asks before it installs one).
/// </summary>
internal sealed class VelopackUpdateService : IUpdateService
{
    private readonly IVelopackLocator? _locator;
    private UpdateManager? _manager;
    private UpdateInfo? _pending;

    /// <summary>Creates the service; <paramref name="updateUrl"/> is the resolved feed URL (see <see cref="UpdateEndpoint.Resolve"/>).</summary>
    public VelopackUpdateService(string? updateUrl)
    {
        UpdateUrl = updateUrl;
        CurrentVersion = typeof(VelopackUpdateService).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()
            ?.InformationalVersion.Split('+')[0] ?? "dev";
        var locator = TryGetLocator();
        if (locator?.CurrentlyInstalledVersion is not { } installed)
        {
            Support = UpdateSupport.Unsupported("Updates are unavailable for this build: it was not installed with the Studio installer.");
        }
        else
        {
            CurrentVersion = installed.ToString();
            Channel = locator.Channel;
            if (updateUrl is null)
            {
                Support = UpdateSupport.Unsupported(
                    $"Update checks are disabled: no update URL is configured. Set {UpdateEndpoint.EnvironmentVariable} or the UpdateUrl setting.");
            }
            else
            {
                _locator = locator;
                Support = UpdateSupport.Supported;
            }
        }
    }

    /// <inheritdoc/>
    public UpdateSupport Support { get; }

    /// <inheritdoc/>
    public string CurrentVersion { get; }

    /// <inheritdoc/>
    public string? Channel { get; }

    /// <inheritdoc/>
    public string? UpdateUrl { get; }

    /// <inheritdoc/>
    public async Task<UpdateCheckResult> CheckAsync(string? channel, CancellationToken cancellationToken)
    {
        if (_locator is null || UpdateUrl is null)
        {
            return UpdateCheckResult.Failed(Support.Reason ?? "Updates are unavailable.");
        }

        try
        {
            var options = new UpdateOptions { ExplicitChannel = channel ?? Channel, AllowVersionDowngrade = true };
            var manager = new UpdateManager(new SimpleWebSource(UpdateUrl), options, _locator);
            var info = await manager.CheckForUpdatesAsync().WaitAsync(cancellationToken);
            _manager = manager;
            if (info is null)
            {
                _pending = null;
                return UpdateCheckResult.UpToDate;
            }

            _pending = info;
            var release = info.TargetFullRelease;
            return UpdateCheckResult.Found(new UpdateOffer(release.Version.ToString(), release.NotesMarkdown, release.Size, info.IsDowngrade));
        }
        catch (Exception error) when (error is not OperationCanceledException)
        {
            _pending = null;
            return IsMissingFeed(error) ? UpdateCheckResult.NoReleaseYet : UpdateCheckResult.Failed(error.Message);
        }
    }

    /// <inheritdoc/>
    public async Task DownloadAsync(UpdateOffer offer, IProgress<int> progress, CancellationToken cancellationToken)
    {
        var info = Pending(offer);
        await _manager!.DownloadUpdatesAsync(info, progress.Report, cancellationToken);
    }

    /// <inheritdoc/>
    public void ApplyAndRestart(UpdateOffer offer) => _manager!.ApplyUpdatesAndRestart(Pending(offer).TargetFullRelease);

    private UpdateInfo Pending(UpdateOffer offer) => _manager is not null && _pending is { } info && info.TargetFullRelease.Version.ToString() == offer.Version
        ? info
        : throw new InvalidOperationException($"Version {offer.Version} is not the update that was found; check for updates again.");

    private static bool IsMissingFeed(Exception error)
    {
        for (var current = error; current is not null; current = current.InnerException)
        {
            if (current is HttpRequestException { StatusCode: System.Net.HttpStatusCode.NotFound }
                || current.Message.Contains("404", StringComparison.Ordinal)
                || current.Message.Contains("Not Found", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    private static IVelopackLocator? TryGetLocator()
    {
        try
        {
            return VelopackLocator.Current;
        }
        catch (Exception error) when (error is InvalidOperationException or PlatformNotSupportedException or NotSupportedException)
        {
            return null;
        }
    }
}
