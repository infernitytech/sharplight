using System.Collections.Concurrent;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using Sharplight.Hub;
using Sharplight.ModelFeed;
using Sharplight.ModelFeed.Client;
using Sharplight.ModelFeed.Resolution;
using Sharplight.Studio.Services.Settings;

namespace Sharplight.Studio.Services.Feed;

/// <summary>
/// <see cref="IFeedService"/> over <see cref="FeedClient"/>: the address and channel come from the settings, the index, key set and manifests are
/// verified against the trusted root keys, listings fall back to the last verified copy on disk, and installs download into the model store with
/// <see cref="PackageDownloader"/> (resumable, verified per chunk and as a whole) and replace an installed version only after the new one verified.
/// All network and file work runs on the caller's task (the view models start it off the UI thread).
/// </summary>
internal sealed class FeedService : IFeedService, IDisposable
{
    private const string RepositoryOwner = "sharplight";
    private const string PendingSuffix = ".pending";

    private readonly ISettingsService _settings;
    private readonly FeedOptions _options;
    private readonly HttpClient _http;
    private readonly FileFeedStateStore _state;
    private readonly DependencyResolver _resolver = new();
    private readonly PackageDownloader _downloader = new();
    private readonly ConcurrentDictionary<string, SemaphoreSlim> _locks = new(StringComparer.Ordinal);

    public FeedService(ISettingsService settings, FeedOptions options)
    {
        _settings = settings;
        _options = options;
        _http = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
        _http.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("SharplightStudio", "1"));
        _state = new FileFeedStateStore(options.StateFile);
    }

    /// <summary>The store record's repository of a feed package.</summary>
    public static string RepositoryOf(string id) => $"{RepositoryOwner}/{id}";

    /// <inheritdoc/>
    public string Url => string.IsNullOrWhiteSpace(_settings.Current.FeedUrl) ? FeedOptions.DefaultUrl : _settings.Current.FeedUrl.Trim();

    /// <inheritdoc/>
    public string Channel => string.Equals(_settings.Current.FeedChannel, FeedOptions.Unstable, StringComparison.OrdinalIgnoreCase) ? FeedOptions.Unstable : FeedOptions.Stable;

    /// <inheritdoc/>
    public CompatibilityEnvironment Environment => _options.Environment;

    /// <inheritdoc/>
    public async Task<FeedListing> GetListingAsync(CancellationToken cancellationToken)
    {
        string url = Url;
        string channel = Channel;
        FeedDocumentCache cache;
        CachingFeedTransport online;
        try
        {
            cache = new FeedDocumentCache(CacheDirectoryFor(url, channel));
            online = new CachingFeedTransport(CreateTransport(url), cache);
        }
        catch (ArgumentException error)
        {
            return Failed(FeedFailureKind.Format, $"The feed address '{url}' is not valid: {error.Message}");
        }

        try
        {
            var snapshot = await CreateClient(online, channel, allowExpired: false).GetSnapshotAsync(cancellationToken).ConfigureAwait(false);
            online.Commit();
            return new FeedListing { Packages = snapshot.Packages, Verified = snapshot.Verified, Updated = snapshot.Index.Updated, Warnings = snapshot.Warnings };
        }
        catch (FeedVerificationException error)
        {
            return Failed(FeedFailureKind.Verification, $"The feed failed verification and was not used: {error.Message}");
        }
        catch (FeedFormatException error)
        {
            return Failed(FeedFailureKind.Format, $"The feed could not be read: {error.Message}");
        }
        catch (Exception error) when (error is IOException or HttpRequestException or TaskCanceledException && !cancellationToken.IsCancellationRequested)
        {
            return await ListCachedAsync(url, channel, cache, error.Message, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <inheritdoc/>
    public async Task<FeedInstallResult> InstallAsync(string id, IProgress<FeedInstallProgress>? progress, CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        string url = Url;
        string channel = Channel;
        var transport = CreateTransport(url);
        var client = CreateClient(transport, channel, allowExpired: false);
        progress?.Report(new FeedInstallProgress(FeedInstallPhase.Resolving, null, 0, 0));
        var snapshot = await client.GetSnapshotAsync(cancellationToken).ConfigureAwait(false);
        if (snapshot.Find(id) is null)
        {
            throw new FeedInstallException($"The {channel} channel of the feed has no package '{id}'.");
        }

        var resolution = await _resolver.ResolveAsync([new PackageRequest(id)], FeedCandidates.From(client, snapshot),
            new ResolutionOptions { Environment = _options.Environment }, cancellationToken).ConfigureAwait(false);
        if (!resolution.Succeeded)
        {
            throw new FeedInstallException($"{id} cannot be installed: {string.Join("; ", resolution.Problems)}");
        }

        var store = new ModelStore(_settings.StoreRoot);
        var plan = new List<(ResolvedPackage Package, VerifiedManifest Manifest, ModelManifest? Existing)>();
        foreach (var package in resolution.Packages.OrderBy(package => package.Requested ? 1 : 0))
        {
            var manifest = await client.GetManifestAsync(snapshot, package.Id, package.Version, cancellationToken).ConfigureAwait(false);
            plan.Add((package, manifest, InstalledAt(store, package.Id, package.Version)));
        }

        long total = plan.Where(item => item.Existing is null).Sum(item => item.Manifest.Manifest.PrimaryFile?.Size ?? 0);
        long done = 0;
        var installed = new List<ModelManifest>();
        ModelManifest? requested = null;
        foreach (var (package, verified, existing) in plan)
        {
            var record = existing;
            if (record is null)
            {
                long before = done;
                record = await InstallOneAsync(store, transport, verified, channel, received =>
                    progress?.Report(new FeedInstallProgress(FeedInstallPhase.Downloading, package.Id, before + received, total)), cancellationToken).ConfigureAwait(false);
                done += verified.Manifest.PrimaryFile?.Size ?? 0;
                installed.Add(record);
                progress?.Report(new FeedInstallProgress(FeedInstallPhase.Installing, package.Id, done, total));
            }

            if (package.Requested)
            {
                requested = record;
            }
        }

        progress?.Report(new FeedInstallProgress(FeedInstallPhase.Done, id, total, total));
        return new FeedInstallResult(installed, requested!);
    }

    /// <summary>Releases the HTTP client.</summary>
    public void Dispose() => _http.Dispose();

    private static ModelManifest? InstalledAt(ModelStore store, string id, string version)
    {
        var existing = store.List().FirstOrDefault(model => string.Equals(model.FeedPackage, id, StringComparison.Ordinal));
        return existing is not null && string.Equals(existing.FeedVersion, version, StringComparison.Ordinal) && File.Exists(existing.PackagePath) ? existing : null;
    }

    private async Task<ModelManifest> InstallOneAsync(ModelStore store, IFeedTransport transport, VerifiedManifest verified, string channel, Action<long> progress,
        CancellationToken cancellationToken)
    {
        var manifest = verified.Manifest;
        var gate = _locks.GetOrAdd(manifest.Id, _ => new SemaphoreSlim(1, 1));
        await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var directory = store.GetModelDirectory(RepositoryOf(manifest.Id), "");
            Directory.CreateDirectory(directory);
            var target = Path.Combine(directory, ModelStore.PackageFileName);
            var pending = target + PendingSuffix;
            await _downloader.DownloadAsync(manifest, [transport], pending, new Runtime.SynchronousProgress<long>(progress), cancellationToken).ConfigureAwait(false);
            // The old package stays in place until the new one has verified; the rename replaces it in one step.
            File.Move(pending, target, overwrite: true);
            var record = new ModelManifest
            {
                Repository = RepositoryOf(manifest.Id),
                Alias = manifest.Id,
                Commit = verified.Digest,
                Format = ModelFormat.Slm,
                ModelName = manifest.Display?.Name ?? manifest.Id,
                Architecture = manifest.Model?.Architecture,
                PackageBytes = manifest.PrimaryFile?.Size ?? new FileInfo(target).Length,
                PulledAt = DateTimeOffset.UtcNow,
                FeedPackage = manifest.Id,
                FeedVersion = manifest.Version,
                FeedChannel = channel,
                Directory = directory,
            };
            store.Save(record);
            return record;
        }
        finally
        {
            gate.Release();
        }
    }

    private async Task<FeedListing> ListCachedAsync(string url, string channel, FeedDocumentCache cache, string reason, CancellationToken cancellationToken)
    {
        try
        {
            var client = CreateClient(new CachedFeedTransport(url, cache), channel, allowExpired: true);
            var snapshot = await client.GetSnapshotAsync(cancellationToken).ConfigureAwait(false);
            return new FeedListing { Packages = snapshot.Packages, Verified = snapshot.Verified, FromCache = true, Updated = snapshot.Index.Updated, Warnings = snapshot.Warnings };
        }
        catch (Exception error) when (error is FeedFormatException or FeedVerificationException or IOException)
        {
            return Failed(FeedFailureKind.Offline, $"The feed could not be reached ({reason}) and there is no verified copy from an earlier visit.");
        }
    }

    private static FeedListing Failed(FeedFailureKind kind, string message) => new() { Failure = new FeedFailure(kind, message) };

    private IFeedTransport CreateTransport(string url) => _options.TransportFactory?.Invoke(url) ?? FeedTransports.Create(url, _http);

    private FeedClient CreateClient(IFeedTransport transport, string channel, bool allowExpired) => new(
        transport,
        new FeedClientOptions { Channel = channel, TrustedRoots = [.. _options.TrustedRoots], AllowExpiredIndex = allowExpired },
        _state,
        _options.Time);

    private string CacheDirectoryFor(string url, string channel)
    {
        var hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes($"{url.TrimEnd('/')}#{channel}")))[..16].ToLowerInvariant();
        return Path.Combine(_options.CacheDirectory, hash);
    }
}
