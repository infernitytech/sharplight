using Sharplight.ModelFeed.Index;
using Sharplight.ModelFeed.Signing;
using Sharplight.ModelFeed.Versioning;

namespace Sharplight.ModelFeed.Client;

/// <summary>
/// Reads and verifies a feed: key set against the trusted roots, index signature, expiry (freeze) and sequence (rollback), then the
/// digest chain index → catalog index → pages → package index → manifest, and each manifest's own signature. Thread safe.
/// </summary>
public sealed class FeedClient
{
    private readonly IFeedTransport _transport;
    private readonly FeedClientOptions _options;
    private readonly IFeedStateStore _state;
    private readonly TimeProvider _time;
    private readonly FeedTrust _trust;

    /// <summary>Creates a client.</summary>
    /// <param name="transport">Where the feed is.</param>
    /// <param name="options">Trust and limits.</param>
    /// <param name="state">Remembers accepted sequences (rollback protection); <see langword="null"/> for process memory only.</param>
    /// <param name="time">Clock; <see langword="null"/> for the system clock.</param>
    /// <exception cref="ArgumentException">Signatures are required but no root key is trusted.</exception>
    public FeedClient(IFeedTransport transport, FeedClientOptions options, IFeedStateStore? state = null, TimeProvider? time = null)
    {
        ArgumentNullException.ThrowIfNull(transport);
        ArgumentNullException.ThrowIfNull(options);
        if (options.Channel is not null && !FeedLayout.IsValidChannel(options.Channel))
        {
            throw new ArgumentException($"'{options.Channel}' is not a valid channel name.", nameof(options));
        }

        if (options.RequireSignature && options.TrustedRoots.Count == 0)
        {
            throw new ArgumentException("Signatures are required but no trusted root key is configured.", nameof(options));
        }

        _transport = transport;
        _options = options;
        _state = state ?? new MemoryFeedStateStore();
        _time = time ?? TimeProvider.System;
        _trust = new FeedTrust(options.TrustedRoots);
    }

    /// <summary>The feed's transport.</summary>
    public IFeedTransport Transport => _transport;

    /// <summary>The channel read, or <see langword="null"/>.</summary>
    public string? Channel => _options.Channel;

    /// <summary>The key under which the accepted sequence is remembered: the location, plus <c>#channel</c> for a channel.</summary>
    public string StateKey => _options.Channel is null ? _transport.Location : $"{_transport.Location}#{_options.Channel}";

    /// <summary>Reads and verifies the index and every catalog page, and records the accepted sequence.</summary>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>The snapshot.</returns>
    /// <exception cref="FeedVerificationException">A signature, digest, expiry or sequence check failed.</exception>
    /// <exception cref="FeedFormatException">A document is malformed or missing.</exception>
    /// <exception cref="IOException">The feed could not be read.</exception>
    public async Task<FeedSnapshot> GetSnapshotAsync(CancellationToken cancellationToken = default)
    {
        var now = _time.GetUtcNow();
        var warnings = new List<string>();
        var channel = _options.Channel;
        var indexBytes = await RequireAsync(FeedLayout.InChannel(channel, FeedLayout.ServiceIndex), cancellationToken).ConfigureAwait(false);
        FeedKeySet? keySet = null;
        if (_options.RequireSignature)
        {
            var keySetBytes = await RequireAsync(FeedLayout.KeySet, cancellationToken).ConfigureAwait(false);
            var keySetSignature = await RequireAsync(FeedLayout.KeySetSignature, cancellationToken).ConfigureAwait(false);
            keySet = _trust.VerifyKeySet(keySetBytes, keySetSignature, now);
        }
        else
        {
            warnings.Add("Signatures are not checked (requireSignature is off).");
        }

        var index = FeedJson.Read(indexBytes, FeedJsonContext.Default.FeedServiceIndex, "The feed index");
        if (index.SchemaVersion != 1)
        {
            throw new FeedFormatException($"Feed index schema version {index.SchemaVersion} is not supported (1 is).");
        }

        if (!string.Equals(index.Channel, channel, StringComparison.Ordinal))
        {
            throw new FeedVerificationException($"The feed index is for channel '{index.Channel ?? "(none)"}', not '{channel ?? "(none)"}'.");
        }

        if (keySet is not null)
        {
            var indexSignature = await RequireAsync(FeedLayout.InChannel(channel, FeedLayout.ServiceIndexSignature), cancellationToken).ConfigureAwait(false);
            DetachedSignature.Verify(indexBytes, indexSignature, kid => FeedTrust.ResolveFeedKey(keySet, kid, index.Updated), "The feed index");
        }

        if (index.Updated > now + _options.ClockSkew)
        {
            throw new FeedVerificationException($"The feed index claims to be written in the future ({index.Updated:u}).");
        }

        if (index.Expires < now)
        {
            if (!_options.AllowExpiredIndex)
            {
                throw new FeedVerificationException($"The feed index expired on {index.Expires:u} (the feed is stale or frozen).");
            }

            warnings.Add($"The feed index expired on {index.Expires:u}; accepted because expired indexes are allowed (offline).");
        }

        var indexDigest = FeedDigest.Of(indexBytes);
        var previous = await _state.GetAsync(StateKey, cancellationToken).ConfigureAwait(false);
        if (previous is not null)
        {
            if (index.Sequence < previous.Sequence)
            {
                throw new FeedVerificationException($"The feed index has sequence {index.Sequence}, older than the {previous.Sequence} seen before (rollback).");
            }

            if (index.Sequence == previous.Sequence && !string.Equals(previous.IndexDigest, indexDigest, StringComparison.Ordinal))
            {
                throw new FeedVerificationException($"The feed index changed without a new sequence ({index.Sequence}).");
            }
        }

        var catalogBytes = await RequireAsync(index.Resources.Catalog, cancellationToken).ConfigureAwait(false);
        FeedDigest.Require(catalogBytes, index.Catalog.Digest, "The catalog index");
        var catalog = FeedJson.Read(catalogBytes, FeedJsonContext.Default.FeedCatalogIndex, "The catalog index");
        var packages = new List<FeedPackageSummary>();
        foreach (var page in catalog.Pages)
        {
            var pageBytes = await RequireAsync(page.Path, cancellationToken).ConfigureAwait(false);
            FeedDigest.Require(pageBytes, page.Digest, $"Catalog page {page.Path}");
            packages.AddRange(FeedJson.Read(pageBytes, FeedJsonContext.Default.FeedCatalogPage, $"Catalog page {page.Path}").Packages);
        }

        if (packages.Count != index.Catalog.Packages || packages.Select(package => package.Id).Distinct(StringComparer.Ordinal).Count() != packages.Count)
        {
            throw new FeedVerificationException("The catalog pages do not match the index (count or duplicate ids).");
        }

        if (previous is null || index.Sequence > previous.Sequence)
        {
            await _state.SetAsync(StateKey, new FeedState { Sequence = index.Sequence, IndexDigest = indexDigest, Updated = index.Updated }, cancellationToken).ConfigureAwait(false);
        }

        return new FeedSnapshot(_transport.Location, channel, index, indexDigest, keySet, keySet is not null, [.. packages.OrderBy(package => package.Id, StringComparer.Ordinal)], warnings);
    }

    /// <summary>Reads a package's version list, verified against the snapshot.</summary>
    /// <param name="snapshot">A snapshot of this feed.</param>
    /// <param name="id">Package id.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>The package index.</returns>
    /// <exception cref="KeyNotFoundException">The feed has no such package.</exception>
    /// <exception cref="FeedVerificationException">The package index does not match its digest.</exception>
    public async Task<FeedPackageIndex> GetPackageIndexAsync(FeedSnapshot snapshot, string id, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        var summary = snapshot.Find(id) ?? throw new KeyNotFoundException($"The feed has no package '{id}'.");
        var bytes = await RequireAsync(FeedLayout.InChannel(snapshot.Channel, FeedLayout.PackageIndex(id)), cancellationToken).ConfigureAwait(false);
        FeedDigest.Require(bytes, summary.IndexDigest, $"The package index of {id}");
        var packageIndex = FeedJson.Read(bytes, FeedJsonContext.Default.FeedPackageIndex, $"The package index of {id}");
        if (packageIndex.Id != id)
        {
            throw new FeedVerificationException($"The package index of {id} names another package ({packageIndex.Id}).");
        }

        return packageIndex;
    }

    /// <summary>Reads and verifies one version's manifest (digest from the package index, then its signature).</summary>
    /// <param name="snapshot">A snapshot of this feed.</param>
    /// <param name="id">Package id.</param>
    /// <param name="version">Version.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>The verified manifest.</returns>
    /// <exception cref="KeyNotFoundException">The feed has no such version.</exception>
    /// <exception cref="FeedVerificationException">The manifest does not match its digest or signature.</exception>
    public async Task<VerifiedManifest> GetManifestAsync(FeedSnapshot snapshot, string id, string version, CancellationToken cancellationToken = default)
    {
        var packageIndex = await GetPackageIndexAsync(snapshot, id, cancellationToken).ConfigureAwait(false);
        var wanted = SemanticVersion.Parse(version);
        var entry = packageIndex.Versions.FirstOrDefault(candidate => SemanticVersion.TryParse(candidate.Version, out var parsed) && parsed == wanted)
            ?? throw new KeyNotFoundException($"The feed has no version {version} of '{id}'.");
        var bytes = await RequireAsync(FeedLayout.Manifest(id, entry.Version), cancellationToken).ConfigureAwait(false);
        byte[]? signature = snapshot.Verified
            ? await RequireAsync(FeedLayout.ManifestSignature(id, entry.Version), cancellationToken).ConfigureAwait(false)
            : null;
        return VerifyManifest(snapshot, bytes, signature, entry);
    }

    /// <summary>
    /// Verifies manifest bytes received from elsewhere (a peer, an offline bundle) against the snapshot's key set and the version's
    /// package index entry.
    /// </summary>
    /// <param name="snapshot">A snapshot of the feed the manifest belongs to.</param>
    /// <param name="manifest">Manifest bytes.</param>
    /// <param name="signature">Signature bytes, or <see langword="null"/> (accepted only for unverified snapshots).</param>
    /// <param name="entry">The version's entry from the package index.</param>
    /// <returns>The verified manifest.</returns>
    /// <exception cref="FeedVerificationException">The manifest does not match the entry or the signature.</exception>
    public static VerifiedManifest VerifyManifest(FeedSnapshot snapshot, byte[] manifest, byte[]? signature, FeedPackageVersion entry)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        ArgumentNullException.ThrowIfNull(manifest);
        ArgumentNullException.ThrowIfNull(entry);
        FeedDigest.Require(manifest, entry.ManifestDigest, $"The manifest of version {entry.Version}");
        string? keyId = null;
        if (snapshot.KeySet is { } keySet)
        {
            if (signature is null)
            {
                throw new FeedVerificationException($"The manifest of version {entry.Version} is not signed.");
            }

            keyId = DetachedSignature.Verify(manifest, signature, kid => FeedTrust.ResolveFeedKey(keySet, kid, entry.Published), $"The manifest of version {entry.Version}").KeyId;
        }

        var parsed = FeedJson.ReadManifest(manifest);
        if (!SemanticVersion.TryParse(parsed.Version, out var version) || version != SemanticVersion.Parse(entry.Version))
        {
            throw new FeedVerificationException($"The manifest names version {parsed.Version}, the package index {entry.Version}.");
        }

        return new VerifiedManifest(parsed, manifest, signature, keyId, entry);
    }

    private async Task<byte[]> RequireAsync(string path, CancellationToken cancellationToken)
    {
        return await _transport.ReadAsync(path, _options.MaxDocumentBytes, cancellationToken).ConfigureAwait(false)
            ?? throw new FeedFormatException($"{_transport.Location}{path} is missing.");
    }
}
