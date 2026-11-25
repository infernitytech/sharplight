using System.Reflection;
using Sharplight.Hub.Catalog;

namespace Sharplight.Studio.Services.Hub;

/// <summary>
/// The verified-model catalog Discover lists (docs/verified-catalog.md): the copy embedded at build time, replaced by a newer one from
/// <see cref="DefaultUrl"/> when that downloads, parses, has a supported schema version and passes <see cref="CatalogValidator"/>. The
/// download is tried at most every <see cref="RefreshInterval"/> and kept in a file, so later sessions and offline starts use it; any
/// failure keeps the current copy. Thread safe.
/// </summary>
internal sealed class VerifiedCatalogSource
{
    /// <summary>Where the published catalog is downloaded from.</summary>
    public const string DefaultUrl = "https://download.sharplight.dev/catalog/verified-models.json";

    /// <summary>Overrides <see cref="DefaultUrl"/> (an empty value disables the download).</summary>
    public const string UrlVariable = "SHARPLIGHT_STUDIO_CATALOG_URL";

    /// <summary>The name of the embedded copy.</summary>
    public const string ResourceName = "Sharplight.Studio.verified-models.json";

    /// <summary>How often the published catalog is asked for.</summary>
    public static readonly TimeSpan RefreshInterval = TimeSpan.FromDays(1);

    private readonly Func<CancellationToken, Task<string>>? _download;
    private readonly string? _cacheFile;
    private readonly Func<string> _embedded;
    private readonly TimeProvider _time;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private VerifiedCatalog? _current;
    private DateTimeOffset? _lastAttempt;

    /// <param name="download">Downloads the published catalog's text, or null to use the local copies only.</param>
    /// <param name="cacheFile">Where the downloaded catalog is kept, or null to keep it in memory only.</param>
    /// <param name="embedded">The embedded catalog's text; null for the resource built into Studio.</param>
    /// <param name="time">The clock; null for the system clock.</param>
    public VerifiedCatalogSource(Func<CancellationToken, Task<string>>? download, string? cacheFile, Func<string>? embedded = null, TimeProvider? time = null)
    {
        _download = download;
        _cacheFile = cacheFile;
        _embedded = embedded ?? ReadEmbedded;
        _time = time ?? TimeProvider.System;
    }

    /// <summary>A source that downloads with <paramref name="http"/> from <see cref="UrlVariable"/>, else <see cref="DefaultUrl"/>.</summary>
    public static VerifiedCatalogSource Create(HttpClient http, string? cacheFile)
    {
        var url = Environment.GetEnvironmentVariable(UrlVariable) ?? DefaultUrl;
        Func<CancellationToken, Task<string>>? download = url.Length == 0 ? null : token => http.GetStringAsync(new Uri(url), token);
        return new VerifiedCatalogSource(download, cacheFile);
    }

    /// <summary>The newest valid catalog: the downloaded one when it is newer than the embedded copy, else the embedded copy.</summary>
    /// <param name="cancellationToken">Cancels the download (the local copy is returned then).</param>
    public async Task<VerifiedCatalog> GetAsync(CancellationToken cancellationToken)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_current is null)
            {
                var embedded = VerifiedCatalogSerializer.Parse(_embedded());
                _current = Newer(embedded, TryRead(_cacheFile is not null && File.Exists(_cacheFile) ? File.ReadAllText(_cacheFile) : null));
            }

            if (_download is not null && IsStale())
            {
                await RefreshAsync(cancellationToken).ConfigureAwait(false);
            }

            return _current;
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>The cached download is missing or older than <see cref="RefreshInterval"/>, and it was not tried within that interval.</summary>
    private bool IsStale()
    {
        var now = _time.GetUtcNow();
        bool cacheOld = _cacheFile is null || !File.Exists(_cacheFile) || now - File.GetLastWriteTimeUtc(_cacheFile) >= RefreshInterval;
        return cacheOld && (_lastAttempt is not { } attempt || now - attempt >= RefreshInterval);
    }

    private async Task RefreshAsync(CancellationToken cancellationToken)
    {
        _lastAttempt = _time.GetUtcNow();
        string text;
        try
        {
            text = await _download!(cancellationToken).ConfigureAwait(false);
        }
        catch (Exception error) when (error is HttpRequestException or TaskCanceledException or IOException && !cancellationToken.IsCancellationRequested)
        {
            return;
        }

        if (TryRead(text) is not { } downloaded)
        {
            return;
        }

        WriteCache(text);
        _current = Newer(_current!, downloaded);
    }

    /// <summary>
    /// <paramref name="candidate"/> (the published copy, downloaded now or earlier) when it is valid and not older (by
    /// <see cref="VerifiedCatalog.Updated"/>), else <paramref name="current"/>. On the same date the published copy wins: entries were
    /// added to the catalog of 2026-11-20 without a new date, and hosts built before them kept their embedded copy.
    /// </summary>
    private static VerifiedCatalog Newer(VerifiedCatalog current, VerifiedCatalog? candidate)
        => candidate is not null && string.CompareOrdinal(candidate.Updated, current.Updated) >= 0 ? candidate : current;

    /// <summary>The catalog in <paramref name="text"/> when it parses and validates, else null.</summary>
    private static VerifiedCatalog? TryRead(string? text)
    {
        if (text is null)
        {
            return null;
        }

        try
        {
            var catalog = VerifiedCatalogSerializer.Parse(text);
            return CatalogValidator.Validate(catalog).Count == 0 ? catalog : null;
        }
        catch (InvalidDataException)
        {
            return null;
        }
    }

    private void WriteCache(string text)
    {
        if (_cacheFile is null)
        {
            return;
        }

        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(_cacheFile)!);
            string temporary = _cacheFile + ".tmp";
            File.WriteAllText(temporary, text);
            File.Move(temporary, _cacheFile, overwrite: true);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // Not cached: downloaded again next session.
        }
    }

    private static string ReadEmbedded()
    {
        using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(ResourceName)
            ?? throw new InvalidOperationException($"The embedded catalog {ResourceName} is missing.");
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }
}
