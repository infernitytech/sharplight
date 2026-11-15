using System.Text.Json;
using Sharplight.Hub;
using Sharplight.Studio.Services.Compatibility;

namespace Sharplight.Studio.Services.Hub;

/// <summary>
/// The models Discover shows before anything is searched: trending, new and most downloaded language models on the Hub, GGUF and
/// safetensors, kept to those whose architecture Sharplight supports (<see cref="CompatibilityChecker.FromSummary"/>). The Hub is
/// asked at most every <see cref="CacheLifetime"/>; the answer is kept in a file, so opening Discover again (or after a restart) does
/// not ask again, and an older answer is shown while the Hub cannot be reached.
/// </summary>
internal sealed class HighlightsCatalog
{
    /// <summary>How long an answer of the Hub is shown before it is asked again.</summary>
    public static readonly TimeSpan CacheLifetime = TimeSpan.FromHours(6);

    /// <summary>How recent a model must be for the <see cref="HighlightKind.New"/> section.</summary>
    public static readonly TimeSpan NewWithin = TimeSpan.FromDays(120);

    /// <summary>Models per section.</summary>
    public const int PerSection = 8;

    /// <summary>Repositories asked for per listing (the supported ones are a part of them).</summary>
    private const int ListingSize = 100;

    /// <summary>Pipeline tags of the models that are listed (repositories without a tag are kept too).</summary>
    private static readonly HashSet<string> s_pipelines = new(StringComparer.OrdinalIgnoreCase) { "text-generation", "image-text-to-text" };

    private static readonly ModelFormat[] s_formats = [ModelFormat.Gguf, ModelFormat.Safetensors];

    private readonly Func<HubSearchQuery, CancellationToken, Task<IReadOnlyList<HubModelSummary>>> _search;
    private readonly string? _cacheFile;
    private readonly ModelTask? _task;
    private readonly TimeProvider _time;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private HighlightsCacheFile? _current;
    private bool _cacheRead;

    /// <param name="search">Lists repositories on the Hub.</param>
    /// <param name="cacheFile">The file the answer is kept in between sessions, or null to keep it in memory only.</param>
    /// <param name="time">The clock; null for the system clock.</param>
    /// <param name="task">
    /// The model type to highlight (every format and every compatibility verdict: Discover marks the models the engine cannot run), or
    /// null for the supported language models.
    /// </param>
    public HighlightsCatalog(Func<HubSearchQuery, CancellationToken, Task<IReadOnlyList<HubModelSummary>>> search, string? cacheFile, TimeProvider? time = null,
        ModelTask? task = null)
    {
        _search = search;
        _task = task;
        _cacheFile = cacheFile;
        _time = time ?? TimeProvider.System;
    }

    /// <summary>The sections, from the cache while it is younger than <see cref="CacheLifetime"/>, else from the Hub.</summary>
    /// <param name="cancellationToken">Cancels the requests.</param>
    /// <exception cref="HubException">The Hub could not be asked and nothing is cached.</exception>
    public async Task<IReadOnlyList<HighlightSection>> GetAsync(CancellationToken cancellationToken)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (!_cacheRead)
            {
                _cacheRead = true;
                _current = ReadCache();
            }

            var now = _time.GetUtcNow();
            if (_current is { } cached && now - cached.FetchedAt < CacheLifetime && now >= cached.FetchedAt)
            {
                return cached.Sections;
            }

            try
            {
                var sections = await FetchAsync(now, cancellationToken).ConfigureAwait(false);
                _current = new HighlightsCacheFile(now, sections);
                WriteCache(_current);
                return sections;
            }
            catch (Exception error) when (_current is not null && !cancellationToken.IsCancellationRequested
                && error is HubException or HttpRequestException or TaskCanceledException)
            {
                return _current.Sections;
            }
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>
    /// The sections from the Hub's listings: trending (the trending listings of both formats, alternating), new (created within
    /// <see cref="NewWithin"/>, newest first) and popular (most downloaded); each model appears in one section only.
    /// </summary>
    /// <param name="trending">Trending listings, one per format, best first.</param>
    /// <param name="downloaded">Most-downloaded listings, one per format, best first.</param>
    /// <param name="now">The current time.</param>
    /// <param name="isEligible">Which repositories are listed; null for <see cref="IsEligible"/>.</param>
    public static IReadOnlyList<HighlightSection> Build(IReadOnlyList<IReadOnlyList<HubModelSummary>> trending,
        IReadOnlyList<IReadOnlyList<HubModelSummary>> downloaded, DateTimeOffset now, Func<HubModelSummary, bool>? isEligible = null)
    {
        isEligible ??= IsEligible;
        var shown = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        List<HubModelSummary> Take(IEnumerable<HubModelSummary> candidates)
        {
            var picked = new List<HubModelSummary>();
            foreach (var model in candidates)
            {
                if (picked.Count == PerSection)
                {
                    break;
                }

                if (isEligible(model) && shown.Add(model.Id))
                {
                    picked.Add(model);
                }
            }

            return picked;
        }

        var trendingModels = Take(Interleave(trending));
        var everything = Interleave(trending).Concat(downloaded.SelectMany(list => list)).DistinctBy(model => model.Id, StringComparer.OrdinalIgnoreCase).ToList();
        var newModels = Take(everything.Where(model => model.CreatedAt is { } created && now - created <= NewWithin)
            .OrderByDescending(model => model.CreatedAt));
        var popular = Take(downloaded.SelectMany(list => list).OrderByDescending(model => model.Downloads));
        return
        [
            new HighlightSection(HighlightKind.Trending, trendingModels),
            new HighlightSection(HighlightKind.New, newModels),
            new HighlightSection(HighlightKind.Popular, popular),
        ];
    }

    /// <summary>A language model whose architecture (and quantization, for a checkpoint) Sharplight supports.</summary>
    public static bool IsEligible(HubModelSummary model)
        => !model.HasProjectorOnly && (model.Pipeline is null || s_pipelines.Contains(model.Pipeline))
           && CompatibilityChecker.FromSummary(model).State == CompatibilityState.Supported;

    private async Task<IReadOnlyList<HighlightSection>> FetchAsync(DateTimeOffset now, CancellationToken cancellationToken)
    {
        Task<IReadOnlyList<HubModelSummary>> List(HubSortOrder sort, ModelFormat? format)
            => _search(new HubSearchQuery { Sort = sort, Format = format, Task = _task, Limit = ListingSize }, cancellationToken);

        // A task lists every format in one listing; the default highlights list GGUF and safetensors separately and interleave them.
        ModelFormat?[] formats = _task is null ? [.. s_formats.Cast<ModelFormat?>()] : [null];
        var trending = formats.Select(format => List(HubSortOrder.Trending, format)).ToList();
        var downloaded = formats.Select(format => List(HubSortOrder.Downloads, format)).ToList();
        await Task.WhenAll([.. trending, .. downloaded]).ConfigureAwait(false);
        return Build([.. trending.Select(list => list.Result)], [.. downloaded.Select(list => list.Result)], now,
            _task is null ? null : model => !model.HasProjectorOnly);
    }

    /// <summary>The first of every list, then the second of every list, and so on.</summary>
    private static IEnumerable<HubModelSummary> Interleave(IReadOnlyList<IReadOnlyList<HubModelSummary>> lists)
    {
        int longest = lists.Count == 0 ? 0 : lists.Max(list => list.Count);
        for (int i = 0; i < longest; i++)
        {
            foreach (var list in lists)
            {
                if (i < list.Count)
                {
                    yield return list[i];
                }
            }
        }
    }

    private HighlightsCacheFile? ReadCache()
    {
        try
        {
            return _cacheFile is not null && File.Exists(_cacheFile)
                ? JsonSerializer.Deserialize(File.ReadAllText(_cacheFile), StudioJsonContext.Default.HighlightsCacheFile)
                : null;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or JsonException or NotSupportedException)
        {
            return null;
        }
    }

    private void WriteCache(HighlightsCacheFile file)
    {
        if (_cacheFile is null)
        {
            return;
        }

        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(_cacheFile)!);
            string temporary = _cacheFile + ".tmp";
            File.WriteAllText(temporary, JsonSerializer.Serialize(file, StudioJsonContext.Default.HighlightsCacheFile));
            File.Move(temporary, _cacheFile, overwrite: true);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // Not cached: the Hub is asked again next time.
        }
    }
}
