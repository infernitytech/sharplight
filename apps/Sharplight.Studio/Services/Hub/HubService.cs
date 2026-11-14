using Sharplight.Hub;
using Sharplight.Package.Import;
using Sharplight.Runtime;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Services.Settings;

namespace Sharplight.Studio.Services.Hub;

/// <summary>
/// <see cref="IHubService"/> over <see cref="HubClient"/> and <see cref="ModelPuller"/>: the same downloader (resumable,
/// verified), converter and store layout as <c>sharplight pull</c>. The token and store come from the settings at each call.
/// </summary>
internal sealed class HubService : IHubService, IDisposable
{
    private const int PrecompileSlots = 4;

    private readonly ISettingsService _settings;
    private readonly HttpClient _http = HubClientOptions.CreateHttpClient();

    public HubService(ISettingsService settings)
    {
        _settings = settings;
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<HubModelSummary>> SearchAsync(HubSearchQuery query, CancellationToken cancellationToken)
    {
        return CreateClient().SearchAsync(query, cancellationToken);
    }

    /// <inheritdoc/>
    public async Task<RepositoryDetails> GetDetailsAsync(string repository, CancellationToken cancellationToken)
    {
        var client = CreateClient();
        var repo = await client.GetRepositoryAsync(repository, "main", cancellationToken).ConfigureAwait(false);
        var paths = repo.Files.Select(file => file.Path).ToList();
        var format = ModelFileSelector.DetectFormat(paths);
        if (format is null)
        {
            return new RepositoryDetails(repo, null, []);
        }

        var tags = ModelFileSelector.ListVariants(paths);
        var variants = new List<VariantInfo>();
        if (tags.Count == 0)
        {
            string? index = null;
            if (ModelFileSelector.IndexPath(repo, null) is { } indexPath)
            {
                index = await client.GetFileTextAsync(repo, indexPath, cancellationToken).ConfigureAwait(false);
            }

            variants.Add(ToVariant(ModelFileSelector.Select(repo, null, index), recommended: true));
        }
        else
        {
            var recommended = ModelFileSelector.Select(repo, null).Tag;
            foreach (var tag in tags)
            {
                var selection = ModelFileSelector.Select(repo, tag);
                variants.Add(ToVariant(selection, string.Equals(selection.Tag, recommended, StringComparison.OrdinalIgnoreCase)));
            }
        }

        return new RepositoryDetails(repo, format, variants);
    }

    /// <inheritdoc/>
    public async Task<ModelManifest> PullAsync(ModelReference reference, IProgress<PullUpdate> progress, CancellationToken cancellationToken)
    {
        EngineBootstrap.Register();
        var settings = _settings.Current;
        var clientOptions = ClientOptions();
        var client = new HubClient(_http, clientOptions);
        var store = new ModelStore(_settings.StoreRoot);
        var puller = new ModelPuller(client, new FileDownloader(_http, clientOptions), store);
        var tracker = new PullProgressTracker(progress);
        var options = new PullOptions
        {
            KeepSource = settings.KeepSourceFiles,
            Precompile = settings.PrecompileAfterDownload ? CreatePrecompileTarget(tracker, cancellationToken) : null,
            Log = tracker.Log,
            Progress = tracker,
        };

        tracker.Report(PullPhase.Resolving, $"Resolving {reference}");
        var result = await puller.PullAsync(reference, options, cancellationToken).ConfigureAwait(false);
        tracker.Report(PullPhase.Completed, result.UpToDate ? $"{result.Manifest.Name} is up to date" : $"Stored {result.Manifest.ShortName}");
        return result.Manifest;
    }

    /// <inheritdoc/>
    public void DiscardPartial(ModelReference reference)
    {
        var store = new ModelStore(_settings.StoreRoot);
        var directory = store.GetModelDirectory(reference.Repository, reference.Tag ?? "");
        if (Directory.Exists(directory) && !File.Exists(Path.Combine(directory, ModelStore.ManifestFileName)))
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>Releases the HTTP client.</summary>
    public void Dispose() => _http.Dispose();

    private HubClient CreateClient() => new(_http, ClientOptions());

    private HubClientOptions ClientOptions()
    {
        var fromEnvironment = HubClientOptions.FromEnvironment();
        var token = _settings.Current.HuggingFaceToken;
        return string.IsNullOrWhiteSpace(token) ? fromEnvironment : new HubClientOptions
        {
            Endpoint = fromEnvironment.Endpoint,
            Token = token.Trim(),
            UserAgent = fromEnvironment.UserAgent,
            RequestTimeout = fromEnvironment.RequestTimeout,
            StallTimeout = fromEnvironment.StallTimeout,
            Retry = fromEnvironment.Retry,
        };
    }

    private static VariantInfo ToVariant(ModelSelection selection, bool recommended)
    {
        return new VariantInfo(selection.Tag, selection.TotalBytes, selection.Files.Count, selection.ProjectorPath is not null, recommended);
    }

    private PrecompileTarget? CreatePrecompileTarget(PullProgressTracker tracker, CancellationToken cancellationToken)
    {
        var settings = _settings.Current;
        var gpu = DeviceCatalog.Discover().FirstOrDefault(entry => entry.Option.IsGpu
            && (settings.DefaultLoad.DeviceId is null || entry.Option.Id == settings.DefaultLoad.DeviceId));
        if (gpu.Provider is null)
        {
            return null;
        }

        var options = new PrecompileOptions
        {
            ContextLength = settings.DefaultLoad.ContextLength,
            BatchTokens = settings.DefaultLoad.BatchTokens,
            Slots = PrecompileSlots,
            DisabledFeatures = settings.DisabledFeatures,
            UseUserKernelCache = true,
            Log = tracker.Log,
            Progress = new SynchronousProgress<LoadProgress>(tracker.Report),
            CancellationToken = cancellationToken,
        };
        return new PrecompileTarget(gpu.Provider, gpu.Info, options);
    }
}
