using Sharplight.Hub;
using Sharplight.Package.Import;
using Sharplight.Runtime;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Compatibility;
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
    private readonly HighlightsCatalog _highlights;

    /// <param name="settings">The settings (token, model store).</param>
    /// <param name="highlightsFile">Where the highlighted models are cached, or null to keep them in memory only.</param>
    public HubService(ISettingsService settings, string? highlightsFile = null)
    {
        _settings = settings;
        _highlights = new HighlightsCatalog(SearchAsync, highlightsFile);
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<HighlightSection>> GetHighlightsAsync(CancellationToken cancellationToken) => _highlights.GetAsync(cancellationToken);

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
    public async Task<ModelProbe> ProbeAsync(RepositoryDetails details, VariantInfo variant, bool readWeightFormats, CancellationToken cancellationToken)
    {
        var client = CreateClient();
        var repo = details.Repository;
        try
        {
            switch (details.Format)
            {
                case ModelFormat.Gguf when variant.PrimaryPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase):
                    Func<long, int, CancellationToken, Task<byte[]>> Reader(string path) => (offset, count, token) => client.GetFileRangeAsync(repo, path, offset, count, token);
                    return await VariantProbe.ProbeGgufAsync(Reader(variant.PrimaryPath), readWeightFormats, cancellationToken,
                        [.. variant.OtherShards.Select(Reader)]).ConfigureAwait(false);
                case ModelFormat.Safetensors or ModelFormat.Onnx:
                    bool onnx = variant.PrimaryPath.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase);
                    var directory = onnx ? variant.PrimaryPath[..Math.Max(0, variant.PrimaryPath.LastIndexOf('/'))] : variant.PrimaryPath;
                    var configPath = FindNear(repo, directory, "config.json");
                    string? config = configPath is null ? null : await client.GetFileTextAsync(repo, configPath, cancellationToken).ConfigureAwait(false);
                    var shape = config is null ? null : ModelShapeReader.FromConfig(config);
                    ModelCompatibility compatibility;
                    if (onnx)
                    {
                        long size = repo.Files.FirstOrDefault(file => file.Path == variant.PrimaryPath)?.Size ?? 0;
                        compatibility = await VariantProbe.ProbeOnnxAsync((offset, count, token) => client.GetFileRangeAsync(repo, variant.PrimaryPath, offset, count, token),
                            size, cancellationToken).ConfigureAwait(false);
                    }
                    else if (config is null)
                    {
                        compatibility = ModelCompatibility.Unknown("The checkpoint has no config.json, so the architecture is not known.");
                    }
                    else
                    {
                        var modelOptPath = FindNear(repo, directory, "hf_quant_config.json");
                        string? modelOpt = modelOptPath is null ? null : await client.GetFileTextAsync(repo, modelOptPath, cancellationToken).ConfigureAwait(false);
                        compatibility = CompatibilityChecker.FromConfig(config, modelOpt);
                    }

                    return new ModelProbe(shape, compatibility, compatibility);
                default:
                    var unknown = ModelCompatibility.Unknown("The repository has no model file whose metadata can be read before downloading.");
                    return new ModelProbe(null, unknown, unknown);
            }
        }
        catch (Exception error) when (error is HubException or HttpRequestException or InvalidDataException)
        {
            var unknown = ModelCompatibility.Unknown($"The metadata could not be read before downloading ({error.Message}).");
            return new ModelProbe(null, unknown, unknown);
        }
    }

    /// <summary>A file in <paramref name="directory"/> of the repository, else at its root, else null.</summary>
    private static string? FindNear(HubRepository repo, string directory, string name)
    {
        var near = directory.Length == 0 ? name : $"{directory.TrimEnd('/')}/{name}";
        foreach (var path in new[] { near, name })
        {
            if (repo.Files.Any(file => file.Path == path))
            {
                return path;
            }
        }

        return null;
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
        var shards = selection.Format == ModelFormat.Gguf
            ? selection.Files.Select(file => file.Path)
                .Where(path => path != selection.PrimaryPath && path != selection.ProjectorPath && path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
                .Order(StringComparer.Ordinal).ToList()
            : [];
        return new VariantInfo(selection.Tag, selection.TotalBytes, selection.Files.Count, selection.ProjectorPath is not null, recommended,
            selection.PrimaryPath, shards);
    }

    private PrecompileTarget? CreatePrecompileTarget(PullProgressTracker tracker, CancellationToken cancellationToken)
    {
        var settings = _settings.Current;
        var gpus = DeviceCatalog.Discover().Where(entry => entry.Option.IsGpu).ToList();
        var gpu = gpus.FirstOrDefault(entry => entry.Option.Id == settings.DefaultLoad.DeviceId);
        if (gpu.Provider is null && settings.DefaultLoad.DeviceId is null or ComputeDeviceOption.AllGpusId)
        {
            gpu = gpus.FirstOrDefault();
        }
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
