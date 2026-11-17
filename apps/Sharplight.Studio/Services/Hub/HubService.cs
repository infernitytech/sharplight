using Microsoft.Extensions.DependencyInjection;
using Sharplight.Hub;
using Sharplight.Hub.Catalog;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Studio.Services.Compatibility;
using Sharplight.Studio.Services.Precompile;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Services.Settings;

namespace Sharplight.Studio.Services.Hub;

/// <summary>
/// <see cref="IHubService"/> over <see cref="HubClient"/> and <see cref="ModelPuller"/>: the same downloader (resumable,
/// verified), converter and store layout as <c>sharplight pull</c>. The token and store come from the settings at each call.
/// </summary>
internal sealed class HubService : IHubService, IDisposable
{

    private readonly ISettingsService _settings;
    private readonly HttpClient _http = HubClientOptions.CreateHttpClient();
    private readonly HighlightsCatalog _highlights;
    private readonly EngineServices _engine;
    private readonly Dictionary<ModelTask, HighlightsCatalog> _taskHighlights = [];
    private readonly string? _highlightsFile;
    private readonly VerifiedCatalogSource _catalog;

    /// <param name="settings">The settings (token, model store).</param>
    /// <param name="engine">The engine services (conversion, providers for precompilation).</param>
    /// <param name="highlightsFile">Where the highlighted models are cached, or null to keep them in memory only.</param>
    /// <param name="catalogFile">Where the downloaded verified catalog is kept, or null to keep it in memory only.</param>
    public HubService(ISettingsService settings, EngineServices engine, string? highlightsFile = null, string? catalogFile = null)
    {
        _settings = settings;
        _engine = engine;
        _highlightsFile = highlightsFile;
        _highlights = new HighlightsCatalog(SearchAsync, highlightsFile);
        _catalog = VerifiedCatalogSource.Create(_http, catalogFile);
    }

    /// <inheritdoc/>
    public Task<VerifiedCatalog> GetCatalogAsync(CancellationToken cancellationToken) => _catalog.GetAsync(cancellationToken);

    /// <inheritdoc/>
    public Task<IReadOnlyList<HighlightSection>> GetHighlightsAsync(ModelTask? task, CancellationToken cancellationToken)
    {
        if (task is not { } value)
        {
            return _highlights.GetAsync(cancellationToken);
        }

        HighlightsCatalog catalog;
        lock (_taskHighlights)
        {
            if (!_taskHighlights.TryGetValue(value, out var existing))
            {
                string? file = _highlightsFile is null ? null
                    : Path.Combine(Path.GetDirectoryName(_highlightsFile) ?? "", $"{Path.GetFileNameWithoutExtension(_highlightsFile)}-{value.ToString().ToLowerInvariant()}.json");
                existing = new HighlightsCatalog(SearchAsync, file, task: value);
                _taskHighlights[value] = existing;
            }

            catalog = existing;
        }

        return catalog.GetAsync(cancellationToken);
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
        string? index = null;
        if (ModelFileSelector.IndexPath(repo, null) is { } indexPath)
        {
            index = await client.GetFileTextAsync(repo, indexPath, cancellationToken).ConfigureAwait(false);
        }

        if (format == ModelFormat.Diffusers)
        {
            var pipeline = index is null ? null : DiffusersPipeline.Parse(index);
            var support = pipeline is null ? null : ClassifyPipeline(pipeline);
            if (support is { Supported: false })
            {
                return new RepositoryDetails(repo, null, [], ModelFormat.Diffusers, RepoLayouts.DiffusersComponents(repo.Files), support.Reason);
            }

            try
            {
                // A component without safetensors weights cannot be read: say so instead of offering a download that fails.
                ModelFileSelector.Select(repo, null, index);
            }
            catch (HubException error)
            {
                return new RepositoryDetails(repo, null, [], ModelFormat.Diffusers, RepoLayouts.DiffusersComponents(repo.Files), error.Message);
            }
        }

        if (format is null)
        {
            if (paths.Any(path => path.EndsWith(".slm", StringComparison.OrdinalIgnoreCase)))
            {
                return new RepositoryDetails(repo, null, [], ModelFormat.Slm,
                    [.. repo.Files.Where(file => file.Path.EndsWith(".slm", StringComparison.OrdinalIgnoreCase)).Select(file => new HubComponent(file.Path, file.Size, 1))]);
            }

            return new RepositoryDetails(repo, null, []);
        }

        var tags = ModelFileSelector.ListVariants(paths, index);
        var variants = new List<VariantInfo>();
        if (tags.Count == 0)
        {
            variants.Add(ToVariant(ModelFileSelector.Select(repo, null, index), recommended: true));
        }
        else
        {
            var recommended = ModelFileSelector.Select(repo, null, index).Tag;
            foreach (var tag in tags)
            {
                var selection = ModelFileSelector.Select(repo, tag, index);
                variants.Add(ToVariant(selection, string.Equals(selection.Tag, recommended, StringComparison.OrdinalIgnoreCase)));
            }
        }

        if (format is ModelFormat.Diffusers or ModelFormat.WhisperCpp)
        {
            var parts = format == ModelFormat.Diffusers ? RepoLayouts.DiffusersComponents(repo.Files) : RepoLayouts.WhisperModels(repo.Files);
            return new RepositoryDetails(repo, format, variants, format, parts);
        }

        return new RepositoryDetails(repo, format, variants);
    }

    /// <summary>Asks the registered importers whether the pipeline can be loaded; null when none knows the pipeline kind.</summary>
    private PipelineSupport? ClassifyPipeline(DiffusersPipeline pipeline)
    {
        var catalog = _engine.Services.GetRequiredService<ModelImporterCatalog>();
        return catalog.Importers.Select(importer => importer.ClassifyPipeline(pipeline.ClassName, pipeline.Components)).FirstOrDefault(support => support is not null);
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
        var (puller, tracker, options) = CreatePuller(progress);
        tracker.Report(PullPhase.Resolving, $"Resolving {reference}");
        var result = await puller.PullAsync(reference, options, cancellationToken).ConfigureAwait(false);
        tracker.Report(PullPhase.Completed, result.UpToDate ? $"{result.Manifest.Name} is up to date" : $"Stored {result.Manifest.ShortName}");
        return result.Manifest;
    }

    /// <summary>A puller on the current settings (token, store, precompile and keep-source choices) reporting to <paramref name="progress"/>.</summary>
    private (ModelPuller Puller, PullProgressTracker Tracker, PullOptions Options) CreatePuller(IProgress<PullUpdate> progress)
    {
        var settings = _settings.Current;
        var clientOptions = ClientOptions();
        var client = new HubClient(_http, clientOptions);
        var store = new ModelStore(_settings.StoreRoot);
        var puller = new ModelPuller(client, new FileDownloader(_http, clientOptions), store, _engine.Converter);
        var tracker = new PullProgressTracker(progress);
        var options = new PullOptions
        {
            KeepSource = settings.KeepSourceFiles,
            Precompile = settings.PrecompileAfterDownload ? CreatePrecompileTarget(tracker) : null,
            Log = tracker.Log,
            Progress = tracker,
        };
        return (puller, tracker, options);
    }

    /// <inheritdoc/>
    public async Task<ModelManifest> InstallAsync(CatalogInstallPlan plan, IProgress<PullUpdate> progress, CancellationToken cancellationToken)
    {
        var (puller, tracker, options) = CreatePuller(progress);
        tracker.Report(PullPhase.Resolving, $"Installing {plan.Entry.Name} ({plan.Variant.Label})");
        var result = await puller.InstallAsync(plan, options, cancellationToken).ConfigureAwait(false);
        tracker.Report(PullPhase.Completed, result.UpToDate ? $"{plan.Entry.Name} is installed" : $"Stored {result.Manifest.ShortName}");
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

    /// <summary>
    /// Precompiles a pulled package for every device of every available execution provider (portable code, then device code), with the
    /// options the library's Precompile action uses (<see cref="PrecompileService.CreateOptions"/>).
    /// </summary>
    private PrecompileTarget CreatePrecompileTarget(PullProgressTracker tracker)
    {
        var providers = _engine.Providers.Discover().Select(p => p.Provider).ToList();
        var options = PrecompileService.CreateOptions(_settings.Current, null, tracker.Log, new SynchronousProgress<LoadProgress>(tracker.Report));
        return new PrecompileTarget(providers, options, new SynchronousProgress<PrecompileProgress>(tracker.Report));
    }
}
