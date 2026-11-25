using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Sharplight.Hosting;
using Sharplight.Hosting.Fit;
using Sharplight.Hosting.Loaders;
using Sharplight.Llm;
using Sharplight.Llm.Embeddings;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Llm.Serving.Speculative;
using Sharplight.Runtime;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Threading;
using Sharplight.Tasks;
using Sharplight.Tensors;
using Sharplight.Vision;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// <see cref="IModelRuntime"/> on the Sharplight engine's <see cref="ModelHost"/>, the same hosting layer <c>sharplight serve</c>
/// uses: several models stay resident at once within the device memory budget (least recently used idle ones are evicted when
/// another does not fit, idle ones unload after a timeout and reload on demand), devices are opened once through a
/// <see cref="DevicePool"/>, and the HTTP endpoints serve the same host, so app chats and API requests are batched together.
/// </summary>
internal sealed class EngineModelRuntime : IModelRuntime
{
    private static readonly TimeSpan s_drainTimeout = TimeSpan.FromSeconds(5);

    private readonly DevicePool _pool = new();
    private readonly ILogger _logger;
    private readonly Dictionary<string, IEngineModel> _models = new(StringComparer.Ordinal);
    private readonly Lock _lock = new();
    private readonly Dictionary<string, (IExecutionProvider Provider, DeviceInfo Info)> _devices;
    private readonly Func<ResidencySettings> _residency;
    private readonly Func<bool> _useCompilationCache;
    private readonly Func<string, ModelLoadRequest?>? _resolve;
    private readonly IReadOnlyList<ComputeDeviceOption> _physical;
    private readonly DeviceProviderSelector _selector;
    private readonly FootprintEstimator _estimator;
    private readonly EngineServices _engine;
    private readonly UpdateThrottle _changed;

    /// <summary>Enumerates the devices and creates the host.</summary>
    /// <param name="engine">The engine services (loader, providers).</param>
    /// <param name="residency">The current residency settings (read when a model is added).</param>
    /// <param name="resolve">Turns a name requested by a chat or API client into a library model to load on demand, or null.</param>
    /// <param name="useCompilationCache">Whether devices use the per-user artifact store (read when a device is opened); default yes.</param>
    /// <param name="logger">Receives the device list and load events (never prompts or other content), or null.</param>
    /// <param name="providerChoices">The per-GPU provider choices of the settings (read when a device list or a load is resolved); null: all automatic.</param>
    /// <param name="hints">Per-device knowledge that <see cref="ProviderChoice.Automatic"/> consults; null: none, so Vulkan.</param>
    public EngineModelRuntime(EngineServices engine, Func<ResidencySettings> residency, Func<string, ModelLoadRequest?>? resolve = null, Func<bool>? useCompilationCache = null,
        ILogger? logger = null, Func<IReadOnlyDictionary<string, ProviderChoice>>? providerChoices = null, IProviderPerformanceHints? hints = null)
    {
        _logger = logger ?? NullLogger.Instance;
        _engine = engine;
        _estimator = new FootprintEstimator(engine.Loader);
        _useCompilationCache = useCompilationCache ?? (() => true);
        var discovery = DeviceCatalog.Discover(engine.Providers);
        _physical = discovery.Options;
        _selector = new DeviceProviderSelector(discovery, providerChoices, hints, _logger);
        _logger.LogInformation("Compute devices: {Devices}", _physical.Count == 0 ? "none"
            : string.Join("; ", _selector.Devices.Select(device => $"{device.Name} [{device.Provider}{(device.IsGpu ? ", GPU" : "")}]")));
        foreach (var (provider, reason) in discovery.Unavailable)
        {
            _logger.LogInformation("Execution provider {Provider} is not available: {Reason}", provider, reason);
        }

        foreach (var note in _selector.Notes)
        {
            _logger.LogWarning("{Note}", note);
        }

        _devices = discovery.Entries.ToDictionary(entry => entry.Option.Id, entry => (entry.Provider, entry.Info));
        _residency = residency;
        _resolve = resolve;
        var settings = residency();
        Host = new ModelHost(new ModelHostOptions
        {
            ReserveBytes = Math.Max(0, settings.ReserveMiB) * (1L << 20),
            MaxLoadedModels = Math.Max(0, settings.MaxLoadedModels),
            IdleCheckInterval = TimeSpan.FromSeconds(15),
            Resolve = Resolve,
        });
        Host.ModelsChanged += OnHostChanged;
        _changed = new UpdateThrottle(() => ModelsChanged?.Invoke(this, EventArgs.Empty));
    }

    /// <summary>The models, shared with the HTTP endpoints.</summary>
    public ModelHost Host { get; }

    /// <inheritdoc/>
    public IReadOnlyList<ComputeDeviceOption> Devices
    {
        get
        {
            // Only GPUs of the preferred provider are split over: on a Mac the Metal and the Vulkan (MoltenVK) device are the same GPU,
            // and on an NVIDIA GPU the Vulkan and the CUDA entry are.
            var devices = _selector.Devices;
            var splittable = SplittableGpus();
            return splittable.Count > 1 ? [.. devices.Where(device => device.IsGpu), ComputeDeviceOption.AllGpus(splittable), .. devices.Where(device => !device.IsGpu)] : devices;
        }
    }

    /// <inheritdoc/>
    public IReadOnlyList<PhysicalGpu> Gpus => _selector.Gpus;

    /// <inheritdoc/>
    public IReadOnlyList<ILoadedModel> LoadedModels
    {
        get
        {
            var order = Host.List().Select(info => info.Name).ToList();
            lock (_lock)
            {
                return [.. order.Where(_models.ContainsKey).Select(name => _models[name])];
            }
        }
    }

    /// <inheritdoc/>
    public event EventHandler? ModelsChanged;

    /// <inheritdoc/>
    public event EventHandler<RequestRecord>? RequestCompleted;

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, string> FindCompanions(string path)
    {
        try
        {
            return _engine.Loader.Importers.FindCompanions(path);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or InvalidDataException or NotSupportedException)
        {
            return new Dictionary<string, string>();
        }
    }

    /// <inheritdoc/>
    public IReadOnlyList<Sharplight.Models.CompanionFile> DescribeCompanions(string path)
    {
        try
        {
            return _engine.Loader.Importers.DescribeCompanions(path);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or InvalidDataException or NotSupportedException)
        {
            return [];
        }
    }

    /// <inheritdoc/>
    public async Task<ILoadedModel> LoadAsync(ModelLoadRequest request, IProgress<LoadStatus>? progress, CancellationToken cancellationToken)
    {
        if (Host.Contains(request.Id))
        {
            throw new InvalidOperationException($"A model named '{request.Id}' is already loaded.");
        }

        // Opening the devices (driver initialization, pipeline caches) can take seconds: never on the caller's (UI) thread.
        bool pinned = _residency().PinManualLoads;
        while (true)
        {
            var model = await Task.Run(() => Add(request, pinned), cancellationToken).ConfigureAwait(false);
            model.LoadListener = progress;
            try
            {
                _logger.LogInformation("Loading a {Kind} model on {Device} ({Provider})", request.Kind, model.Device.Name, ProviderTitles.Of(model.Device.Provider));
                progress?.Report(LoadStatus.Message($"Loading {Path.GetFileName(request.Path)} on {model.Device.Name}"));
                await Host.LoadAsync(request.Id, cancellationToken).ConfigureAwait(false);
                return model;
            }
            catch (Exception loadError)
            {
                // Cancelled or failed: abandon a load still running and forget the model.
                _logger.LogWarning(loadError, "Loading a {Kind} model failed", request.Kind);
                AbandonLoad(request.Id);
                await RemoveAsync(request.Id, force: true).ConfigureAwait(false);

                // A GPU another provider can drive (CUDA failing at its kernels, an old driver): load again on that one, once per failing provider.
                if (_selector.Failover(model.Device, loadError) is not { } fallback)
                {
                    throw;
                }

                progress?.Report(LoadStatus.Message($"{ProviderTitles.Of(model.Device.Provider)} failed on {model.Device.Name}; loading with {ProviderTitles.Of(fallback.Provider)}"));
                request = request with { Settings = request.Settings with { DeviceId = fallback.Id } };
            }
            finally
            {
                model.LoadListener = null;
            }
        }
    }

    /// <inheritdoc/>
    public Task UnloadAsync(string id) => RemoveAsync(id, force: true);

    /// <inheritdoc/>
    public IReadOnlyList<DeviceUsage> GetDeviceUsage()
    {
        var usage = new List<DeviceUsage>();
        foreach (var device in _pool.Devices)
        {
            var info = device.GetMemoryInfo();
            var id = $"{device.Info.Provider}:{device.Info.Index}";
            long used = info.IsExact ? info.Usage : info.UsedByUs;
            usage.Add(new DeviceUsage(id, device.Info.Name, used, info.Total > 0 ? info.Total : device.Info.TotalMemory, info.IsExact));
        }

        return usage;
    }

    /// <inheritdoc/>
    public Task<ModelFootprint> EstimateFootprintAsync(ModelLoadRequest request, CancellationToken cancellationToken)
        => Task.Run(() => _estimator.Estimate(request, cancellationToken), cancellationToken);

    /// <inheritdoc/>
    public Task<long> EstimateExpertBytesAsync(ModelLoadRequest request, CancellationToken cancellationToken)
        => Task.Run(() => _estimator.ExpertBytes(request, cancellationToken), cancellationToken);

    /// <inheritdoc/>
    public Task<double> EstimateExpertActiveShareAsync(ModelLoadRequest request, CancellationToken cancellationToken)
        => Task.Run(() => _estimator.ExpertActiveShare(request, cancellationToken), cancellationToken);

    /// <inheritdoc/>
    public Task<ModelWeightFormats> DescribeWeightFormatsAsync(ModelLoadRequest request, CancellationToken cancellationToken)
        => Task.Run(() => _estimator.WeightFormats(request, cancellationToken), cancellationToken);

    /// <inheritdoc/>
    public Task<IReadOnlyList<DeviceMemory>> GetDeviceMemoryAsync(IReadOnlyList<string> disabledFeatures, CancellationToken cancellationToken)
    {
        var disabled = new HashSet<string>(disabledFeatures, StringComparer.OrdinalIgnoreCase);
        return Task.Run<IReadOnlyList<DeviceMemory>>(() =>
        {
            var memory = new List<DeviceMemory>();
            var inUse = _selector.Gpus.Select(gpu => gpu.Effective.Id).ToHashSet(StringComparer.Ordinal);
            foreach (var option in _physical.Where(option => !option.IsGpu || option.GpuKey is null || inUse.Contains(option.Id)))
            {
                cancellationToken.ThrowIfCancellationRequested();
                IComputeDevice device;
                try
                {
                    device = Open(option, disabled);
                }
                catch (Exception error) when (error is not OutOfMemoryException)
                {
                    // A device that cannot be opened cannot take a model either.
                    continue;
                }

                var info = device.GetMemoryInfo();
                long total = info.Total > 0 ? info.Total : option.TotalMemory;
                long capacity = Math.Min(total, info.Capacity);
                memory.Add(new DeviceMemory(option.Id, option.Name, option.IsGpu, total, Math.Min(total, info.Available), info.IsExact, capacity));
                if (option.GpuKey is not null)
                {
                    // The same GPU through another provider shares its memory: no second context is opened just to ask.
                    foreach (var sibling in _physical.Where(other => other.GpuKey == option.GpuKey && other.Id != option.Id))
                    {
                        memory.Add(new DeviceMemory(sibling.Id, sibling.Name, true, total, Math.Min(total, info.Available), info.IsExact, capacity));
                    }
                }
            }

            return memory;
        }, cancellationToken);
    }

    /// <summary>Unloads every model, then closes the devices.</summary>
    public void Dispose()
    {
        Host.ModelsChanged -= OnHostChanged;
        List<IEngineModel> models;
        lock (_lock)
        {
            models = [.. _models.Values];
            _models.Clear();
        }

        foreach (var model in models)
        {
            model.Dispose();
        }

        Host.Dispose();
        _changed.Dispose();
        _estimator.Dispose();
        _pool.Dispose();
    }

    private async Task RemoveAsync(string id, bool force)
    {
        try
        {
            await Host.UnregisterAsync(id, s_drainTimeout, force).ConfigureAwait(false);
        }
        finally
        {
            IEngineModel? model;
            lock (_lock)
            {
                _models.Remove(id, out model);
            }

            model?.Dispose();

            // Not throttled: a removed model must leave the pickers at once (a pending throttled call fires later, harmlessly).
            ModelsChanged?.Invoke(this, EventArgs.Empty);
        }
    }

    /// <summary>Cancels a load in progress so <see cref="RemoveAsync"/> does not wait for it to finish.</summary>
    private void AbandonLoad(string id)
    {
        lock (_lock)
        {
            if (_models.TryGetValue(id, out var model))
            {
                model.Abandon.Cancel();
            }
        }
    }

    private ModelRegistration? Resolve(string name)
    {
        if (_resolve?.Invoke(name) is not { } request)
        {
            return null;
        }

        var model = CreateModel(request with { Id = name }, pinned: false, out var registration);
        lock (_lock)
        {
            _models.TryAdd(name, model);
        }

        return registration;
    }

    private IEngineModel Add(ModelLoadRequest request, bool pinned)
    {
        var model = CreateModel(request, pinned, out var registration);
        lock (_lock)
        {
            _models[request.Id] = model;
        }

        Host.Register(registration);
        return model;
    }

    private IEngineModel CreateModel(ModelLoadRequest request, bool pinned, out ModelRegistration registration)
    {
        var device = ResolveDevice(request.Settings.DeviceId);
        try
        {
            return BuildModel(request, device, pinned, out registration);
        }
        catch (Exception error) when (_selector.Failover(device, error) is { } fallback)
        {
            // Opening the device failed (the CUDA driver does not initialize): the GPU's next provider takes over.
            return BuildModel(request, fallback, pinned, out registration);
        }
    }

    private IEngineModel BuildModel(ModelLoadRequest request, ComputeDeviceOption device, bool pinned, out ModelRegistration registration)
    {
        var idle = _residency().IdleUnloadMinutes;
        if (TaskKindClients.Handles(request.Kind) || request.Kind is ModelKindCatalog.Embeddings or ModelKindCatalog.Reranker)
        {
            var task = new EngineTaskModel(request, device, Host, pinned, RaiseChanged);
            registration = new ModelRegistration
            {
                Name = request.Id,
                Loader = new ObservedLoader(request.Kind switch
                {
                    ModelKindCatalog.Embeddings => CreateRetrievalLoader(request, device, task, reranker: false),
                    ModelKindCatalog.Reranker => CreateRetrievalLoader(request, device, task, reranker: true),
                    _ => CreateTaskLoader(request, device, task),
                }, task.OnInstanceLoaded),
                Pinned = pinned,
                IdleTimeout = idle > 0 ? TimeSpan.FromMinutes(idle) : null,
                Description = request.Path,
            };
            return task;
        }

        var model = new EngineLoadedModel(request, device, Host, pinned, request.Capabilities, RaiseChanged);
        var loader = CreateLoader(request, device, model);
        registration = new ModelRegistration
        {
            Name = request.Id,
            Loader = new ObservedLoader(loader, model.OnInstanceLoaded),
            Pinned = pinned,
            IdleTimeout = idle > 0 ? TimeSpan.FromMinutes(idle) : null,
            Description = request.Path,
        };
        return model;
    }

    private ComputeDeviceOption ResolveDevice(string? deviceId)
    {
        var devices = Devices;
        var chosen = deviceId is not null && devices.FirstOrDefault(device => device.Id == deviceId) is { } named ? named : devices.FirstOrDefault()
            ?? throw new InvalidOperationException("No execution provider is available on this machine.");

        // A provider that cannot be used on the GPU (CUDA with an old driver) hands the load to the GPU's next provider.
        return _selector.Usable(chosen);
    }

    /// <summary>The GPUs a model can be split over: one entry per GPU, all of the first GPU's provider.</summary>
    private List<ComputeDeviceOption> SplittableGpus()
    {
        var gpus = _selector.Gpus.Select(gpu => gpu.Effective).ToList();
        return gpus.Count == 0 ? [] : [.. gpus.Where(device => device.Provider == gpus[0].Provider)];
    }

    private TaskModelLoader CreateRetrievalLoader(ModelLoadRequest request, ComputeDeviceOption device, EngineTaskModel model, bool reranker)
    {
        var disabled = new HashSet<string>(request.DisabledFeatures, StringComparer.OrdinalIgnoreCase);
        IReadOnlyList<DeviceSlot> slots = device.IsSplit ? SplitSlots(disabled) : [new DeviceSlot { Device = Open(device, disabled) }];
        var options = new TaskLoadOptions();
        TextEmbeddingLoadDefaults.Apply(options);
        options.ModelPath = request.Path;
        options.Devices = [.. slots];
        options.DisabledFeatures = disabled;
        options.AllowReducedPrecision = request.Settings.AllowReducedPrecision;
        options.Log = model.OnLoadMessage;
        options.Progress = new SynchronousProgress<LoadProgress>(model.OnLoadProgress);
        return new TaskModelLoader(reranker ? Sharplight.Models.ModelKinds.TextReranking : Sharplight.Models.ModelKinds.TextEmbedding, options,
            [.. slots.Select(slot => new ComputeDeviceMemoryDomain(slot.Device))], _engine.Services, model.Abandon.Token);
    }

    private TaskModelLoader CreateTaskLoader(ModelLoadRequest request, ComputeDeviceOption device, EngineTaskModel model)
    {
        var settings = request.Settings;
        var disabled = new HashSet<string>(request.DisabledFeatures, StringComparer.OrdinalIgnoreCase);
        IReadOnlyList<DeviceSlot> slots = device.IsSplit ? SplitSlots(disabled) : [new DeviceSlot { Device = Open(device, disabled) }];
        var cpu = _physical.FirstOrDefault(candidate => !candidate.IsGpu);
        bool offload = settings.CpuOffload && device.IsGpu && cpu is not null;
        var options = new TaskLoadOptions
        {
            ModelPath = request.Path,
            Devices = [.. slots],
            Fallback = offload ? new DeviceSlot { Device = Open(cpu!, disabled) } : null,
            ExpertsOnDisk = offload && settings.ExpertsOnDisk,
            ContextLength = settings.ContextLength,
            MaxBatchTokens = settings.BatchTokens,
            Auxiliary = (request.Auxiliary ?? []).ToDictionary(entry => entry.Role, entry => entry.Path, StringComparer.Ordinal),
            DisabledFeatures = CompileDisabledFeatures(disabled, settings),
            BlockScaledFp8 = settings.BlockScaledFp8,
            AllowReducedPrecision = settings.AllowReducedPrecision,
            Log = model.OnLoadMessage,
            Progress = new SynchronousProgress<LoadProgress>(model.OnLoadProgress),
        };
        return new TaskModelLoader(request.Kind, options, [.. slots.Select(slot => new ComputeDeviceMemoryDomain(slot.Device))], _engine.Services, model.Abandon.Token);
    }

    private LlmChatModelLoader CreateLoader(ModelLoadRequest request, ComputeDeviceOption device, EngineLoadedModel model)
    {
        var settings = request.Settings;
        var disabled = new HashSet<string>(request.DisabledFeatures, StringComparer.OrdinalIgnoreCase);
        IReadOnlyList<DeviceSlot> slots = device.IsSplit ? SplitSlots(disabled) : [new DeviceSlot { Device = Open(device, disabled) }];
        var cpu = _physical.FirstOrDefault(candidate => !candidate.IsGpu);
        bool offload = settings.CpuOffload && device.IsGpu && cpu is not null;
        var load = new LlmLoadOptions
        {
            Devices = [.. slots],
            Fallback = offload ? new DeviceSlot { Device = Open(cpu!, disabled) } : null,
            ExpertsOnDisk = offload && settings.ExpertsOnDisk,
            ContextLength = settings.ContextLength,
            MaxBatchTokens = settings.BatchTokens,
            KvCacheType = DataType.Get(settings.KvCacheType),
            DisabledFeatures = CompileDisabledFeatures(disabled, settings),
            BlockScaledFp8 = settings.BlockScaledFp8,
            AllowReducedPrecision = settings.AllowReducedPrecision,
            Auxiliary = request.Projector is { } projector ? new Dictionary<string, string> { [AuxiliaryRoles.Mmproj] = projector } : [],
            Log = model.OnLoadMessage,
            Progress = new SynchronousProgress<LoadProgress>(model.OnLoadProgress),
            CancellationToken = model.Abandon.Token,
        };

        var id = request.Id;
        var defaults = new EngineOptions();
        var chat = new ChatModelOptions
        {
            Slots = Math.Max(1, settings.Slots),
            Engine = new EngineOptions
            {
                RequestFinished = finished => OnRequestFinished(id, device, finished),
                StepFloor = settings.StepFloor ?? defaults.StepFloor,
                BatchTokensPerSequence = settings.StepRowsPerSequence is > 0 and var rows ? rows : defaults.BatchTokensPerSequence,
                KvAdmission = settings.KvAdmission ?? defaults.KvAdmission,
            },
            Speculator = settings.NgramDraftTokens > 0 ? _ => new NgramSpeculator(settings.NgramDraftTokens) : null,
            MtpDraftTokens = settings.MtpDraftTokens,
            Service = new ModelServiceOptions { DefaultMaxTokens = 0, ImageDecoder = ChatRequestMapper.DecodeImage },
        };
        return new LlmChatModelLoader(_engine.Loader, id, request.Path, load, chat);
    }

    /// <summary>
    /// The device features the compilations of a load must not use: the disabled ones, plus the FP4 GEMM (W4A4) when the load turns FP4
    /// activations off. The devices themselves are opened with <paramref name="disabled"/> only, so they stay shared with other loads.
    /// </summary>
    private static HashSet<string> CompileDisabledFeatures(HashSet<string> disabled, LoadSettings settings)
    {
        var features = new HashSet<string>(disabled, StringComparer.OrdinalIgnoreCase);
        if (!settings.Fp4Activations)
        {
            features.Add(DeviceFeatures.BlockScaledFp4);
        }

        return features;
    }

    /// <summary>
    /// Every GPU in pipeline order, each weighted by the memory it has free beyond the reserve (the share of the model it gets, as
    /// <see cref="FitPlanner"/> assumes).
    /// </summary>
    private List<DeviceSlot> SplitSlots(IReadOnlySet<string> disabled)
    {
        long reserve = Math.Max(0, _residency().ReserveMiB) * (1L << 20);
        return [.. SplittableGpus().Select(option =>
        {
            var device = Open(option, disabled);
            return new DeviceSlot { Device = device, Weight = FitPlanner.SplitWeight(device.GetMemoryInfo().Available, reserve) };
        })];
    }

    private IComputeDevice Open(ComputeDeviceOption option, IReadOnlySet<string> disabled)
    {
        var (provider, info) = _devices[option.Id];
        var options = new DeviceOptions { UseCompilationCache = _useCompilationCache() };
        foreach (var feature in disabled)
        {
            options.DisabledFeatures.Add(feature);
        }

        return _pool.Open(provider, info, options);
    }

    private void OnHostChanged(object? sender, EventArgs e)
    {
        List<IEngineModel> models;
        lock (_lock)
        {
            models = [.. _models.Values];
        }

        foreach (var model in models)
        {
            model.OnHostChanged();
        }

        RaiseChanged();
    }

    /// <summary>
    /// Raises <see cref="ModelsChanged"/>, throttled: host state changes, lease releases and load progress would otherwise make every
    /// listening view model refresh many times a second.
    /// </summary>
    private void RaiseChanged() => _changed.Request();

    private void OnRequestFinished(string model, ComputeDeviceOption device, FinishedRequest finished)
    {
        var source = finished.Tenant == ChatRequestMapper.StudioTenant ? RequestSource.Chat : RequestSource.Api;
        RequestCompleted?.Invoke(this, new RequestRecord
        {
            Model = model,
            Source = source,
            Device = device.Name,
            Stats = ChatRequestMapper.ToStats(finished),
        });
    }
}
