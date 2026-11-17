using Sharplight.Hosting;
using Sharplight.Hosting.Fit;
using Sharplight.Hosting.Loaders;
using Sharplight.Llm;
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
    private readonly Dictionary<string, IEngineModel> _models = new(StringComparer.Ordinal);
    private readonly Lock _lock = new();
    private readonly Dictionary<string, (IExecutionProvider Provider, DeviceInfo Info)> _devices;
    private readonly Func<ResidencySettings> _residency;
    private readonly Func<bool> _useCompilationCache;
    private readonly Func<string, ModelLoadRequest?>? _resolve;
    private readonly IReadOnlyList<ComputeDeviceOption> _physical;
    private readonly FootprintEstimator _estimator;
    private readonly EngineServices _engine;
    private readonly UpdateThrottle _changed;

    /// <summary>Enumerates the devices and creates the host.</summary>
    /// <param name="engine">The engine services (loader, providers).</param>
    /// <param name="residency">The current residency settings (read when a model is added).</param>
    /// <param name="resolve">Turns a name requested by a chat or API client into a library model to load on demand, or null.</param>
    /// <param name="useCompilationCache">Whether devices use the per-user artifact store (read when a device is opened); default yes.</param>
    public EngineModelRuntime(EngineServices engine, Func<ResidencySettings> residency, Func<string, ModelLoadRequest?>? resolve = null, Func<bool>? useCompilationCache = null)
    {
        _engine = engine;
        _estimator = new FootprintEstimator(engine.Loader);
        _useCompilationCache = useCompilationCache ?? (() => true);
        var discovered = DeviceCatalog.Discover(engine.Providers);
        _physical = [.. discovered.Select(entry => entry.Option)];
        // Only GPUs of the preferred provider are split over: on a Mac the Metal and the Vulkan (MoltenVK) device are the same GPU.
        var gpus = _physical.Where(device => device.IsGpu).ToList();
        var splittable = gpus.Where(device => device.Provider == gpus[0].Provider).ToList();
        Devices = splittable.Count > 1 ? [.. gpus, ComputeDeviceOption.AllGpus(splittable), .. _physical.Where(device => !device.IsGpu)] : _physical;
        _devices = discovered.ToDictionary(entry => entry.Option.Id, entry => (entry.Provider, entry.Info));
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
    public IReadOnlyList<ComputeDeviceOption> Devices { get; }

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
        var model = await Task.Run(() => Add(request, pinned), cancellationToken).ConfigureAwait(false);
        model.LoadListener = progress;
        try
        {
            progress?.Report(LoadStatus.Message($"Loading {Path.GetFileName(request.Path)} on {model.Device.Name}"));
            await Host.LoadAsync(request.Id, cancellationToken).ConfigureAwait(false);
            return model;
        }
        catch (Exception)
        {
            // Cancelled or failed: abandon a load still running and forget the model.
            AbandonLoad(request.Id);
            _ = RemoveAsync(request.Id, force: true);
            throw;
        }
        finally
        {
            model.LoadListener = null;
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
    public Task<IReadOnlyList<DeviceMemory>> GetDeviceMemoryAsync(IReadOnlyList<string> disabledFeatures, CancellationToken cancellationToken)
    {
        var disabled = new HashSet<string>(disabledFeatures, StringComparer.OrdinalIgnoreCase);
        return Task.Run<IReadOnlyList<DeviceMemory>>(() =>
        {
            var memory = new List<DeviceMemory>();
            foreach (var option in _physical)
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
                memory.Add(new DeviceMemory(option.Id, option.Name, option.IsGpu, total, Math.Min(total, info.Available), info.IsExact));
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
        var idle = _residency().IdleUnloadMinutes;
        if (TaskKindClients.Handles(request.Kind))
        {
            var task = new EngineTaskModel(request, device, Host, pinned, RaiseChanged);
            registration = new ModelRegistration
            {
                Name = request.Id,
                Loader = new ObservedLoader(CreateTaskLoader(request, device, task), task.OnInstanceLoaded),
                Pinned = pinned,
                IdleTimeout = idle > 0 ? TimeSpan.FromMinutes(idle) : null,
                Description = request.Path,
            };
            return task;
        }

        var model = new EngineLoadedModel(request, device, Host, pinned, request.Capabilities, OnVisionFinished, RaiseChanged);
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
        if (deviceId is not null && Devices.FirstOrDefault(device => device.Id == deviceId) is { } chosen)
        {
            return chosen;
        }

        return Devices.FirstOrDefault() ?? throw new InvalidOperationException("No execution provider is available on this machine.");
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
            ContextLength = settings.ContextLength,
            MaxBatchTokens = settings.BatchTokens,
            Auxiliary = (request.Auxiliary ?? []).ToDictionary(entry => entry.Role, entry => entry.Path, StringComparer.Ordinal),
            DisabledFeatures = disabled,
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
            ContextLength = settings.ContextLength,
            MaxBatchTokens = settings.BatchTokens,
            KvCacheType = DataType.Get(settings.KvCacheType),
            DisabledFeatures = disabled,
            AllowReducedPrecision = settings.AllowReducedPrecision,
            Auxiliary = request.Projector is { } projector ? new Dictionary<string, string> { [AuxiliaryRoles.Mmproj] = projector } : [],
            Log = model.OnLoadMessage,
            Progress = new SynchronousProgress<LoadProgress>(model.OnLoadProgress),
            CancellationToken = model.Abandon.Token,
        };

        var id = request.Id;
        var chat = new ChatModelOptions
        {
            Slots = Math.Max(1, settings.Slots),
            Engine = new EngineOptions { RequestFinished = finished => OnRequestFinished(id, device, finished) },
            Speculator = settings.NgramDraftTokens > 0 ? _ => new NgramSpeculator(settings.NgramDraftTokens) : null,
            MtpDraftTokens = settings.MtpDraftTokens,
            Service = new ModelServiceOptions { DefaultMaxTokens = 0 },
        };
        return new LlmChatModelLoader(_engine.Loader, id, request.Path, load, chat);
    }

    /// <summary>
    /// Every GPU in pipeline order, each weighted by the memory it has free beyond the reserve (the share of the model it gets, as
    /// <see cref="FitPlanner"/> assumes).
    /// </summary>
    private List<DeviceSlot> SplitSlots(IReadOnlySet<string> disabled)
    {
        long reserve = Math.Max(0, _residency().ReserveMiB) * (1L << 20);
        return [.. _physical.Where(option => option.IsGpu).Select(option =>
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

    private void OnVisionFinished(EngineLoadedModel model, MessageStats stats)
    {
        RequestCompleted?.Invoke(this, new RequestRecord { Model = model.Id, Source = RequestSource.Chat, Device = model.Device.Name, Stats = stats });
    }
}
