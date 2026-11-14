using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Llm.Serving.Speculative;
using Sharplight.Runtime;
using Sharplight.Studio.Models;
using Sharplight.Tensors;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// <see cref="IModelRuntime"/> on the Sharplight engine's <see cref="ModelHost"/>, the same hosting layer <c>sharplight-server</c>
/// uses: several models stay resident at once within the device memory budget (least recently used idle ones are evicted when
/// another does not fit, idle ones unload after a timeout and reload on demand), devices are opened once through a
/// <see cref="DevicePool"/>, and the HTTP endpoints serve the same host, so app chats and API requests are batched together.
/// </summary>
internal sealed class EngineModelRuntime : IModelRuntime
{
    private static readonly TimeSpan s_drainTimeout = TimeSpan.FromSeconds(5);

    private readonly DevicePool _pool = new();
    private readonly Dictionary<string, EngineLoadedModel> _models = new(StringComparer.Ordinal);
    private readonly Lock _lock = new();
    private readonly Dictionary<string, (IExecutionProvider Provider, DeviceInfo Info)> _devices;
    private readonly Func<ResidencySettings> _residency;
    private readonly Func<string, ModelLoadRequest?>? _resolve;

    /// <summary>Registers the providers and importers, enumerates the devices and creates the host.</summary>
    /// <param name="residency">The current residency settings (read when a model is added).</param>
    /// <param name="resolve">Turns a name requested by a chat or API client into a library model to load on demand, or null.</param>
    public EngineModelRuntime(Func<ResidencySettings> residency, Func<string, ModelLoadRequest?>? resolve = null)
    {
        EngineBootstrap.Register();
        var discovered = DeviceCatalog.Discover();
        Devices = [.. discovered.Select(entry => entry.Option)];
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
    public async Task<ILoadedModel> LoadAsync(ModelLoadRequest request, IProgress<LoadStatus>? progress, CancellationToken cancellationToken)
    {
        if (Host.Contains(request.Id))
        {
            throw new InvalidOperationException($"A model named '{request.Id}' is already loaded.");
        }

        var model = Add(request, _residency().PinManualLoads);
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

    /// <summary>Unloads every model, then closes the devices.</summary>
    public void Dispose()
    {
        Host.ModelsChanged -= OnHostChanged;
        List<EngineLoadedModel> models;
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
            EngineLoadedModel? model;
            lock (_lock)
            {
                _models.Remove(id, out model);
            }

            model?.Dispose();
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

    private EngineLoadedModel Add(ModelLoadRequest request, bool pinned)
    {
        var model = CreateModel(request, pinned, out var registration);
        lock (_lock)
        {
            _models[request.Id] = model;
        }

        Host.Register(registration);
        return model;
    }

    private EngineLoadedModel CreateModel(ModelLoadRequest request, bool pinned, out ModelRegistration registration)
    {
        var device = ResolveDevice(request.Settings.DeviceId);
        var idle = _residency().IdleUnloadMinutes;

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

    private LlmChatModelLoader CreateLoader(ModelLoadRequest request, ComputeDeviceOption device, EngineLoadedModel model)
    {
        var settings = request.Settings;
        var disabled = new HashSet<string>(request.DisabledFeatures, StringComparer.OrdinalIgnoreCase);
        var primary = Open(device, disabled);
        var cpu = Devices.FirstOrDefault(candidate => !candidate.IsGpu);
        bool offload = settings.CpuOffload && device.IsGpu && cpu is not null;
        var load = new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = primary }],
            Fallback = offload ? new DeviceSlot { Device = Open(cpu!, disabled) } : null,
            ContextLength = settings.ContextLength,
            MaxBatchTokens = settings.BatchTokens,
            KvCacheType = DataType.Get(settings.KvCacheType),
            DisabledFeatures = disabled,
            AllowReducedPrecision = settings.AllowReducedPrecision,
            Auxiliary = request.Projector is { } projector ? [("mmproj", projector)] : [],
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
        return new LlmChatModelLoader(id, request.Path, load, chat);
    }

    private IComputeDevice Open(ComputeDeviceOption option, IReadOnlySet<string> disabled)
    {
        var (provider, info) = _devices[option.Id];
        var options = new DeviceOptions();
        foreach (var feature in disabled)
        {
            options.DisabledFeatures.Add(feature);
        }

        return _pool.Open(provider, info, options);
    }

    private void OnHostChanged(object? sender, EventArgs e)
    {
        List<EngineLoadedModel> models;
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

    private void RaiseChanged() => ModelsChanged?.Invoke(this, EventArgs.Empty);

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
