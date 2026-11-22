using System.Collections.Concurrent;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Sharplight.Hosting.Fit;
using Sharplight.Hosting.Loaders;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Runtime;

namespace Sharplight.Hosting;

/// <summary>
/// The default <see cref="IModelHostManager"/>: builds the <see cref="ModelHost"/> from <see cref="ModelHostingOptions"/> (registering every
/// configured model, and the on-demand models of the <see cref="IHostedModelResolver"/> when enabled) and owns it. Dispose (normally the
/// container does) to unload every model.
/// </summary>
public sealed class ModelHostManager : IModelHostManager, IDisposable
{
    private readonly ModelHostingOptions _options;
    private readonly IModelPlacementPolicy _placement;
    private readonly IReadOnlyList<IHostedModelLoaderFactory> _factories;
    private readonly IServiceProvider _services;
    private readonly IHostedModelResolver? _resolver;
    private readonly ConcurrentDictionary<string, HostedModelSettings> _settings = new(StringComparer.Ordinal);

    /// <summary>Creates the host and registers the configured models (nothing is loaded yet).</summary>
    /// <param name="options">The hosting options.</param>
    /// <param name="devices">The devices.</param>
    /// <param name="placement">The placement policy.</param>
    /// <param name="factories">The loader factories (the last registered is asked first).</param>
    /// <param name="services">The application's services (model loader, task clients, optional <see cref="IHostedModelResolver"/> and logging).</param>
    /// <exception cref="ArgumentException">A configured model is invalid or no factory loads it.</exception>
    /// <exception cref="InvalidOperationException">Two configured models have the same name.</exception>
    public ModelHostManager(IOptions<ModelHostingOptions> options, DeviceInventory devices, IModelPlacementPolicy placement,
        IEnumerable<IHostedModelLoaderFactory> factories, IServiceProvider services)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(devices);
        ArgumentNullException.ThrowIfNull(placement);
        ArgumentNullException.ThrowIfNull(factories);
        ArgumentNullException.ThrowIfNull(services);
        _options = options.Value;
        Devices = devices;
        _placement = placement;
        _factories = [.. factories.Reverse()];
        _services = services;
        _resolver = _options.OnDemand ? services.GetService<IHostedModelResolver>() : null;
        var logger = services.GetService<ILoggerFactory>()?.CreateLogger<ModelHostManager>();
        Host = new ModelHost(new ModelHostOptions
        {
            ReserveBytes = _options.Reserve,
            MaxLoadedModels = _options.MaxLoadedModels,
            IdleTimeout = _options.IdleTimeout,
            Log = _options.Log ?? (logger is null ? null : message => logger.LogInformation("{Message}", message)),
            Resolve = _resolver is null ? null : ResolveOnDemand,
            Time = services.GetService<TimeProvider>() ?? TimeProvider.System,
        });

        foreach (var model in _options.Models)
        {
            Register(model);
        }

        foreach (var model in _resolver?.List() ?? [])
        {
            if (!Host.Contains(model.Name))
            {
                Register(model);
            }
        }
    }

    /// <inheritdoc/>
    public ModelHost Host { get; }

    /// <inheritdoc/>
    public DeviceInventory Devices { get; }

    /// <inheritdoc/>
    public HostedModelSettings Register(HostedModelOptions model)
    {
        var (settings, registration) = CreateRegistration(model);
        Host.Register(registration);
        _settings[settings.Name] = settings;
        return settings;
    }

    /// <inheritdoc/>
    public HostedModelSettings? Find(string name) => _settings.TryGetValue(name, out var settings) && Host.Contains(name) ? settings : null;

    /// <inheritdoc/>
    public Task LoadAsync(string name, CancellationToken cancellationToken = default) => Host.LoadAsync(name, cancellationToken);

    /// <inheritdoc/>
    public Task<bool> UnloadAsync(string name, TimeSpan? drainTimeout = null, bool force = false, CancellationToken cancellationToken = default)
        => Host.UnloadAsync(name, drainTimeout, force, cancellationToken);

    /// <inheritdoc/>
    public async Task<bool> RemoveAsync(string name, TimeSpan? drainTimeout = null, bool force = false, CancellationToken cancellationToken = default)
    {
        bool removed = await Host.UnregisterAsync(name, drainTimeout, force, cancellationToken);
        if (removed)
        {
            _settings.TryRemove(name, out _);
        }

        return removed;
    }

    /// <inheritdoc/>
    public HostStatus GetStatus()
    {
        var models = Host.List();
        var instances = Host.LoadedInstances().ToDictionary(entry => entry.Name, entry => entry.Instance, StringComparer.Ordinal);
        var modelStatus = models.Select(model => ToStatus(model, instances.GetValueOrDefault(model.Name))).ToList();
        return new HostStatus
        {
            Timestamp = Host.Options.Time.GetUtcNow(),
            ReserveBytes = _options.Reserve,
            MaxLoadedModels = _options.MaxLoadedModels,
            Devices = DeviceStatuses(modelStatus),
            Models = modelStatus,
        };
    }

    /// <inheritdoc/>
    public FitReport PlanFit(ModelFootprint footprint, string? device = null, bool cpuOffload = false, bool expertsOnDisk = false)
    {
        var memory = CurrentMemory();
        var targets = new List<DeviceMemory>();
        foreach (var part in DeviceSpecifications.Split(device ?? _options.Defaults.Device))
        {
            if (FindTarget(memory, part) is { } target && !targets.Contains(target))
            {
                targets.Add(target);
            }
        }

        var residents = Host.List()
            .Where(model => model.State == ModelState.Loaded)
            .Select(model => new ResidentUsage(model.Name, model.Devices ?? [], Math.Max(model.ResidentBytes, model.EstimatedBytes),
                !model.Pinned && model.ActiveLeases == 0, model.LastUsed))
            .ToList();
        return FitPlanner.Plan(new FitQuery
        {
            Footprint = footprint,
            Targets = targets,
            SystemMemory = Devices.Cpu is { } cpu ? memory.First(entry => entry.Id == cpu.Id) : null,
            ReserveBytes = _options.Reserve,
            Residents = residents,
            CpuOffload = cpuOffload,
            ExpertsOnDisk = expertsOnDisk,
        });
    }

    /// <summary>
    /// The memory entry of one part of a device specification: a pool id as listed, else the inventory device it names; <c>auto</c> is the
    /// GPU with the most free memory (else the CPU).
    /// </summary>
    private DeviceMemory? FindTarget(List<DeviceMemory> memory, string part)
    {
        if (memory.FirstOrDefault(entry => entry.Id.Equals(part, StringComparison.OrdinalIgnoreCase)) is { } listed)
        {
            return listed;
        }

        if (Devices.Resolve(part)[0] is { } named)
        {
            return memory.First(entry => entry.Id == named.Id);
        }

        return memory.Where(entry => entry.IsGpu).MaxBy(entry => entry.FreeBytes) ?? memory.FirstOrDefault(entry => !entry.IsGpu);
    }

    /// <summary>Unloads every model.</summary>
    public void Dispose() => Host.Dispose();

    private (HostedModelSettings Settings, ModelRegistration Registration) CreateRegistration(HostedModelOptions model)
    {
        var settings = HostedModelSettings.Resolve(model, _options.Defaults, _options.Lazy);
        var factory = _factories.FirstOrDefault(candidate => candidate.CanLoad(settings))
            ?? throw new ArgumentException($"No loader is registered for model '{settings.Name}' of kind {settings.Kind}.", nameof(model));
        var loader = factory.CreateLoader(new HostedModelLoadContext(settings, _options, _placement, _services));
        var registration = new ModelRegistration
        {
            Name = settings.Name,
            Loader = loader,
            LoadAtStartup = !settings.Lazy,
            Pinned = settings.Pinned,
            IdleTimeout = settings.IdleTimeout,
            Description = settings.Description,
        };
        return (settings, registration);
    }

    private ModelRegistration? ResolveOnDemand(string name)
    {
        try
        {
            if (_resolver!.Resolve(name) is not { } model)
            {
                return null;
            }

            // On-demand models always load on their first request.
            model.Lazy = true;
            var (settings, registration) = CreateRegistration(model);
            _settings[settings.Name] = settings;
            return registration;
        }
        catch (Exception error) when (error is ArgumentException or InvalidOperationException or IOException)
        {
            Host.Options.Log?.Invoke($"cannot resolve model '{name}': {error.Message}");
            return null;
        }
    }

    private HostedModelStatus ToStatus(ModelInfo model, LoadedModel? instance)
    {
        var metrics = (instance as IChatModel)?.Engine.GetMetrics();
        return new HostedModelStatus
        {
            Name = model.Name,
            Kind = model.Kind,
            Task = _settings.TryGetValue(model.Name, out var settings) ? settings.Task : null,
            State = model.State,
            Pinned = model.Pinned,
            Devices = model.Devices ?? [],
            ResidentBytes = model.ResidentBytes,
            EstimatedBytes = model.EstimatedBytes,
            ActiveLeases = model.ActiveLeases,
            QueueDepth = metrics?.QueueDepth ?? 0,
            ActiveSequences = metrics?.ActiveSequences ?? 0,
            Slots = metrics?.SlotCount ?? 0,
            TokensPerSecond = metrics?.RecentTokensPerSecond ?? 0,
            RequestsSubmitted = metrics?.RequestsSubmitted ?? 0,
            RequestsRejected = metrics?.RequestsRejected ?? 0,
            PromptTokens = metrics?.PromptTokens ?? 0,
            CompletionTokens = metrics?.CompletionTokens ?? 0,
            LoadedAt = model.LoadedAt,
            LastUsed = model.LastUsed,
            Error = model.Error,
        };
    }

    /// <summary>Free and total memory of every device and model-host pool, as the fit planner sees it.</summary>
    private List<DeviceMemory> CurrentMemory()
    {
        var memory = new List<DeviceMemory>();
        foreach (var device in Devices.Devices)
        {
            if (Devices.FindOpened(device) is { } opened)
            {
                var info = opened.GetMemoryInfo();
                memory.Add(new DeviceMemory(device.Id, device.Name, device.IsGpu, info.Total, info.Available, info.IsExact));
                continue;
            }

            memory.Add(new DeviceMemory(device.Id, device.Name, device.IsGpu, device.TotalBytes, device.TotalBytes, false));
        }

        foreach (var (domain, info) in Host.MemoryPools().Where(pool => memory.All(entry => entry.Id != pool.Domain.Id)))
        {
            memory.Add(new DeviceMemory(domain.Id, domain.Name, true, info.Total, info.Available, info.IsExact));
        }

        return memory;
    }

    private List<DeviceStatus> DeviceStatuses(IReadOnlyList<HostedModelStatus> models)
    {
        var loaded = models.Where(model => model.State == ModelState.Loaded && model.Devices.Count > 0).ToList();
        return [.. CurrentMemory().Select(device =>
        {
            var onDevice = loaded.Where(model => model.Devices.Contains(device.Id)).ToList();
            long modelBytes = onDevice.Sum(model => Math.Max(model.ResidentBytes, model.EstimatedBytes) / model.Devices.Count);
            return new DeviceStatus
            {
                Id = device.Id,
                Name = device.Name,
                IsGpu = device.IsGpu,
                IsOpen = device.IsExact || Devices.OpenedDevices.Any(opened => $"{opened.Info.Provider}:{opened.Info.Index}" == device.Id)
                    || Devices.Devices.All(known => known.Id != device.Id),
                TotalBytes = device.TotalBytes,
                UsedBytes = Math.Max(0, device.TotalBytes - device.FreeBytes),
                FreeBytes = device.FreeBytes,
                AvailableBytes = Math.Max(0, device.FreeBytes - _options.Reserve),
                ModelBytes = modelBytes,
                IsExact = device.IsExact,
                Models = [.. onDevice.Select(model => model.Name)],
            };
        })];
    }
}
