using Sharplight.Hub;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Llm.Serving.Speculative;
using Sharplight.Runtime;
using Sharplight.Samples;

namespace Sharplight.Server;

/// <summary>
/// Creates the <see cref="ModelHost"/> and registers every model on it. Devices come from a shared pool, so several models
/// on one GPU use one opened device.
/// </summary>
internal sealed class ModelHostBuilder
{
    private const string CpuDevice = "cpu";

    private readonly ServeSettings _settings;
    private readonly ServerPlan _plan;
    private readonly DevicePool _devices;
    private readonly DeviceOptions _gpuOptions;
    private IComputeDevice? _cpuDevice;

    public ModelHostBuilder(ServeSettings settings, ServerPlan plan, DevicePool devices)
    {
        _settings = settings;
        _plan = plan;
        _devices = devices;
        _gpuOptions = new DeviceOptions { MemoryBudget = plan.BudgetBytes };
        foreach (var feature in settings.Runtime.DisabledFeatures)
        {
            _gpuOptions.DisabledFeatures.Add(feature);
        }
    }

    public ModelHost Build()
    {
        var host = new ModelHost(new ModelHostOptions
        {
            ReserveBytes = _plan.ReserveBytes,
            MaxLoadedModels = _plan.MaxLoadedModels,
            IdleTimeout = ToTimeout(_plan.IdleTimeoutSeconds),
            Log = Console.Error.WriteLine,
            Resolve = _plan.OnDemand ? ResolveStored : null,
        });

        foreach (var spec in _plan.Models)
        {
            host.Register(CreateRegistration(spec));
        }

        if (_plan.OnDemand)
        {
            // Listed in /v1/models right away; models stored later are found by name on their first request.
            foreach (var manifest in new ModelStore(_settings.StoreRoot).List().Where(manifest => !host.Contains(manifest.Name)))
            {
                host.Register(CreateRegistration(OnDemandSpec(manifest.Name, manifest.PackagePath)));
            }
        }

        return host;
    }

    /// <summary>The registration of a stored model requested by name (<c>owner/name[:tag]</c> or <c>alias[:tag]</c>), or null.</summary>
    private ModelRegistration? ResolveStored(string name)
    {
        try
        {
            return new ModelStore(_settings.StoreRoot).Find(name) is { } manifest && File.Exists(manifest.PackagePath)
                ? CreateRegistration(OnDemandSpec(name, manifest.PackagePath))
                : null;
        }
        catch (HubException)
        {
            return null;   // ambiguous name
        }
    }

    private static ModelSpec OnDemandSpec(string name, string path) => new() { Name = name, Path = path, Lazy = true };

    private static TimeSpan? ToTimeout(double? seconds)
    {
        return seconds is > 0 ? TimeSpan.FromSeconds(seconds.Value) : null;
    }

    private ModelRegistration CreateRegistration(ModelSpec spec)
    {
        var loader = new LlmChatModelLoader(spec.Name, spec.Path, CreateLoadOptions(spec), CreateChatOptions(spec));
        return new ModelRegistration
        {
            Name = spec.Name,
            Loader = loader,
            LoadAtStartup = !(spec.Lazy ?? _plan.LazyByDefault),
            Pinned = spec.Pin ?? false,
            IdleTimeout = ToTimeout(spec.Idle),
            Description = spec.Path,
        };
    }

    private LlmLoadOptions CreateLoadOptions(ModelSpec spec)
    {
        var runtime = _settings.Runtime;
        var deviceName = spec.Device ?? (runtime.UseCpu ? CpuDevice : runtime.DeviceIndex.ToString(System.Globalization.CultureInfo.InvariantCulture));
        bool onCpu = deviceName.Equals(CpuDevice, StringComparison.OrdinalIgnoreCase);
        bool offload = !onCpu && (spec.Offload ?? runtime.CpuOffload);
        var mmproj = spec.Mmproj ?? _settings.MmprojPath;

        // With several models the log lines need a prefix to tell them apart.
        bool prefixLog = _plan.LazyByDefault || _plan.Models.Count > 1;
        Action<string> log = prefixLog ? message => Console.Error.WriteLine($"[{spec.Name}] {message}") : Console.Error.WriteLine;

        return new LlmLoadOptions
        {
            Auxiliary = mmproj is not null ? [("mmproj", mmproj)] : [],
            Devices = [new DeviceSlot { Device = OpenDevice(deviceName) }],
            Fallback = offload ? new DeviceSlot { Device = GetCpuDevice() } : null,
            ContextLength = spec.Ctx ?? runtime.ContextLength,
            MaxBatchTokens = spec.Batch ?? runtime.BatchTokens,
            KvCacheType = runtime.KvCacheType,
            AllowReducedPrecision = runtime.ReducedPrecision,
            Log = log,
        };
    }

    private ChatModelOptions CreateChatOptions(ModelSpec spec)
    {
        var templatePath = spec.Template ?? _settings.ChatTemplatePath;
        return new ChatModelOptions
        {
            Slots = spec.Slots ?? _settings.Slots,
            KvCacheTokens = _settings.KvCacheTokens,
            Engine = new EngineOptions
            {
                MaxQueue = _settings.MaxQueue, PrefillChunk = _settings.PrefillChunk, MaxBatchTokens = _settings.StepTokens ?? 0,
                SpeculationMaxBatch = _settings.SpeculationMaxBatch,
            },
            Speculator = _settings.NgramDraftTokens > 0 ? _ => new NgramSpeculator(_settings.NgramDraftTokens) : null,
            MtpDraftTokens = _settings.MtpDraftTokens,
            Service = new ModelServiceOptions { ChatTemplate = templatePath is not null ? ReadTemplate(templatePath) : null },
        };
    }

    private static string ReadTemplate(string path)
    {
        if (!File.Exists(path))
        {
            throw new SampleException($"Chat template '{path}' does not exist.");
        }

        return File.ReadAllText(path);
    }

    private IComputeDevice OpenDevice(string deviceName)
    {
        bool cpu = deviceName.Equals(CpuDevice, StringComparison.OrdinalIgnoreCase);
        if (!cpu && !int.TryParse(deviceName, out _))
        {
            throw new SampleException($"Device '{deviceName}' is neither a device index nor 'cpu'.");
        }

        var provider = SampleRuntime.SelectProvider(cpu);
        var info = SampleRuntime.GetDevice(provider, cpu ? 0 : int.Parse(deviceName, System.Globalization.CultureInfo.InvariantCulture));
        return _devices.Open(provider, info, cpu ? new DeviceOptions() : _gpuOptions);
    }

    private IComputeDevice GetCpuDevice()
    {
        return _cpuDevice ??= OpenDevice(CpuDevice);
    }
}
