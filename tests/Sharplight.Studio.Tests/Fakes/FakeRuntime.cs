using Sharplight.Hosting.Fit;
using Sharplight.Runtime;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A runtime whose loads create <see cref="FakeModel"/>s.</summary>
internal sealed class FakeRuntime : IModelRuntime
{
    public static readonly ComputeDeviceOption Gpu = new("vulkan:0", "Test GPU", "Vulkan", true, 16L << 30, ["fp16", "subgroups"]);

    public static readonly ComputeDeviceOption Cpu = new("cpu", "CPU", "Cpu", false, 32L << 30, []);

    private readonly List<ILoadedModel> _models = [];

    private IReadOnlyList<ComputeDeviceOption> _devices = [Gpu, Cpu];

    public IReadOnlyList<ComputeDeviceOption> Devices
    {
        get => DevicesSource?.Invoke() ?? _devices;
        set => _devices = value;
    }

    /// <summary>Computes the devices on every read (a selector that follows the settings), when set.</summary>
    public Func<IReadOnlyList<ComputeDeviceOption>>? DevicesSource { get; set; }

    /// <summary>Computes the physical GPUs on every read, when set; none otherwise.</summary>
    public Func<IReadOnlyList<PhysicalGpu>>? GpuSource { get; set; }

    public IReadOnlyList<PhysicalGpu> Gpus => GpuSource?.Invoke() ?? [];

    /// <summary>The footprint estimates return (unless <see cref="Estimator"/> is set).</summary>
    public ModelFootprint Footprint { get; set; } = new(4L << 30, 1L << 30, 512L << 20);

    /// <summary>Computes the footprint of a request, when set.</summary>
    public Func<ModelLoadRequest, ModelFootprint>? Estimator { get; set; }

    /// <summary>Requests estimated so far.</summary>
    public List<ModelLoadRequest> Estimates { get; } = [];

    /// <summary>The devices' memory; by default the GPU minus what the models hold, and 24 GB of 32 GB system memory free.</summary>
    public Func<IReadOnlyList<DeviceMemory>>? Memory { get; set; }

    public IReadOnlyList<ILoadedModel> LoadedModels => _models;

    public List<ModelLoadRequest> Requests { get; } = [];

    /// <summary>Called with every model a load creates, before it is registered (to give it task clients).</summary>
    public Action<FakeModel>? Customize { get; set; }

    public event EventHandler? ModelsChanged;

    public event EventHandler<RequestRecord>? RequestCompleted;

    public FakeModel Add(FakeModel model)
    {
        _models.Add(model);
        ModelsChanged?.Invoke(this, EventArgs.Empty);
        return model;
    }

    public Task<ILoadedModel> LoadAsync(ModelLoadRequest request, IProgress<LoadStatus>? progress, CancellationToken cancellationToken)
    {
        Requests.Add(request);
        progress?.Report(new LoadStatus(0.5, "Compiling kernels 1 / 2"));
        var created = new FakeModel(request.Id) { Settings = request.Settings, Kind = request.Kind, Auxiliary = request.Auxiliary ?? [] };
        Customize?.Invoke(created);
        ILoadedModel model = Add(created);
        return Task.FromResult(model);
    }

    public Task UnloadAsync(string id)
    {
        _models.RemoveAll(model => model.Id == id);
        ModelsChanged?.Invoke(this, EventArgs.Empty);
        return Task.CompletedTask;
    }

    public IReadOnlyList<DeviceUsage> GetDeviceUsage() =>
        [new DeviceUsage(Gpu.Id, Gpu.Name, _models.Sum(model => model.ResidentBytes), Gpu.TotalMemory, true)];

    public Task<ModelFootprint> EstimateFootprintAsync(ModelLoadRequest request, CancellationToken cancellationToken)
    {
        lock (Estimates)
        {
            Estimates.Add(request);
        }

        return Task.FromResult(Estimator?.Invoke(request) ?? Footprint);
    }

    public Task<IReadOnlyList<DeviceMemory>> GetDeviceMemoryAsync(IReadOnlyList<string> disabledFeatures, CancellationToken cancellationToken) =>
        Task.FromResult(Memory?.Invoke() ??
        [
            new DeviceMemory(Gpu.Id, Gpu.Name, true, Gpu.TotalMemory, Gpu.TotalMemory - _models.Sum(model => model.ResidentBytes), true),
            new DeviceMemory(Cpu.Id, Cpu.Name, false, Cpu.TotalMemory, 24L << 30, true),
        ]);

    public void RaiseCompleted(RequestRecord record) => RequestCompleted?.Invoke(this, record);

    public void Dispose()
    {
    }
}
