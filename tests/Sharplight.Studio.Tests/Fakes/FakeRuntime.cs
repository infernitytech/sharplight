using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A runtime whose loads create <see cref="FakeModel"/>s.</summary>
internal sealed class FakeRuntime : IModelRuntime
{
    public static readonly ComputeDeviceOption Gpu = new("vulkan:0", "Test GPU", "Vulkan", true, 16L << 30, ["fp16", "subgroups"]);

    public static readonly ComputeDeviceOption Cpu = new("cpu", "CPU", "Cpu", false, 32L << 30, []);

    private readonly List<ILoadedModel> _models = [];

    public IReadOnlyList<ComputeDeviceOption> Devices { get; } = [Gpu, Cpu];

    public IReadOnlyList<ILoadedModel> LoadedModels => _models;

    public List<ModelLoadRequest> Requests { get; } = [];

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
        ILoadedModel model = Add(new FakeModel(request.Id) { Settings = request.Settings });
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

    public void RaiseCompleted(RequestRecord record) => RequestCompleted?.Invoke(this, record);

    public void Dispose()
    {
    }
}
