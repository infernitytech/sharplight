using Microsoft.Extensions.DependencyInjection;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// Loads a task-layer model (speech recognition, speech synthesis, ...) onto devices shared with the other models of the host. The
/// footprint is estimated from the files before anything is read (the weights plus room for activations and state), so the host can
/// evict other models first; <see cref="IPreparedModel.Load"/> reads the model and creates the client of its kind.
/// </summary>
internal sealed class TaskModelLoader : IHostedModelLoader
{
    private const double OverheadFactor = 1.25;
    private const long OverheadBytes = 256L << 20;
    private readonly ModelLoadRequest _request;
    private readonly TaskLoadOptions _options;
    private readonly IReadOnlyList<DeviceMemoryPool> _devices;
    private readonly IServiceProvider _services;
    private readonly CancellationToken _cancellation;

    /// <param name="request">The model to load.</param>
    /// <param name="options">The task load options (devices, compile settings, progress).</param>
    /// <param name="devices">The devices of the options, as memory pools of the host.</param>
    /// <param name="services">The engine services that provide the task clients.</param>
    /// <param name="cancellation">Cancelled to abandon a load.</param>
    public TaskModelLoader(ModelLoadRequest request, TaskLoadOptions options, IReadOnlyList<DeviceMemoryPool> devices, IServiceProvider services, CancellationToken cancellation)
    {
        _request = request;
        _options = options;
        _devices = devices;
        _services = services;
        _cancellation = cancellation;
    }

    /// <inheritdoc/>
    public ModelKind Kind => ModelKind.Task;

    /// <inheritdoc/>
    public IPreparedModel Prepare(CancellationToken ct)
    {
        long bytes = (long)(FileBytes(_request.Path) * OverheadFactor) + OverheadBytes;
        foreach (var (_, path) in _request.Auxiliary ?? [])
        {
            bytes += FileBytes(path);
        }

        long share = bytes / Math.Max(1, _devices.Count);
        return new Prepared(this, [.. _devices.Select(pool => new MemoryRequirement(pool, share))], bytes);
    }

    private static long FileBytes(string path)
    {
        if (File.Exists(path))
        {
            return new FileInfo(path).Length;
        }

        return Directory.Exists(path)
            ? new DirectoryInfo(path).EnumerateFiles("*", SearchOption.AllDirectories).Sum(file => file.Length)
            : 0;
    }

    private sealed class Prepared(TaskModelLoader owner, IReadOnlyList<MemoryRequirement> requirements, long bytes) : IPreparedModel
    {
        /// <inheritdoc/>
        public IReadOnlyList<MemoryRequirement> Requirements => requirements;

        /// <inheritdoc/>
        public bool CanOffload => false;

        /// <inheritdoc/>
        public LoadedModel Load(long? primaryLimit)
        {
            var model = TaskModel.Load(owner._services.GetRequiredService<ModelLoader>(), owner._options, owner._cancellation);
            try
            {
                var provider = owner._services.GetService(typeof(TaskClientProvider)) as TaskClientProvider
                    ?? throw new InvalidOperationException("The engine services have no task client provider.");
                var client = TaskKindClients.Create(provider, model, owner._request.Kind);
                return new TaskModelInstance(model, [client], bytes);
            }
            catch
            {
                model.Dispose();
                throw;
            }
        }

        /// <inheritdoc/>
        public void Dispose()
        {
        }
    }
}
