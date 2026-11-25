using Microsoft.Extensions.DependencyInjection;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Hosting.Loaders;

/// <summary>
/// Loads a task-layer model (speech recognition, speech synthesis, image generation, text embedding, ...) onto devices shared with the other models of
/// the host. The footprint is estimated from the files before anything is read (the weights plus room for activations and state), so the
/// host can evict other models first; <see cref="IPreparedModel.Load"/> reads the model and creates the client of its kind
/// (<see cref="TaskKindClients"/>).
/// </summary>
/// <param name="taskKind">The task kind (<see cref="ModelKinds"/>), which selects the client.</param>
/// <param name="options">The load options (<see cref="TaskLoadOptions.ModelPath"/>, devices, compile settings, progress).</param>
/// <param name="devices">The devices of the options, as memory pools of the host.</param>
/// <param name="services">The services that provide the <see cref="ModelLoader"/> and the <see cref="TaskClientProvider"/>.</param>
/// <param name="cancellation">Cancelled to abandon a load.</param>
public sealed class TaskModelLoader(string taskKind, TaskLoadOptions options, IReadOnlyList<IMemoryDomain> devices, IServiceProvider services,
    CancellationToken cancellation = default) : IHostedModelLoader
{
    /// <inheritdoc/>
    public ModelKind Kind => TaskKindClients.HostedKind(taskKind);

    /// <inheritdoc/>
    public IPreparedModel Prepare(CancellationToken ct)
    {
        long bytes = FileFootprint.Estimate(options.ModelPath ?? "", options.Auxiliary.Values);
        return new FileSizedPreparedModel(FileSizedPreparedModel.Spread(bytes, devices), () => Load(bytes));
    }

    private TaskModelInstance Load(long bytes)
    {
        var model = TaskModel.Load(services.GetRequiredService<ModelLoader>(), options, cancellation);
        try
        {
            var provider = services.GetService<TaskClientProvider>()
                ?? throw new InvalidOperationException("The services have no task client provider (services.AddSharplight()).");
            var client = TaskKindClients.Create(provider, model, taskKind);
            return new TaskModelInstance(model, [client], bytes, Kind);
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }
}
