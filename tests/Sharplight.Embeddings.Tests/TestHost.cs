using Microsoft.Extensions.DependencyInjection;
using Sharplight.Llm.Embeddings;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Embeddings.Tests;

/// <summary>
/// The services the tests load models through: the formats, architectures and execution providers of Sharplight
/// (<c>services.AddSharplight()</c>), built once per test assembly.
/// </summary>
internal static class TestHost
{
    private static readonly Lazy<ServiceProvider> s_services = new(() => SharplightDefaults.CreateServiceProvider());

    /// <summary>The container.</summary>
    public static IServiceProvider Services => s_services.Value;

    /// <summary>Opens and builds models.</summary>
    public static ModelLoader Loader => Services.GetRequiredService<ModelLoader>();

    /// <summary>The execution providers (Vulkan, then CPU).</summary>
    public static ExecutionProviderCatalog Providers => Services.GetRequiredService<ExecutionProviderCatalog>();

    /// <summary>Loads a model with the text-embedding load defaults (and <paramref name="maxBatchTokens"/> when positive) and creates its <typeparamref name="TClient"/>; the caller disposes both, the client first.</summary>
    public static (TaskModel Model, TClient Client) Load<TClient>(string path, DeviceSlot device, int maxBatchTokens = 0)
        where TClient : class, ITaskClient
    {
        var options = new TaskLoadOptions { ModelPath = path, Devices = [device] };
        TextEmbeddingLoadDefaults.Apply(options);
        if (maxBatchTokens > 0)
        {
            options.MaxBatchTokens = maxBatchTokens;
        }

        var model = TaskModel.Load(Loader, options);
        try
        {
            return (model, Services.GetRequiredService<TaskClientProvider>().Create<TClient>(model));
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    /// <summary>Converts models to packages.</summary>
    public static ModelConverter Converter => Services.GetRequiredService<ModelConverter>();
}
