using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Hosting.Loaders;

/// <summary>
/// Creates the <see cref="IHostedModelLoader"/> of a hosted model. The manager asks the registered factories in registration order
/// (later registrations first, so applications can override the built-in chat, embedding, reranker and task factories) and uses the
/// first that accepts the model.
/// </summary>
public interface IHostedModelLoaderFactory
{
    /// <summary>True when this factory loads <paramref name="model"/> (normally decided by <see cref="HostedModelSettings.Kind"/>).</summary>
    /// <param name="model">The model's settings.</param>
    public bool CanLoad(HostedModelSettings model);

    /// <summary>Creates the loader. It is called once per registration; the loader may be asked to prepare the model many times.</summary>
    /// <param name="context">The model's settings, the placement and the services.</param>
    /// <returns>The loader.</returns>
    public IHostedModelLoader CreateLoader(HostedModelLoadContext context);
}
