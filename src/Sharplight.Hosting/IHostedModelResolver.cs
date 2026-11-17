namespace Sharplight.Hosting;

/// <summary>
/// Supplies models on demand (with <see cref="ModelHostingOptions.OnDemand"/>): names that are not registered are looked up here when a
/// request asks for them, e.g. every model of a local model store, ollama-style. Register one implementation in the container.
/// </summary>
public interface IHostedModelResolver
{
    /// <summary>The models known up front, listed (and loadable) right away.</summary>
    /// <returns>The models; their names must be unique.</returns>
    public IReadOnlyList<HostedModelOptions> List();

    /// <summary>The model registered under <paramref name="name"/> when a request asks for it, or null when the name is unknown.</summary>
    /// <param name="name">The requested name.</param>
    /// <returns>The model (its <see cref="HostedModelOptions.Name"/> must equal <paramref name="name"/>), or null.</returns>
    public HostedModelOptions? Resolve(string name);
}
