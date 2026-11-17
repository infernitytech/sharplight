using Sharplight.Hosting;
using Sharplight.Hub;

namespace Sharplight.Cli;

/// <summary>
/// On-demand models of <c>serve --on-demand</c>: every model of the local model store, by its stored name (<c>owner/name[:tag]</c> or
/// <c>alias[:tag]</c>), ollama-style.
/// </summary>
/// <param name="storeRoot">The model store.</param>
internal sealed class StoreModelResolver(string storeRoot) : IHostedModelResolver
{
    public IReadOnlyList<HostedModelOptions> List()
    {
        return [.. new ModelStore(storeRoot).List().Select(manifest => Model(manifest.Name, manifest.PackagePath))];
    }

    public HostedModelOptions? Resolve(string name)
    {
        try
        {
            return new ModelStore(storeRoot).Find(name) is { } manifest && File.Exists(manifest.PackagePath) ? Model(name, manifest.PackagePath) : null;
        }
        catch (HubException)
        {
            return null;   // ambiguous name
        }
    }

    private static HostedModelOptions Model(string name, string path) => new() { Name = name, Path = path, Lazy = true };
}
