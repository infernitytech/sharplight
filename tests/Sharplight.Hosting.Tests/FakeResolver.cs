namespace Sharplight.Hosting.Tests;

/// <summary>Knows one listed model and resolves names under <c>store/</c>.</summary>
public sealed class FakeResolver : IHostedModelResolver
{
    public IReadOnlyList<HostedModelOptions> List() => [new HostedModelOptions { Name = "listed", Path = "fake://1G", Lazy = true }];

    public HostedModelOptions? Resolve(string name) => name.StartsWith("store/", StringComparison.Ordinal)
        ? new HostedModelOptions { Name = name, Path = "fake://1G" }
        : null;
}
