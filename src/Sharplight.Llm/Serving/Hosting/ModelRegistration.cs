namespace Sharplight.Llm.Serving.Hosting;

/// <summary>Registration of a model under an API-visible name.</summary>
public sealed class ModelRegistration
{
    public required string Name { get; init; }
    public required IModelLoader Loader { get; init; }
    /// <summary>Load when the host starts (<see cref="ModelHost.LoadStartupModelsAsync"/>) instead of on the first request.</summary>
    public bool LoadAtStartup { get; init; }
    /// <summary>Never evicted for memory and never unloaded when idle.</summary>
    public bool Pinned { get; init; }
    /// <summary>Overrides <see cref="ModelHostOptions.IdleTimeout"/> for this model.</summary>
    public TimeSpan? IdleTimeout { get; init; }
    public string? Description { get; init; }
    public ModelKind Kind => Loader.Kind;
}
