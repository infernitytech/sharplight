namespace Sharplight.Llm.Serving.Hosting;

/// <summary>Registration of a model under an API-visible name.</summary>
public sealed class ModelRegistration
{
    /// <summary>The name clients pass as <c>model</c> (unique per host).</summary>
    public required string Name { get; init; }
    /// <summary>How to load the model.</summary>
    public required IModelLoader Loader { get; init; }
    /// <summary>Load when the host starts (<see cref="ModelHost.LoadStartupModelsAsync"/>) instead of on the first request.</summary>
    public bool LoadAtStartup { get; init; }
    /// <summary>Never evicted for memory and never unloaded when idle.</summary>
    public bool Pinned { get; init; }
    /// <summary>Overrides <see cref="ModelHostOptions.IdleTimeout"/> for this model.</summary>
    public TimeSpan? IdleTimeout { get; init; }
    /// <summary>Free text shown in listings.</summary>
    public string? Description { get; init; }
    /// <summary>The loader's model kind.</summary>
    public ModelKind Kind => Loader.Kind;
}
