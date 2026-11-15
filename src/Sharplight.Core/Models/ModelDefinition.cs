using Sharplight.Graph;

namespace Sharplight.Models;

/// <summary>
/// The format-agnostic result of importing a model: weights, persistent state definitions and one
/// or more named compute graphs (e.g. <c>"decode"</c>, <c>"vision_encode"</c>, <c>"mtp_draft"</c>).
/// </summary>
public sealed class ModelDefinition
{
    /// <summary>Canonical architecture id the model was built for.</summary>
    public required string Architecture { get; init; }
    /// <summary>Display name; empty when unknown.</summary>
    public string Name { get; init; } = "";
    /// <summary>
    /// All weights by name, as referenced by the graphs. The bytes stay in their source until a backend reads them.
    /// </summary>
    public required IReadOnlyDictionary<string, WeightInfo> Weights { get; init; }
    /// <summary>The compute graphs by name (see <see cref="WellKnown"/> for the standard names).</summary>
    public required IReadOnlyDictionary<string, ComputeGraph> Graphs { get; init; }
    /// <summary>Persistent states (KV caches, recurrent states) shared by the graphs.</summary>
    public IReadOnlyList<StateDefinition> States { get; init; } = [];

    /// <summary>Architecture-level facts runtimes may need (vocab size, context length, eos ids, ...).</summary>
    public IReadOnlyDictionary<string, object> Properties { get; init; } = new Dictionary<string, object>();

    /// <summary>Canonical source metadata (<see cref="CanonicalKeys"/>) for consumers such as tokenizers.</summary>
    public IReadOnlyDictionary<string, object> SourceMetadata { get; init; } = new Dictionary<string, object>();

    /// <summary>
    /// What the model is and which components it consists of, or <see langword="null"/> for models that do not describe themselves
    /// (they resolve to a single-component text-generation model, see <see cref="ModelDescription.Resolve"/>).
    /// </summary>
    public ModelDescription? Description { get; init; }

    /// <summary>
    /// Named binary attachments the components refer to (<see cref="ModelComponent.Attachments"/>): tokenizer files, scheduler
    /// configurations, filter banks, voice embeddings, ... Stored as aligned, memory-mapped sections in an <c>.slm</c> package.
    /// </summary>
    public IReadOnlyDictionary<string, IWeightSource> Attachments { get; init; } = new Dictionary<string, IWeightSource>();

    /// <summary>Resources that must be disposed with the definition (memory maps, file handles).</summary>
    public IReadOnlyList<IDisposable> OwnedResources { get; init; } = [];

    /// <summary>
    /// Reads a property, converting it to <typeparamref name="T"/>.
    /// </summary>
    /// <exception cref="KeyNotFoundException">The property is absent.</exception>
    public T Property<T>(string key) => (T)Convert.ChangeType(Properties[key], typeof(T), System.Globalization.CultureInfo.InvariantCulture);
    /// <summary>Reads an optional property, converting it to <typeparamref name="T"/>.</summary>
    public T Property<T>(string key, T defaultValue) => Properties.TryGetValue(key, out var v)
        ? (T)Convert.ChangeType(v, typeof(T), System.Globalization.CultureInfo.InvariantCulture) : defaultValue;

    /// <summary>
    /// Gets a graph by name.
    /// </summary>
    /// <exception cref="KeyNotFoundException">The model has no such graph.</exception>
    public ComputeGraph Graph(string name) => Graphs.TryGetValue(name, out var g)
        ? g : throw new KeyNotFoundException($"Model '{Architecture}' has no graph '{name}'. Available: {string.Join(", ", Graphs.Keys)}");
}
