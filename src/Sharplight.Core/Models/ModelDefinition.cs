using Sharplight.Graph;

namespace Sharplight.Models;

/// <summary>
/// The format-agnostic result of importing a model: weights, persistent state definitions and one
/// or more named compute graphs (e.g. <c>"decode"</c>, <c>"vision_encode"</c>, <c>"mtp_draft"</c>).
/// </summary>
public sealed class ModelDefinition
{
    public required string Architecture { get; init; }
    public string Name { get; init; } = "";
    public required IReadOnlyDictionary<string, WeightInfo> Weights { get; init; }
    public required IReadOnlyDictionary<string, ComputeGraph> Graphs { get; init; }
    public IReadOnlyList<StateDefinition> States { get; init; } = [];

    /// <summary>Architecture-level facts runtimes may need (vocab size, context length, eos ids, ...).</summary>
    public IReadOnlyDictionary<string, object> Properties { get; init; } = new Dictionary<string, object>();

    /// <summary>Canonical source metadata (<see cref="CanonicalKeys"/>) for consumers such as tokenizers.</summary>
    public IReadOnlyDictionary<string, object> SourceMetadata { get; init; } = new Dictionary<string, object>();

    /// <summary>Resources that must be disposed with the definition (memory maps, file handles).</summary>
    public IReadOnlyList<IDisposable> OwnedResources { get; init; } = [];

    public T Property<T>(string key) => (T)Convert.ChangeType(Properties[key], typeof(T), System.Globalization.CultureInfo.InvariantCulture);
    public T Property<T>(string key, T defaultValue) => Properties.TryGetValue(key, out var v)
        ? (T)Convert.ChangeType(v, typeof(T), System.Globalization.CultureInfo.InvariantCulture) : defaultValue;

    public ComputeGraph Graph(string name) => Graphs.TryGetValue(name, out var g)
        ? g : throw new KeyNotFoundException($"Model '{Architecture}' has no graph '{name}'. Available: {string.Join(", ", Graphs.Keys)}");
}
