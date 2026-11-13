namespace Sharplight.Models;

/// <summary>
/// Interprets a <see cref="ModelSource"/> of a particular architecture and emits the format-agnostic
/// <see cref="ModelDefinition"/> (weights + compute graphs). This is the only architecture-specific code.
/// </summary>
public interface IModelAdapter
{
    /// <summary>Display name of the adapter.</summary>
    public string Name { get; }
    /// <summary>Architecture ids handled (canonical ids, e.g. "qwen3_5").</summary>
    public IReadOnlyList<string> Architectures { get; }
    /// <summary>
    /// Whether this adapter can build a model from the source; by default true when the source architecture is one of <see cref="Architectures"/>.
    /// </summary>
    public bool CanLoad(ModelSource source) => Architectures.Contains(source.Architecture, StringComparer.OrdinalIgnoreCase);
    /// <summary>
    /// Builds the model: weights (possibly re-labelled or derived views of the source weights), states and compute graphs.
    /// </summary>
    /// <param name="sources">The primary source and any auxiliary sources.</param>
    /// <param name="options">Adapter options.</param>
    /// <exception cref="NotSupportedException">The sources lack data the architecture needs.</exception>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options);
}
