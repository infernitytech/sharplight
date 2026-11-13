namespace Sharplight.Models;

/// <summary>
/// Interprets a <see cref="ModelSource"/> of a particular architecture and emits the format-agnostic
/// <see cref="ModelDefinition"/> (weights + compute graphs). This is the only architecture-specific code.
/// </summary>
public interface IModelAdapter
{
    public string Name { get; }
    /// <summary>Architecture ids handled (canonical ids, e.g. "qwen3_5").</summary>
    public IReadOnlyList<string> Architectures { get; }
    public bool CanLoad(ModelSource source) => Architectures.Contains(source.Architecture, StringComparer.OrdinalIgnoreCase);
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options);
}
