namespace Sharplight.Models;

/// <summary>
/// Builds one model out of several sources (e.g. a language model plus the codec decoder of a speech synthesizer) and decides on the whole
/// <see cref="ModelSourceSet"/>. Composite adapters are registered with <c>services.AddCompositeModelAdapter&lt;T&gt;()</c>; the
/// <see cref="ArchitectureCatalog"/> consults them before the adapters registered per architecture, and only for source sets with
/// auxiliary sources.
/// </summary>
public interface ICompositeModelAdapter
{
    /// <summary>Display name of the adapter.</summary>
    public string Name { get; }

    /// <summary>Whether this adapter builds a model from the source set (the primary source and its auxiliary sources).</summary>
    /// <param name="sources">The primary source and its auxiliary sources.</param>
    /// <returns><see langword="true"/> when the adapter handles the set.</returns>
    public bool CanLoad(ModelSourceSet sources);

    /// <summary>Builds the model; the parts of one source can be built with the adapters of <paramref name="architectures"/>.</summary>
    /// <param name="sources">The primary source and its auxiliary sources.</param>
    /// <param name="options">Adapter options.</param>
    /// <param name="architectures">The catalog the adapter is registered with, to build the component models.</param>
    /// <returns>The built definition; it takes over the weights of <paramref name="sources"/>.</returns>
    /// <exception cref="NotSupportedException">The sources lack data the architecture needs.</exception>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options, ArchitectureCatalog architectures);
}
