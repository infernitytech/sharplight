using System.Diagnostics;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Package.Import;

/// <summary>
/// Builds a model from its source through the registered importers (<see cref="ModelImporterRegistry"/>) and architecture
/// adapters (<see cref="ModelAdapterRegistry"/>) and writes it as a package; the precompiled artifact sets of a package source
/// are kept unless <see cref="ConversionOptions.KeepArtifacts"/> is off. The host registers the formats it supports before
/// converting; the built-in architecture adapters (<see cref="BuiltinAdapters"/>) are registered automatically.
/// </summary>
public static class ModelConverter
{
    /// <summary>The built-in architecture adapters are always available, as for <see cref="LlmModel"/> loads.</summary>
    static ModelConverter() => BuiltinAdapters.RegisterAll();

    /// <summary>Opens <paramref name="sourcePath"/> (plus auxiliary sources such as a vision projector) and converts it.</summary>
    /// <param name="sourcePath">A model file or directory any registered importer can open.</param>
    /// <param name="destinationPath">The package to write; replaced atomically.</param>
    /// <param name="options">Build options, or <see langword="null"/> for the defaults.</param>
    /// <param name="auxiliary">Auxiliary sources by role (e.g. <c>("mmproj", path)</c>), or <see langword="null"/>.</param>
    /// <returns>A summary of the written package.</returns>
    /// <exception cref="FileNotFoundException">The source does not exist.</exception>
    /// <exception cref="NotSupportedException">No registered importer or adapter handles the source.</exception>
    public static ConversionResult Convert(string sourcePath, string destinationPath, ConversionOptions? options = null,
        IReadOnlyList<(string Role, string Path)>? auxiliary = null)
    {
        if (!File.Exists(sourcePath) && !Directory.Exists(sourcePath))
        {
            throw new FileNotFoundException($"Source '{sourcePath}' does not exist.", sourcePath);
        }

        var stopwatch = Stopwatch.StartNew();
        var sources = ModelImporterRegistry.OpenSet(sourcePath, [.. auxiliary ?? []]);
        return Convert(sources, destinationPath, options ?? new ConversionOptions(), stopwatch);
    }

    /// <summary>Converts already opened sources. Takes ownership of <paramref name="sources"/> and disposes them.</summary>
    /// <param name="sources">The opened primary and auxiliary sources.</param>
    /// <param name="destinationPath">The package to write; replaced atomically.</param>
    /// <param name="options">Build options, or <see langword="null"/> for the defaults.</param>
    /// <returns>A summary of the written package.</returns>
    /// <exception cref="NotSupportedException">No registered adapter builds the source.</exception>
    public static ConversionResult Convert(ModelSourceSet sources, string destinationPath, ConversionOptions? options = null)
    {
        return Convert(sources, destinationPath, options ?? new ConversionOptions(), Stopwatch.StartNew());
    }

    private static ConversionResult Convert(ModelSourceSet sources, string destinationPath, ConversionOptions options, Stopwatch stopwatch)
    {
        var definition = Build(sources, options);
        try
        {
            var buildTime = stopwatch.Elapsed;
            var artifacts = SourceArtifacts(sources.Primary, options);
            int dropped = (sources.Primary.CompilationCaches?.ArtifactSets.Count ?? 0) - artifacts.Count;
            stopwatch.Restart();
            ModelPackage.Save(destinationPath, definition, artifacts, options.Log);
            return new ConversionResult(
                destinationPath,
                definition.Name,
                definition.Architecture,
                sources.Primary.Format,
                definition.Graphs.Count,
                definition.Weights.Count,
                new FileInfo(destinationPath).Length,
                buildTime,
                stopwatch.Elapsed,
                artifacts.Count,
                Math.Max(dropped, 0),
                definition.SourceMetadata.TryGetValue(CanonicalKeys.Architecture, out var architecture) ? architecture as string : null);
        }
        finally
        {
            Dispose(definition.OwnedResources.Concat(SourceResources(sources)));
        }
    }

    /// <summary>The artifact sets of a package source to store with the result, unless dropped.</summary>
    private static IReadOnlyList<CompilationCache> SourceArtifacts(ModelSource source, ConversionOptions options)
    {
        if (source.CompilationCaches is not { ArtifactSets.Count: > 0 } stored || source.Path is not { } path)
        {
            return [];
        }

        if (!options.KeepArtifacts)
        {
            options.Log?.Invoke($"Dropping the {stored.ArtifactSets.Count} artifact set(s) of the source package.");
            return [];
        }

        var sets = ModelPackage.ReadArtifactSets(path);
        foreach (var set in sets)
        {
            options.Log?.Invoke($"Keeping {ArtifactDescriptor.Format(set.Descriptor)} ({set.Entries.Count} entries)");
        }

        return sets;
    }

    /// <summary>Builds the definition; the graph optimization phase runs inside the build unless disabled.</summary>
    private static ModelDefinition Build(ModelSourceSet sources, ConversionOptions options)
    {
        try
        {
            var adapterOptions = new AdapterOptions
            {
                KvCacheType = options.KvCacheType,
                ContextLength = options.ContextLength,
                OptimizeGraphs = options.OptimizeGraphs,
            };
            return ModelAdapterRegistry.Build(sources, adapterOptions, options.Log);
        }
        catch
        {
            Dispose(SourceResources(sources));
            throw;
        }
    }

    private static IEnumerable<IDisposable> SourceResources(ModelSourceSet sources)
    {
        return sources.Primary.OwnedResources.Concat(sources.Auxiliary.Values.SelectMany(source => source.OwnedResources));
    }

    private static void Dispose(IEnumerable<IDisposable> resources)
    {
        foreach (var resource in resources.Distinct())
        {
            resource.Dispose();
        }
    }
}
