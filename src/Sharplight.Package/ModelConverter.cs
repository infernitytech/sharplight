using System.Diagnostics;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Package;

/// <summary>
/// Builds a model from its source through a <see cref="ModelLoader"/> (the formats and architectures the host registered) and writes
/// it as a package; the precompiled artifact sets of a package source are kept unless <see cref="ConversionOptions.KeepArtifacts"/> is
/// off. Registered as a singleton by <c>services.AddPackageImporter()</c> and <c>services.AddSharplight()</c>; thread-safe.
/// </summary>
/// <param name="loader">Opens and builds the sources.</param>
public sealed class ModelConverter(ModelLoader loader)
{
    /// <summary>Opens <paramref name="sourcePath"/> (plus auxiliary sources such as a vision projector) and converts it.</summary>
    /// <param name="sourcePath">A model file or directory any registered importer can open.</param>
    /// <param name="destinationPath">The package to write; replaced atomically.</param>
    /// <param name="options">Build options, or <see langword="null"/> for the defaults.</param>
    /// <param name="auxiliary">Auxiliary files by role (e.g. <see cref="Vision.AuxiliaryRoles.Mmproj"/>), or <see langword="null"/>.</param>
    /// <returns>A summary of the written package.</returns>
    /// <exception cref="FileNotFoundException">The source does not exist.</exception>
    /// <exception cref="NotSupportedException">No registered importer or adapter handles the source.</exception>
    public ConversionResult Convert(string sourcePath, string destinationPath, ConversionOptions? options = null,
        IEnumerable<KeyValuePair<string, string>>? auxiliary = null)
    {
        if (!File.Exists(sourcePath) && !Directory.Exists(sourcePath))
        {
            throw new FileNotFoundException($"Source '{sourcePath}' does not exist.", sourcePath);
        }

        var stopwatch = Stopwatch.StartNew();
        var sources = loader.Open(sourcePath, auxiliary);
        return Convert(sources, destinationPath, options ?? new ConversionOptions(), stopwatch);
    }

    /// <summary>Converts already opened sources. Takes ownership of <paramref name="sources"/> and disposes them.</summary>
    /// <param name="sources">The opened primary and auxiliary sources.</param>
    /// <param name="destinationPath">The package to write; replaced atomically.</param>
    /// <param name="options">Build options, or <see langword="null"/> for the defaults.</param>
    /// <returns>A summary of the written package.</returns>
    /// <exception cref="NotSupportedException">No registered adapter builds the source.</exception>
    public ConversionResult Convert(ModelSourceSet sources, string destinationPath, ConversionOptions? options = null)
    {
        return Convert(sources, destinationPath, options ?? new ConversionOptions(), Stopwatch.StartNew());
    }

    private ConversionResult Convert(ModelSourceSet sources, string destinationPath, ConversionOptions options, Stopwatch stopwatch)
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
    private ModelDefinition Build(ModelSourceSet sources, ConversionOptions options)
    {
        try
        {
            var adapterOptions = new AdapterOptions
            {
                KvCacheType = options.KvCacheType,
                ContextLength = options.ContextLength,
                OptimizeGraphs = options.OptimizeGraphs,
            };
            return loader.Build(sources, adapterOptions, options.Log);
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
