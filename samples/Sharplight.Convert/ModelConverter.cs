using System.Diagnostics;
using Sharplight.Models;
using Sharplight.Onnx;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary>Builds a model from its source through the registered importers and adapters and writes it as a package.</summary>
internal sealed class ModelConverter(TextWriter output)
{
    private const string OnnxExtension = ".onnx";
    private const string MmprojRole = "mmproj";

    public void Convert(ConvertSettings settings)
    {
        if (!File.Exists(settings.SourcePath) && !Directory.Exists(settings.SourcePath))
        {
            throw new SampleException($"Source '{settings.SourcePath}' does not exist.", ExitCodes.InvalidInput);
        }

        var stopwatch = Stopwatch.StartNew();
        var sources = Open(settings);
        var definition = Build(sources, settings, output);
        try
        {
            output.WriteLine($"Built {definition.Name} ({definition.Architecture}) from {sources.Primary.Format}: {definition.Graphs.Count} graph(s), "
                + $"{definition.Weights.Count} weights [{stopwatch.ElapsedMilliseconds} ms]");
            var artifacts = SourceArtifacts(sources.Primary, settings);
            stopwatch.Restart();
            ModelPackage.Save(settings.DestinationPath, definition, artifacts, output.WriteLine);
            output.WriteLine($"Saved {settings.DestinationPath}: {ByteSize.Format(new FileInfo(settings.DestinationPath).Length)}, {artifacts.Count} artifact set(s) "
                + $"[{stopwatch.ElapsedMilliseconds} ms]");
        }
        finally
        {
            foreach (var resource in definition.OwnedResources.Concat(sources.Primary.OwnedResources).Concat(sources.Auxiliary.Values.SelectMany(s => s.OwnedResources)).Distinct())
            {
                resource.Dispose();
            }
        }
    }

    /// <summary>
    /// The artifact sets of a package source, to store with the result unless dropped. Code entries are content addressed (a
    /// kernel the converted graphs no longer use is never asked for) and the driver validates pipeline caches, so they stay valid.
    /// </summary>
    private IReadOnlyList<CompilationCache> SourceArtifacts(ModelSource source, ConvertSettings settings)
    {
        if (source.CompilationCaches is not { ArtifactSets.Count: > 0 } stored || source.Path is not { } path)
        {
            return [];
        }

        if (!settings.KeepArtifacts)
        {
            output.WriteLine($"Dropping the {stored.ArtifactSets.Count} artifact set(s) of the source package.");
            return [];
        }

        var sets = ModelPackage.ReadArtifactSets(path);
        foreach (var set in sets)
        {
            output.WriteLine($"Keeping {ArtifactDescriptor.Format(set.Descriptor)} ({set.Entries.Count} entries)");
        }

        return sets;
    }

    private static ModelSourceSet Open(ConvertSettings settings)
    {
        try
        {
            if (string.Equals(Path.GetExtension(settings.SourcePath), OnnxExtension, StringComparison.OrdinalIgnoreCase))
            {
                var options = new OnnxImportOptions();
                foreach (var (name, shape) in settings.InputShapes)
                {
                    options.InputShapes[name] = shape;
                }

                return new ModelSourceSet { Primary = OnnxImporter.Open(settings.SourcePath, options) };
            }

            (string, string)[] auxiliary = settings.MmprojPath is { } mmproj ? [(MmprojRole, mmproj)] : [];
            return ModelImporterRegistry.OpenSet(settings.SourcePath, auxiliary);
        }
        catch (NotSupportedException error)
        {
            throw new SampleException(error.Message, error);
        }
    }

    /// <summary>Builds the definition; the graph optimization phase runs inside the build unless disabled.</summary>
    private static ModelDefinition Build(ModelSourceSet sources, ConvertSettings settings, TextWriter output)
    {
        try
        {
            var options = new AdapterOptions { KvCacheType = settings.KvCacheType, ContextLength = settings.ContextLength, OptimizeGraphs = settings.Optimize };
            return ModelAdapterRegistry.Build(sources, options, output.WriteLine);
        }
        catch (NotSupportedException error)
        {
            foreach (var resource in sources.Primary.OwnedResources)
            {
                resource.Dispose();
            }

            throw new SampleException(error.Message, error);
        }
    }
}
