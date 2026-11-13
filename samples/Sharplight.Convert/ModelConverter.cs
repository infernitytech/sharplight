using System.Diagnostics;
using Sharplight.Models;
using Sharplight.Onnx;
using Sharplight.Package;
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
        var definition = Build(sources, settings);
        try
        {
            output.WriteLine($"Built {definition.Name} ({definition.Architecture}) from {sources.Primary.Format}: {definition.Graphs.Count} graph(s), "
                + $"{definition.Weights.Count} weights [{stopwatch.ElapsedMilliseconds} ms]");
            if (sources.Primary.CompilationCaches is { ArtifactSets.Count: > 0 } stored)
            {
                output.WriteLine($"Note: the {stored.ArtifactSets.Count} artifact set(s) of the source package are not copied; run precompile on the result.");
            }

            var toStore = settings.Optimize ? Optimize(definition) : definition;
            stopwatch.Restart();
            ModelPackage.Save(settings.DestinationPath, toStore, [], output.WriteLine);
            output.WriteLine($"Saved {settings.DestinationPath}: {ByteSize.Format(new FileInfo(settings.DestinationPath).Length)} [{stopwatch.ElapsedMilliseconds} ms]");
        }
        finally
        {
            foreach (var resource in definition.OwnedResources.Concat(sources.Primary.OwnedResources).Concat(sources.Auxiliary.Values.SelectMany(s => s.OwnedResources)).Distinct())
            {
                resource.Dispose();
            }
        }
    }

    private ModelDefinition Optimize(ModelDefinition definition)
    {
        var optimized = GraphOptimization.Run(definition, out bool applied);
        if (!applied)
        {
            output.WriteLine("Graph optimization is not available yet; the graphs are stored as built.");
        }

        return optimized;
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

    private static ModelDefinition Build(ModelSourceSet sources, ConvertSettings settings)
    {
        try
        {
            return ModelAdapterRegistry.Build(sources, new AdapterOptions { KvCacheType = settings.KvCacheType, ContextLength = settings.ContextLength });
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
