using Sharplight.Models;
using Sharplight.Onnx;
using Sharplight.Packaging;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary>Runs <c>convert</c>: opens the source (ONNX with the given input shapes) and writes the package through <see cref="ModelConverter"/>.</summary>
internal sealed class ConvertRunner(TextWriter output)
{
    private const string OnnxExtension = ".onnx";

    public void Run(ConvertSettings settings)
    {
        if (!File.Exists(settings.SourcePath) && !Directory.Exists(settings.SourcePath))
        {
            throw new SampleException($"Source '{settings.SourcePath}' does not exist.", ExitCodes.InvalidInput);
        }

        var options = new ConversionOptions
        {
            OptimizeGraphs = settings.Optimize,
            KvCacheType = settings.KvCacheType,
            ContextLength = settings.ContextLength,
            KeepArtifacts = settings.KeepArtifacts,
            Log = output.WriteLine,
        };
        try
        {
            var result = ModelConverter.Convert(Open(settings), settings.DestinationPath, options);
            output.WriteLine($"Built {result.ModelName} ({result.Architecture}) from {result.SourceFormat}: {result.GraphCount} graph(s), "
                + $"{result.WeightCount} weights [{result.BuildTime.TotalMilliseconds:F0} ms]");
            output.WriteLine($"Saved {result.PackagePath}: {ByteSize.Format(result.PackageBytes)}, {result.KeptArtifactSets} artifact set(s) "
                + $"[{result.SaveTime.TotalMilliseconds:F0} ms]");
        }
        catch (NotSupportedException error)
        {
            throw new SampleException(error.Message, error);
        }
    }

    private static ModelSourceSet Open(ConvertSettings settings)
    {
        if (string.Equals(Path.GetExtension(settings.SourcePath), OnnxExtension, StringComparison.OrdinalIgnoreCase))
        {
            var onnxOptions = new OnnxImportOptions();
            foreach (var (name, shape) in settings.InputShapes)
            {
                onnxOptions.InputShapes[name] = shape;
            }

            return new ModelSourceSet { Primary = OnnxImporter.Open(settings.SourcePath, onnxOptions) };
        }

        (string, string)[] auxiliary = settings.MmprojPath is { } mmproj ? [(Vision.AuxiliaryRoles.Mmproj, mmproj)] : [];
        return ModelImporterRegistry.OpenSet(settings.SourcePath, auxiliary);
    }
}
