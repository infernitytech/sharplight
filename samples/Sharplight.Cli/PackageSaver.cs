using System.Diagnostics;
using Sharplight.Llm;
using Sharplight.Package;
using Sharplight.Runtime;

namespace Sharplight.Cli;

/// <summary>Stores the built model plus this device's compiled artifacts (kernel code and driver pipeline cache) in one <c>.slm</c> package.</summary>
internal static class PackageSaver
{
    private const double BytesPerGigabyte = 1e9;

    /// <summary>
    /// Writes <paramref name="savePath"/>. When the model already is that package, the artifact sets are appended to it
    /// instead of rewriting the weights.
    /// </summary>
    public static void Save(LlmModel model, string savePath, TextWriter log)
    {
        var stopwatch = Stopwatch.StartNew();
        var artifacts = model.ExportCompilation();
        if (IsSamePackage(model, savePath))
        {
            ModelPackage.AddArtifacts(savePath, artifacts);
        }
        else
        {
            ModelPackage.Save(savePath, model.Definition, artifacts, log.WriteLine);
        }

        double sizeGigabytes = new FileInfo(savePath).Length / BytesPerGigabyte;
        var setSummary = string.Join("; ", artifacts.Select(set => $"{ArtifactDescriptor.Format(set.Descriptor)} ({set.Entries.Count} entries)"));
        log.WriteLine($"Saved {savePath}: {sizeGigabytes:F2} GB, artifact sets: {setSummary} [{stopwatch.ElapsedMilliseconds} ms]");
    }

    private static bool IsSamePackage(LlmModel model, string savePath)
    {
        return model.Source.Path is { } sourcePath
            && ModelPackage.IsPackage(sourcePath)
            && Path.GetFullPath(sourcePath) == Path.GetFullPath(savePath);
    }
}
