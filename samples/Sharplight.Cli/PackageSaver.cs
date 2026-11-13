using System.Diagnostics;
using Sharplight.Llm;
using Sharplight.Package;

namespace Sharplight.Cli;

/// <summary>Stores the built model plus this device's compiled kernels and pipeline cache in one <c>.slm</c> package.</summary>
internal static class PackageSaver
{
    private const double BytesPerGigabyte = 1e9;

    /// <summary>
    /// Writes <paramref name="savePath"/>. When the model already is that package, the compiled targets are appended to it
    /// instead of rewriting the weights.
    /// </summary>
    public static void Save(LlmModel model, string savePath, TextWriter log)
    {
        var stopwatch = Stopwatch.StartNew();
        var targets = model.ExportCompilation();
        if (IsSamePackage(model, savePath))
        {
            ModelPackage.AddTargets(savePath, targets);
        }
        else
        {
            ModelPackage.Save(savePath, model.Definition, targets, log.WriteLine);
        }

        double sizeGigabytes = new FileInfo(savePath).Length / BytesPerGigabyte;
        var targetSummary = string.Join("; ", targets.Select(target => $"{target.Target} ({target.Entries.Count} entries)"));
        log.WriteLine($"Saved {savePath}: {sizeGigabytes:F2} GB, targets: {targetSummary} [{stopwatch.ElapsedMilliseconds} ms]");
    }

    private static bool IsSamePackage(LlmModel model, string savePath)
    {
        return model.Source.Path is { } sourcePath
            && ModelPackage.IsPackage(sourcePath)
            && Path.GetFullPath(sourcePath) == Path.GetFullPath(savePath);
    }
}
