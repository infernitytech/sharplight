using System.Diagnostics;
using Sharplight.Graph.Optimization;
using Sharplight.Runtime;

namespace Sharplight.Package;

/// <summary>
/// Runs the device-independent graph optimization phase (<see cref="GraphOptimizer"/>) over the stored definition of an existing
/// package and writes the result, in place or as a new package: for packages converted with an older pipeline (or with the phase
/// disabled), so loading them no longer repeats the work. Weights, states, metadata, the model description and attachments are
/// carried over unchanged; precompiled artifact sets are kept unless <see cref="PackageOptimizeOptions.KeepArtifacts"/> is off.
/// A package that already carries the pipeline's id is left as it is unless <see cref="PackageOptimizeOptions.Force"/> is set.
/// </summary>
public static class PackageOptimizer
{
    /// <summary>Optimizes <paramref name="path"/> and writes the result.</summary>
    /// <param name="path">The existing package.</param>
    /// <param name="destinationPath">
    /// The package to write (replaced atomically), or <see langword="null"/> (or <paramref name="path"/> itself) to replace the
    /// input package in place; the input is only replaced after the new package was written completely.
    /// </param>
    /// <param name="options">Options, or <see langword="null"/> for the defaults.</param>
    /// <exception cref="FileNotFoundException">The package does not exist.</exception>
    /// <exception cref="InvalidDataException">The file is not a supported package.</exception>
    public static PackageOptimizeResult Optimize(string path, string? destinationPath = null, PackageOptimizeOptions? options = null)
    {
        options ??= new PackageOptimizeOptions();
        var optimizer = options.Optimizer ?? GraphOptimizer.Default;
        bool inPlace = destinationPath is null || Path.GetFullPath(destinationPath) == Path.GetFullPath(path);
        var file = new PackageFile(path);
        try
        {
            if (!options.Force && GraphOptimizer.IsOptimized(file.Definition, optimizer.Id))
            {
                return new PackageOptimizeResult(path, Optimized: false, optimizer.Id,
                    new Dictionary<string, int>(), new Dictionary<string, int>(), [],
                    file.ArtifactSets.Count, new FileInfo(path).Length, TimeSpan.Zero, TimeSpan.Zero);
            }

            var optimization = optimizer.Optimize(file.Definition);
            var artifacts = ReadArtifacts(path, file.ArtifactSets.Count, options);
            var stopwatch = Stopwatch.StartNew();
            var finalPath = inPlace ? path : destinationPath!;
            if (inPlace)
            {
                // Save reads the weights from the open mapping, so it cannot overwrite the input directly: write a sibling
                // temporary package, close the mapping and move it over the input.
                var temporary = path + ".optimized";
                try
                {
                    ModelPackage.Save(temporary, optimization.Definition, artifacts, options.Log);
                    file.Dispose();
                    File.Move(temporary, path, overwrite: true);
                }
                catch
                {
                    File.Delete(temporary);
                    throw;
                }
            }
            else
            {
                ModelPackage.Save(destinationPath!, optimization.Definition, artifacts, options.Log);
            }

            return new PackageOptimizeResult(finalPath, Optimized: true, optimizer.Id,
                optimization.NodesBefore, optimization.NodesAfter, optimization.Statistics,
                artifacts.Count, new FileInfo(finalPath).Length, optimization.Elapsed, stopwatch.Elapsed);
        }
        finally
        {
            file.Dispose();
        }
    }

    /// <summary>The package's artifact sets, read into memory so they survive closing the mapping, unless dropped.</summary>
    private static IReadOnlyList<CompilationCache> ReadArtifacts(string path, int stored, PackageOptimizeOptions options)
    {
        if (stored == 0)
        {
            return [];
        }

        if (!options.KeepArtifacts)
        {
            options.Log?.Invoke($"Dropping the {stored} artifact set(s) of the package.");
            return [];
        }

        var sets = ModelPackage.ReadArtifactSets(path);
        foreach (var set in sets)
        {
            options.Log?.Invoke($"Keeping {ArtifactDescriptor.Format(set.Descriptor)} ({set.Entries.Count} entries)");
        }

        return sets;
    }
}
