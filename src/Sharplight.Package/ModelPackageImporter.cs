using Sharplight.Models;

namespace Sharplight.Package;

/// <summary>Opens <c>.slm</c> packages through <see cref="ModelImporterRegistry"/>.</summary>
public sealed class ModelPackageImporter : IModelImporter
{
    /// <inheritdoc/>
    public string Format => "sharplight";

    /// <summary>Registers an importer for <c>.slm</c> packages with <see cref="ModelImporterRegistry"/>.</summary>
    public static void Register() => ModelImporterRegistry.Register(new ModelPackageImporter());

    /// <inheritdoc/>
    public bool CanOpen(string path) => ModelPackage.IsPackage(path);

    /// <inheritdoc/>
    public ModelSource Open(string path)
    {
        var file = new PackageFile(path);
        try
        {
            return new ModelSource
            {
                Format = Format,
                Architecture = file.Definition.Architecture,
                Metadata = file.Definition.SourceMetadata,
                Weights = file.Weights,
                Definition = file.Definition,
                OwnedResources = [file],
                Path = path,
                CompilationCaches = file,
            };
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }
}
