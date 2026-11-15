using Sharplight.Models;

namespace Sharplight.Package;

/// <summary>Opens <c>.slm</c> packages for the <see cref="ModelLoader"/> (registered with <c>services.AddPackageImporter()</c>).</summary>
public sealed class ModelPackageImporter : IModelImporter
{
    /// <inheritdoc/>
    public string Format => "sharplight";

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
