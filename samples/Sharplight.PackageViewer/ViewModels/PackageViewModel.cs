using Sharplight.Package;

namespace Sharplight.PackageViewer.ViewModels;

/// <summary>An opened package: one view model per tab. Built off the UI thread by <see cref="Load"/>.</summary>
public sealed class PackageViewModel
{
    private PackageViewModel(PackageInfo info, DefinitionInfo? definition, string? definitionError)
    {
        Path = info.Path;
        FileName = System.IO.Path.GetFileName(info.Path);
        Overview = new OverviewViewModel(info, definition, definitionError);
        Graphs = new GraphsViewModel(definition?.Graphs ?? []);
        Weights = new WeightsViewModel(info.Weights);
        Artifacts = new ArtifactsViewModel(info.ArtifactSets);
    }

    public string Path { get; }

    public string FileName { get; }

    public OverviewViewModel Overview { get; }

    public GraphsViewModel Graphs { get; }

    public WeightsViewModel Weights { get; }

    public ArtifactsViewModel Artifacts { get; }

    /// <summary>
    /// Reads the package structure and decodes its definition (no weight or artifact bytes are read). A definition that does not
    /// decode with this version of Sharplight is reported in the overview; the rest of the package is still shown.
    /// </summary>
    /// <exception cref="InvalidDataException">The file is not a supported package or is damaged.</exception>
    /// <exception cref="IOException">The file cannot be read.</exception>
    public static PackageViewModel Load(string path)
    {
        var info = ModelPackageReader.Open(path);
        DefinitionInfo? definition = null;
        string? error = null;
        try
        {
            definition = info.Definition;
        }
        catch (InvalidDataException e)
        {
            error = e.Message;
        }

        return new PackageViewModel(info, definition, error);
    }
}
