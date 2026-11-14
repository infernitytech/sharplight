using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Hub;
using Sharplight.Package;
using Sharplight.Studio.Services.Inspection;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>The tab view models of an opened package; built off the UI thread by <see cref="Read"/>.</summary>
internal sealed partial class PackageInspection : ObservableObject
{
    private PackageInspection(PackageInfo info, DefinitionInfo? definition, string? definitionError, ModelManifest? manifest)
    {
        Info = info;
        var weights = info.Weights.GroupBy(w => w.Name, StringComparer.Ordinal).ToDictionary(g => g.Key, g => g.First(), StringComparer.Ordinal);
        Sections = new SectionsViewModel(info);
        Overview = new OverviewViewModel(info, definition, definitionError, manifest, Sections.KindTotals);
        Graphs = new GraphsViewModel(definition?.Graphs ?? [], weights);
        Weights = new WeightsViewModel(info.Weights, info.Sections.Where(s => s.Kind == "WDAT").Sum(s => s.Length));
        Artifacts = new ArtifactsViewModel(info.ArtifactSets);
        Metadata = new MetadataViewModel(definition);
        Sections.ShowRequested += ShowSection;
    }

    /// <summary>The visible tab (<see cref="InspectorTab"/>).</summary>
    [ObservableProperty]
    public partial int SelectedTab { get; set; }

    /// <summary>The package structure.</summary>
    public PackageInfo Info { get; }

    /// <summary>Overview tab.</summary>
    public OverviewViewModel Overview { get; }

    /// <summary>Graphs tab.</summary>
    public GraphsViewModel Graphs { get; }

    /// <summary>Weights tab.</summary>
    public WeightsViewModel Weights { get; }

    /// <summary>Artifacts tab.</summary>
    public ArtifactsViewModel Artifacts { get; }

    /// <summary>Metadata tab.</summary>
    public MetadataViewModel Metadata { get; }

    /// <summary>Sections tab.</summary>
    public SectionsViewModel Sections { get; }

    /// <summary>Shows the tab that decodes a file region (for an artifact set, selects it).</summary>
    public void ShowSection(SectionRow? row)
    {
        if (row is null || !row.CanShow)
        {
            return;
        }

        if (row.View.Tab == InspectorTab.Artifacts)
        {
            Artifacts.Select(row.Offset);
        }

        SelectedTab = (int)row.View.Tab;
    }

    /// <summary>
    /// Reads the package structure and decodes its definition: header, section table, weight index, definition and artifact
    /// directories, but no weight bytes and no artifact entry bytes. A definition that does not decode with this version is reported
    /// in the overview; the rest of the package is still shown.
    /// </summary>
    /// <exception cref="InvalidDataException">The file is not a supported package or is damaged.</exception>
    /// <exception cref="IOException">The file cannot be read.</exception>
    public static PackageInspection Read(string path, ModelManifest? manifest)
    {
        var info = ModelPackageReader.Open(path);
        DefinitionInfo? definition = null;
        string? error = null;
        try
        {
            definition = info.Definition;
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException or KeyNotFoundException or EndOfStreamException)
        {
            error = e.Message;
        }

        return new PackageInspection(info, definition, error, manifest);
    }
}
