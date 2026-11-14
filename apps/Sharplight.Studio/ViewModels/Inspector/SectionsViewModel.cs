using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Package;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Inspection;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>
/// The sections tab: the whole file as a list of regions in offset order (header, every section of the section table including
/// kinds this version does not know, the section table itself, and unused bytes between them), with the bytes per kind.
/// </summary>
internal sealed partial class SectionsViewModel : ObservableObject
{
    /// <param name="info">The package.</param>
    public SectionsViewModel(PackageInfo info)
    {
        double file = Math.Max(1, info.FileSize);
        var regions = new List<SectionRow>
        {
            new(SectionViews.HeaderKind, "", 0, info.HeaderLength, 100 * info.HeaderLength / file),
        };
        regions.AddRange(info.Sections.Select(s => new SectionRow(s.Kind, s.Name, s.Offset, s.Length, 100 * s.Length / file)));
        if (info.SectionTableLength > 0)
        {
            regions.Add(new SectionRow(SectionViews.TableKind, "", info.SectionTableOffset, info.SectionTableLength, 100 * info.SectionTableLength / file));
        }

        regions.Sort((a, b) => a.Offset.CompareTo(b.Offset));
        var rows = new List<SectionRow>();
        long end = 0;
        foreach (var region in regions)
        {
            if (region.Offset > end)
            {
                rows.Add(new SectionRow(SectionViews.UnusedKind, "", end, region.Offset - end, 100 * (region.Offset - end) / file));
            }

            rows.Add(region);
            end = Math.Max(end, region.Offset + region.Length);
        }

        if (info.FileSize > end)
        {
            rows.Add(new SectionRow(SectionViews.UnusedKind, "", end, info.FileSize - end, 100 * (info.FileSize - end) / file));
        }

        Rows = rows;
        KindTotals = [.. rows.GroupBy(r => r.Kind)
            .Select(g => new ShareRow(g.First().Title, g.Count() == 1 ? g.Key : $"{g.Key} × {g.Count()}", g.Sum(r => r.Length), g.Sum(r => r.Percent)))
            .OrderByDescending(r => r.Bytes)];
        Summary = $"{info.Sections.Count} sections in {ByteFormat.Exact(info.FileSize)}, format version {info.Version}";
        SelectedRow = rows.FirstOrDefault(r => r.Kind == "DEFN") ?? rows.FirstOrDefault();
    }

    /// <summary>Every region in offset order.</summary>
    public IReadOnlyList<SectionRow> Rows { get; }

    /// <summary>Bytes per kind, largest first.</summary>
    public IReadOnlyList<ShareRow> KindTotals { get; }

    /// <summary>Count, file size and version.</summary>
    public string Summary { get; }

    /// <summary>The region whose details are shown.</summary>
    [ObservableProperty]
    public partial SectionRow? SelectedRow { get; set; }

    /// <summary>Raised when the tab that decodes a region is asked for.</summary>
    public event Action<SectionRow>? ShowRequested;

    /// <summary>Asks for the tab that decodes <paramref name="row"/> (nothing for regions without one).</summary>
    [RelayCommand]
    public void Show(SectionRow? row)
    {
        if (row is { CanShow: true })
        {
            ShowRequested?.Invoke(row);
        }
    }
}
