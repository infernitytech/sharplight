using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Inspection;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>One stored artifact set: what it was built for, its descriptor, sizes and its (filterable, sortable) entry list.</summary>
internal sealed partial class ArtifactSetViewModel : ObservableObject
{
    private const string KeyColumn = "Key";
    private const string SizeColumn = "Size";
    private const string OffsetColumn = "Offset";

    private readonly IReadOnlyList<ArtifactEntryRow> _allEntries;

    /// <param name="set">The stored set.</param>
    public ArtifactSetViewModel(ArtifactSetInfo set)
    {
        Id = set.Id;
        Provider = set.Provider;
        Tier = set.Tier;
        Kind = set.Kind;
        IsLegacy = set.IsLegacy;
        EntryCount = set.EntryCount;
        TotalBytes = set.TotalBytes;
        SectionOffset = set.Offset;
        SectionLength = set.Length;
        Descriptor = [.. set.Descriptor.OrderBy(kv => Order(kv.Key)).ThenBy(kv => kv.Key, StringComparer.Ordinal)
            .Select(kv => new FactRow(ArtifactDescriptorLabels.Label(kv.Key), ArtifactDescriptorLabels.Value(kv.Key, kv.Value)))];
        var distinguishing = set.Descriptor
            .Where(kv => kv.Key is not (ArtifactDescriptorKeys.Provider or ArtifactDescriptorKeys.Tier or ArtifactDescriptorKeys.Kind))
            .OrderBy(kv => Order(kv.Key)).ThenBy(kv => kv.Key, StringComparer.Ordinal)
            .Select(kv => $"{ArtifactDescriptorLabels.Label(kv.Key).ToLowerInvariant()} {ArtifactDescriptorLabels.Value(kv.Key, kv.Value)}");
        BuiltFor = string.Join(" · ", distinguishing);
        _allEntries = [.. set.Entries.Select(e => new ArtifactEntryRow(e.Key, e.Size, e.Offset))];
        Entries = _allEntries;
        Sizes = $"{ByteFormat.Count(set.EntryCount, "entry", "entries")} · {ByteFormat.Exact(set.TotalBytes)} · section {ByteFormat.Format(set.Length)} at offset {ByteFormat.Number(set.Offset)}";
        EntrySummary = ByteFormat.Count(_allEntries.Count, "entry", "entries");
    }

    /// <summary>Section name (descriptor id, or the device target of a version 1 kernel set).</summary>
    public string Id { get; }

    /// <summary>Execution provider.</summary>
    public string Provider { get; }

    /// <summary>Tier (<c>code</c>, <c>binary</c>, <c>mixed</c>).</summary>
    public string Tier { get; }

    /// <summary>Artifact kind (e.g. <c>spirv</c>, <c>pipeline-cache</c>).</summary>
    public string Kind { get; }

    /// <summary>Whether this is a version 1 kernel set (KSET).</summary>
    public bool IsLegacy { get; }

    /// <summary>Number of entries.</summary>
    public int EntryCount { get; }

    /// <summary>Sum of the entry sizes.</summary>
    public long TotalBytes { get; }

    /// <summary>Absolute offset of the section.</summary>
    public long SectionOffset { get; }

    /// <summary>Length of the section.</summary>
    public long SectionLength { get; }

    /// <summary>
    /// The precompilation stage the set holds: portable code (code tier, generated for a device profile), device code (binary tier,
    /// compiled by one device's driver) or the mixed kernel sets of version 1 packages.
    /// </summary>
    public string Stage => Tier switch
    {
        ArtifactTiers.Code => "Portable code",
        ArtifactTiers.Binary => "Device code",
        ArtifactTiers.Mixed => "Legacy kernels",
        _ => Tier,
    };

    /// <summary>Sort order of <see cref="Stage"/>: portable, device, legacy, others.</summary>
    public int StageOrder => Tier switch
    {
        ArtifactTiers.Code => 0,
        ArtifactTiers.Binary => 1,
        ArtifactTiers.Mixed => 2,
        _ => 3,
    };

    /// <summary><c>provider · stage</c> for the list.</summary>
    public string Heading => IsLegacy ? $"{Provider} · {Stage} · version 1" : $"{Provider} · {Stage}";

    /// <summary>Kind, entry count and size for the list.</summary>
    public string Label => $"{Kind} · {ByteFormat.Count(EntryCount, "entry", "entries")} · {ByteFormat.Format(TotalBytes)}";

    /// <summary>The device, driver or code target the set was built for (the descriptor without provider, tier and kind).</summary>
    public string BuiltFor { get; }

    /// <summary>Entry count, entry bytes, section size and offset.</summary>
    public string Sizes { get; }

    /// <summary>The descriptor with readable labels.</summary>
    public IReadOnlyList<FactRow> Descriptor { get; }

    /// <summary>Filter of the entry list (whitespace-separated terms matched against the key).</summary>
    [ObservableProperty]
    public partial string? EntryFilter { get; set; }

    /// <summary>The entries matching <see cref="EntryFilter"/>, sorted.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<ArtifactEntryRow> Entries { get; private set; }

    /// <summary>Count of the shown entries.</summary>
    [ObservableProperty]
    public partial string EntrySummary { get; private set; }

    /// <summary>The sort column (<c>Key</c>, <c>Size</c> or <c>Offset</c>).</summary>
    [ObservableProperty]
    public partial string SortColumn { get; private set; } = KeyColumn;

    /// <summary>Largest (or last) first.</summary>
    [ObservableProperty]
    public partial bool SortDescending { get; private set; }

    /// <summary>Header of the key column.</summary>
    public string KeyHeader => Header(KeyColumn, "Key");

    /// <summary>Header of the size column.</summary>
    public string SizeHeader => Header(SizeColumn, "Size");

    /// <summary>Header of the offset column.</summary>
    public string OffsetHeader => Header(OffsetColumn, "Offset");

    /// <summary>Sorts by a column; the same column again reverses the order.</summary>
    [RelayCommand]
    public void SortBy(string column)
    {
        SortDescending = column == SortColumn ? !SortDescending : column == SizeColumn;
        SortColumn = column;
        Refresh();
        OnPropertyChanged(nameof(KeyHeader));
        OnPropertyChanged(nameof(SizeHeader));
        OnPropertyChanged(nameof(OffsetHeader));
    }

    partial void OnEntryFilterChanged(string? value) => Refresh();

    private void Refresh()
    {
        var shown = TextFilter.Apply(_allEntries, EntryFilter);
        IEnumerable<ArtifactEntryRow> ordered = SortColumn switch
        {
            SizeColumn => shown.OrderBy(e => e.Bytes).ThenBy(e => e.Key, StringComparer.Ordinal),
            OffsetColumn => shown.OrderBy(e => e.Offset),
            _ => shown.OrderBy(e => e.Key, StringComparer.Ordinal),
        };
        Entries = SortDescending ? [.. ordered.Reverse()] : [.. ordered];
        EntrySummary = Entries.Count == _allEntries.Count ? ByteFormat.Count(_allEntries.Count, "entry", "entries") : $"Showing {Entries.Count:N0} of {_allEntries.Count:N0} entries";
    }

    private string Header(string column, string text) => column != SortColumn ? text : SortDescending ? text + " ↓" : text + " ↑";

    private static int Order(string key) => key switch
    {
        ArtifactDescriptorKeys.Provider => 0,
        ArtifactDescriptorKeys.Tier => 1,
        ArtifactDescriptorKeys.Kind => 2,
        ArtifactDescriptorKeys.Target => 3,
        ArtifactDescriptorKeys.DeviceName => 4,
        ArtifactDescriptorKeys.Vendor => 5,
        ArtifactDescriptorKeys.Device => 6,
        ArtifactDescriptorKeys.Driver => 7,
        _ => 10,
    };
}
