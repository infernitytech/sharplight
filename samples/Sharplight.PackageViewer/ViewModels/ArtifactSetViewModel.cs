using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Package;
using Sharplight.Runtime;

namespace Sharplight.PackageViewer.ViewModels;

/// <summary>One stored artifact set: descriptor, sizes and its (filterable) entry list.</summary>
public sealed partial class ArtifactSetViewModel : ObservableObject
{
    private readonly IReadOnlyList<ArtifactEntryRow> _allEntries;

    public ArtifactSetViewModel(ArtifactSetInfo set)
    {
        Id = set.Id;
        Provider = set.Provider;
        Tier = set.Tier;
        Kind = set.Kind;
        IsLegacy = set.IsLegacy;
        EntryCount = set.EntryCount;
        Descriptor = [.. set.Descriptor.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => new KeyValueRow("", kv.Key, kv.Value))];
        _allEntries = [.. set.Entries.Select(e => new ArtifactEntryRow(e.Key, e.Size))];
        Entries = _allEntries;
        Sizes = $"{set.EntryCount:N0} entries, {ByteSizeFormat.FormatExact(set.TotalBytes)}; section {ByteSizeFormat.Format(set.Length)} at offset {set.Offset:N0}";
        Label = $"{set.Kind}: {set.EntryCount:N0} entries, {ByteSizeFormat.Format(set.TotalBytes)}";
        DistinguishingDescriptor = ArtifactDescriptor.Format(set.Descriptor
            .Where(kv => kv.Key is not (ArtifactDescriptorKeys.Provider or ArtifactDescriptorKeys.Tier or ArtifactDescriptorKeys.Kind))
            .ToDictionary());
        TotalBytes = set.TotalBytes;
    }

    /// <summary>Section name (descriptor id, or the device target of a version 1 kernel set).</summary>
    public string Id { get; }

    public string Provider { get; }

    public string Tier { get; }

    public string Kind { get; }

    /// <summary>Whether this is a version 1 kernel set (KSET).</summary>
    public bool IsLegacy { get; }

    public int EntryCount { get; }

    public long TotalBytes { get; }

    /// <summary>Tree label: kind, entry count and size.</summary>
    public string Label { get; }

    /// <summary>The descriptor without provider, tier and kind (which the tree already shows).</summary>
    public string DistinguishingDescriptor { get; }

    /// <summary>Entry count, entry bytes, section size and offset.</summary>
    public string Sizes { get; }

    public IReadOnlyList<KeyValueRow> Descriptor { get; }

    /// <summary>Filter of the entry list (whitespace-separated terms matched against the key).</summary>
    [ObservableProperty]
    public partial string? EntryFilter { get; set; }

    [ObservableProperty]
    public partial IReadOnlyList<ArtifactEntryRow> Entries { get; private set; }

    partial void OnEntryFilterChanged(string? value) => Entries = TextFilter.Apply(_allEntries, value);
}
