namespace Sharplight.Studio.Services.Inspection;

/// <summary>
/// The view of each section kind of the package format (<see cref="Sharplight.Package.ModelPackageReader.SectionKinds"/>), plus
/// the regions that are not sections (header, section table, unused space). Unknown kinds are shown as raw byte ranges.
/// </summary>
internal static class SectionViews
{
    /// <summary>Pseudo kind of the file header.</summary>
    public const string HeaderKind = "HEAD";

    /// <summary>Pseudo kind of the section table.</summary>
    public const string TableKind = "TABLE";

    /// <summary>Pseudo kind of bytes no section or table covers (superseded sets, alignment padding).</summary>
    public const string UnusedKind = "FREE";

    private static readonly Dictionary<string, SectionView> s_views = new(StringComparer.Ordinal)
    {
        [HeaderKind] = new("Header", "Magic, format version and the location of the section table.", InspectorTab.Overview, true),
        ["DEFN"] = new("Definition", "The model in Sharplight IR: graphs, states, properties and source metadata (tokenizer, chat template).",
            InspectorTab.Graphs, true),
        ["WIDX"] = new("Weight index", "Name, type, shape, offset and length of every weight.", InspectorTab.Weights, true),
        ["WDAT"] = new("Weight data", "The weight bytes in their stored types, 64 KiB aligned. Not read by the inspector.", InspectorTab.Weights, true),
        ["ARTS"] = new("Artifact set", "Precompiled artifacts for one device target: a descriptor and keyed entries (kernels, pipeline caches).",
            InspectorTab.Artifacts, true),
        ["KSET"] = new("Kernel set (version 1)", "Precompiled kernels of one device target in the version 1 layout, read as an artifact set.",
            InspectorTab.Artifacts, true),
        ["DESC"] = new("Model description", "What the model is: its kind and components (graphs, weights, configuration, attachments), as JSON.",
            InspectorTab.Overview, true),
        ["ATCH"] = new("Attachment", "A file a component refers to (tokenizer files, scheduler configuration, filter banks, voices), 64 KiB aligned.",
            InspectorTab.Overview, true),
        [TableKind] = new("Section table", "Kind, name, offset and length of every section; written last.", InspectorTab.Sections, true),
        [UnusedKind] = new("Unused", "Bytes no section covers: alignment padding, or superseded sets and old tables that compacting removes.",
            InspectorTab.Sections, true),
    };

    /// <summary>The view of <paramref name="kind"/>; a raw view for kinds this version does not know.</summary>
    public static SectionView For(string kind) => s_views.TryGetValue(kind, out var view)
        ? view
        : new SectionView($"Unknown section {kind}", "A section of a newer format version. This version skips it; only its byte range is shown.",
            InspectorTab.Sections, false);
}
