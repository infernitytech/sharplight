namespace Sharplight.Studio.Services.Inspection;

/// <summary>The tabs of the package inspector, in display order (the values are the tab indexes).</summary>
internal enum InspectorTab
{
    /// <summary>File, model, source, tokenizer and chat template.</summary>
    Overview,

    /// <summary>The stored graphs: graph view, IR and node list.</summary>
    Graphs,

    /// <summary>The weight index.</summary>
    Weights,

    /// <summary>Precompiled artifact sets.</summary>
    Artifacts,

    /// <summary>Model properties and source metadata.</summary>
    Metadata,

    /// <summary>The file layout: header, every section, the section table and unused space.</summary>
    Sections,
}
