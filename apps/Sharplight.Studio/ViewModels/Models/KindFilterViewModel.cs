using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A choice of the library's kind filter: one model kind with the number of library models of it, or all kinds.</summary>
internal sealed class KindFilterViewModel
{
    /// <param name="info">The kind, or null for all kinds.</param>
    /// <param name="count">How many library models there are of it.</param>
    public KindFilterViewModel(ModelKindInfo? info, int count)
    {
        Info = info;
        Count = count;
    }

    /// <summary>The kind, or null for the "All" choice.</summary>
    public ModelKindInfo? Info { get; }

    /// <summary>Number of library models of this kind (of all kinds for "All").</summary>
    public int Count { get; }

    /// <summary>The choice that lists every kind.</summary>
    public bool IsAll => Info is null;

    /// <summary>The kind id, or null for "All".</summary>
    public string? Kind => Info?.Id;

    /// <summary>The label: <c>Image generation (2)</c>.</summary>
    public string Title => $"{Info?.Title ?? "All kinds"} ({Count})";

    /// <summary>The kind's icon resource key.</summary>
    public string Icon => Info?.Icon ?? "IconModels";
}
