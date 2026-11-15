using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Hub;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Compatibility;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A search result.</summary>
internal sealed partial class HubResultViewModel : ObservableObject
{
    public HubResultViewModel(HubModelSummary summary)
    {
        Summary = summary;
        Compatibility = new CompatibilityBadgeViewModel(CompatibilityChecker.FromSummary(summary));
    }

    /// <summary>
    /// The likely compatibility: from the search summary at first, from the files' metadata once the repository was opened.
    /// </summary>
    [ObservableProperty]
    public partial CompatibilityBadgeViewModel Compatibility { get; set; }

    /// <summary>This repository's details are shown (highlight cards mark it).</summary>
    [ObservableProperty]
    public partial bool IsSelected { get; set; }

    /// <summary>The Hub's data.</summary>
    public HubModelSummary Summary { get; }

    /// <summary><c>owner/name</c>.</summary>
    public string Id => Summary.Id;

    /// <summary>The owner.</summary>
    public string Owner => Summary.Id.Split('/')[0];

    /// <summary>The repository name.</summary>
    public string Name => Summary.Id.Split('/').Last();

    /// <summary>Downloads (30 days).</summary>
    public string Downloads => ByteFormat.Count(Summary.Downloads);

    /// <summary>Likes.</summary>
    public string Likes => ByteFormat.Count(Summary.Likes);

    /// <summary>Gated repository.</summary>
    public bool Gated => Summary.Gated;

    /// <summary>Formats as text.</summary>
    public string Formats => Summary.Formats.Count == 0 ? "No model files" : string.Join(" · ", Summary.Formats.Select(format => ModelFormatTitles.Title(format).ToUpperInvariant()));

    /// <summary>How the model's kind is presented (from its Hub task).</summary>
    public ModelKindInfo Kind => ModelKindCatalog.For(Summary.Task);

    /// <summary>The kind's name (<c>Image generation</c>), or the task for models whose kind is not known (no pipeline tag).</summary>
    public string KindTitle => Summary.Task == Sharplight.Hub.ModelTask.Other ? "" : Kind.Title;

    /// <summary>The card shows a kind badge (the repository names a task).</summary>
    public bool HasKind => Summary.Task != Sharplight.Hub.ModelTask.Other;

    /// <summary>The kind's icon resource key.</summary>
    public string KindIcon => Kind.Icon;

    /// <summary>The license id, or null.</summary>
    public string? License => Summary.License;

    /// <summary>Parameter count, e.g. <c>4.0B params</c>.</summary>
    public string? Parameters => Summary.Parameters is { } count ? ByteFormat.Count(count) + " params" : null;

    /// <summary>Pipeline tag.</summary>
    public string? Pipeline => Summary.Pipeline;

    /// <summary>Number of variants.</summary>
    public string Variants => Summary.Variants.Count switch
    {
        0 => "single variant",
        1 => "1 variant",
        var count => $"{count} variants",
    };
}
