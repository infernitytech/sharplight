using Sharplight.Hub;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A search result.</summary>
internal sealed class HubResultViewModel
{
    public HubResultViewModel(HubModelSummary summary)
    {
        Summary = summary;
    }

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
    public string Formats => string.Join(" · ", Summary.Formats.Select(format => format.ToString().ToUpperInvariant()));

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
