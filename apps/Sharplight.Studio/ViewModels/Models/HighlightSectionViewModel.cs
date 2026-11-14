using Sharplight.Studio.Services.Hub;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A section of highlighted models on Discover (shown while nothing is searched).</summary>
internal sealed class HighlightSectionViewModel
{
    /// <param name="kind">Which section.</param>
    /// <param name="items">Its models, best first.</param>
    public HighlightSectionViewModel(HighlightKind kind, IReadOnlyList<HubResultViewModel> items)
    {
        Kind = kind;
        Items = items;
    }

    /// <summary>Which section.</summary>
    public HighlightKind Kind { get; }

    /// <summary>The models, in the same cards as search results.</summary>
    public IReadOnlyList<HubResultViewModel> Items { get; }

    /// <summary>The heading.</summary>
    public string Title => Kind switch
    {
        HighlightKind.Trending => "Trending",
        HighlightKind.New => "New",
        _ => "Most downloaded",
    };

    /// <summary>A line under the heading.</summary>
    public string Description => Kind switch
    {
        HighlightKind.Trending => "Popular on Hugging Face right now",
        HighlightKind.New => "Published in the last few months",
        _ => "Downloaded most in the last 30 days",
    };
}
