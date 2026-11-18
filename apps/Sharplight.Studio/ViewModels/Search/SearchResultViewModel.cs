using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Search;

/// <summary>A passage in the result list.</summary>
internal sealed class SearchResultViewModel
{
    public SearchResultViewModel(SearchHitRecord hit, string folder)
    {
        Hit = hit;
        string relative = folder.Length > 0 && hit.File.StartsWith(folder, StringComparison.Ordinal) ? Path.GetRelativePath(folder, hit.File) : hit.File;
        FileName = relative;
    }

    /// <summary>The hit.</summary>
    public SearchHitRecord Hit { get; }

    /// <summary>The document, relative to the searched folder.</summary>
    public string FileName { get; }

    /// <summary>The passage.</summary>
    public string Text => Hit.Text;

    /// <summary>The similarity, as shown (<c>0.83</c>).</summary>
    public string ScoreText => Hit.Score.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture);

    /// <summary>The similarity as a fraction for the bar (0 to 1).</summary>
    public double ScoreFraction => Math.Clamp(Hit.Score, 0, 1);
}
