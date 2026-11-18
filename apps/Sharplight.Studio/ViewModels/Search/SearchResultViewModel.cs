using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Search;
using CommunityToolkit.Mvvm.ComponentModel;

namespace Sharplight.Studio.ViewModels.Search;

/// <summary>A passage in the result list.</summary>
internal sealed partial class SearchResultViewModel : ObservableObject
{
    /// <param name="hit">The hit.</param>
    /// <param name="folder">The searched folder (file names are shown relative to it).</param>
    /// <param name="number">The rank, 1-based; answers cite passages by it.</param>
    /// <param name="query">The question, whose words are highlighted.</param>
    public SearchResultViewModel(SearchHitRecord hit, string folder, int number = 0, string query = "")
    {
        Hit = hit;
        Number = number;
        FileName = folder.Length > 0 && hit.File.StartsWith(folder, StringComparison.Ordinal) ? Path.GetRelativePath(folder, hit.File) : hit.File;
        Segments = QueryHighlighter.Split(hit.Text, query);
    }

    /// <summary>The hit.</summary>
    public SearchHitRecord Hit { get; }

    /// <summary>The rank (1-based), or 0 when unknown.</summary>
    public int Number { get; }

    /// <summary>The rank as answers cite it (<c>[2]</c>).</summary>
    public string NumberText => Number > 0 ? $"[{Number}]" : "";

    /// <summary>The label of the citation button under an answer (<c>[2] notes.md</c>).</summary>
    public string Citation => $"{NumberText} {Path.GetFileName(Hit.File)}";

    /// <summary>The document, relative to the searched folder.</summary>
    public string FileName { get; }

    /// <summary>The place in the document (<c>page 3 · Introduction</c>, <c>line 41</c>), or empty.</summary>
    public string Location => LocationLabel.For(Hit.File, Hit.Page, Hit.Line, Hit.Heading);

    /// <summary>The passage.</summary>
    public string Text => Hit.Text;

    /// <summary>The passage in plain and highlighted pieces.</summary>
    public IReadOnlyList<TextSegment> Segments { get; }

    /// <summary>The similarity, as shown (<c>0.83</c>).</summary>
    public string ScoreText => Hit.Score.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture);

    /// <summary>The similarity as a fraction for the bar (0 to 1).</summary>
    public double ScoreFraction => Math.Clamp(Hit.Score, 0, 1);

    /// <summary>The result an answer's citation pointed to last.</summary>
    [ObservableProperty]
    public partial bool IsFocused { get; set; }
}
