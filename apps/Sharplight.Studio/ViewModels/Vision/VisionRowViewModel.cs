using Sharplight.Studio.Services.Imaging;

namespace Sharplight.Studio.ViewModels.Vision;

/// <summary>One line of a vision result: a class with its score (a bar), or a detected object with the color of its box.</summary>
internal sealed class VisionRowViewModel
{
    /// <param name="label">The class name.</param>
    /// <param name="score">The score, 0 to 1.</param>
    /// <param name="detail">More about the row (the box of a detection), or empty.</param>
    /// <param name="colorIndex">The class index whose box color the row shows, or -1 for no color.</param>
    public VisionRowViewModel(string label, float score, string detail = "", int colorIndex = -1)
    {
        Label = label;
        Score = score;
        Detail = detail;
        Color = colorIndex < 0 ? null : $"#{ImageAnnotator.ColorOf(colorIndex):X6}";
    }

    /// <summary>The class name.</summary>
    public string Label { get; }

    /// <summary>The score, 0 to 1.</summary>
    public float Score { get; }

    /// <summary>The score as text.</summary>
    public string ScoreText => Percent(Score);

    /// <summary>A score as a percentage without a space before the sign (<c>93%</c>, <c>4.5%</c>), whatever the culture.</summary>
    /// <param name="score">The score, 0 to 1.</param>
    public static string Percent(float score)
        => (score >= 0.1f ? (score * 100).ToString("0", System.Globalization.CultureInfo.InvariantCulture) : (score * 100).ToString("0.0", System.Globalization.CultureInfo.InvariantCulture)) + "%";

    /// <summary>More about the row, or empty.</summary>
    public string Detail { get; }

    /// <summary>The color of the box in the annotated image (<c>#RRGGBB</c>), or null.</summary>
    public string? Color { get; }

    /// <summary>The row has a box color.</summary>
    public bool HasColor => Color is not null;
}
