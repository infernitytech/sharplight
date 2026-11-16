namespace Sharplight.Studio.ViewModels.Audio;

/// <summary>One candidate of a zero-shot tagging result: the label and its probability as a bar.</summary>
internal sealed class TagScoreViewModel
{
    private const double BarLength = 320;

    /// <param name="label">The candidate text.</param>
    /// <param name="score">The probability (0 to 1).</param>
    public TagScoreViewModel(string label, double score)
    {
        Label = label;
        Score = score;
    }

    /// <summary>The candidate text.</summary>
    public string Label { get; }

    /// <summary>The probability.</summary>
    public double Score { get; }

    /// <summary>The width of the bar.</summary>
    public double BarWidth => Math.Max(2, Score * BarLength);

    /// <summary>The probability as text.</summary>
    public string Text => $"{Score * 100:0.0} %";
}
