namespace Sharplight.Vision;

/// <summary>Settings of <see cref="DocumentReader"/>.</summary>
public sealed class DocumentReaderOptions
{
    /// <summary>Layout regions scoring below this are dropped.</summary>
    public float ScoreThreshold { get; set; } = 0.5f;

    /// <summary>The most tokens the recognizer generates for one region.</summary>
    public int MaxTokensPerRegion { get; set; } = 1024;

    /// <summary>Pixels added around a region's box before it is cropped and read (a tight box can cut the first and last glyphs).</summary>
    public int Padding { get; set; } = 4;

    /// <summary>Layout labels that are not read (pictures have no text to recognize).</summary>
    public ISet<string> SkippedLabels { get; } = new HashSet<string>(["image", "chart", "seal"], StringComparer.Ordinal);

    /// <summary>
    /// The recognition task (<see cref="TextRecognitionOptions.Task"/>) per layout label, for labels that need another than the default, e.g. <c>table</c> to <c>table</c>
    /// and <c>formula</c> to <c>formula</c>. A label whose task the recognizer does not list is read with its default task.
    /// </summary>
    public IDictionary<string, string> TaskByLabel { get; } = new Dictionary<string, string>(StringComparer.Ordinal) { ["table"] = "table", ["formula"] = "formula" };
}
