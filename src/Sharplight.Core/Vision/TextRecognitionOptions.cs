namespace Sharplight.Vision;

/// <summary>Per-request settings of <see cref="ITextRecognizer"/>.</summary>
public sealed record TextRecognitionOptions
{
    /// <summary>The most tokens the decoder generates (the text of one image).</summary>
    public int MaxTokens { get; init; } = 128;

    /// <summary>
    /// The task of a model that is prompted (Florence-2: <c>&lt;OCR&gt;</c>, <c>&lt;CAPTION&gt;</c>, <c>&lt;DETAILED_CAPTION&gt;</c>, <c>&lt;OD&gt;</c>, ... the names the model lists in
    /// <see cref="ITextRecognizer"/>'s model description); <see langword="null"/> selects the first task. Models without prompts ignore it.
    /// </summary>
    public string? Task { get; init; }

    /// <summary>The text a task refers to (the caption to ground, the phrase to locate); <see langword="null"/> for tasks without input.</summary>
    public string? Input { get; init; }
}
