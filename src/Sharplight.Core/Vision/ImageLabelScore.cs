namespace Sharplight.Vision;

/// <summary>One class of an <see cref="ImageClassification"/>.</summary>
/// <param name="Index">The class index (the position of the label in the model's label list, or in the zero-shot candidates).</param>
/// <param name="Label">The class name.</param>
/// <param name="Score">The probability (softmax models) or independent confidence (sigmoid models), in 0..1.</param>
/// <param name="Logit">The raw model output the score was computed from.</param>
public sealed record ImageLabelScore(int Index, string Label, float Score, float Logit);
