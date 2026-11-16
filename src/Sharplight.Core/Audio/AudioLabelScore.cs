namespace Sharplight.Audio;

/// <summary>One class of an <see cref="AudioClassification"/>.</summary>
/// <param name="Index">The position of the label in the candidates.</param>
/// <param name="Label">The candidate text.</param>
/// <param name="Score">The probability (softmax over the candidates), in 0..1.</param>
/// <param name="Logit">The scaled similarity the score was computed from.</param>
public sealed record AudioLabelScore(int Index, string Label, float Score, float Logit);
