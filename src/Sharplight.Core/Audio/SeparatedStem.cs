namespace Sharplight.Audio;

/// <summary>One stem of a separated mixture.</summary>
/// <param name="Name">The stem's name (<c>drums</c>, <c>bass</c>, <c>other</c>, <c>vocals</c>, ...).</param>
/// <param name="Audio">The stem, at the model's rate and channel count, as long as the mixture.</param>
public sealed record SeparatedStem(string Name, AudioBuffer Audio);
