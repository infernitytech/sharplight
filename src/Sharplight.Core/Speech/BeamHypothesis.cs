namespace Sharplight.Speech;

/// <summary>One sequence of a beam search.</summary>
/// <param name="Tokens">The generated tokens (without the end token).</param>
/// <param name="Score">
/// Running beams: the sum of the token log probabilities. Finished beams: that sum including the end token, divided by the generated length
/// raised to the length penalty.
/// </param>
/// <param name="LogProbability">The sum of the token log probabilities (including the end token of a finished beam).</param>
internal sealed record BeamHypothesis(int[] Tokens, double Score, double LogProbability);
