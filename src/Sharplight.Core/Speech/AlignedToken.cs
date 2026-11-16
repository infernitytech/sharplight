namespace Sharplight.Speech;

/// <summary>A token of a frame-aligned decoding (CTC, transducers) with the encoder rows it covers.</summary>
/// <param name="Token">The token id.</param>
/// <param name="Row">The encoder row the token was emitted at.</param>
/// <param name="Rows">The encoder rows the token spans (at least 1).</param>
/// <param name="LogProbability">The log probability of the token at its row.</param>
internal readonly record struct AlignedToken(int Token, int Row, int Rows, float LogProbability);
