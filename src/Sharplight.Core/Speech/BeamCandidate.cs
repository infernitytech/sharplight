namespace Sharplight.Speech;

/// <summary>A one-token continuation of a running beam.</summary>
/// <param name="Beam">The index of the running beam it extends.</param>
/// <param name="Token">The next token.</param>
/// <param name="Score">The beam's score plus the token's log probability.</param>
internal readonly record struct BeamCandidate(int Beam, int Token, double Score);
