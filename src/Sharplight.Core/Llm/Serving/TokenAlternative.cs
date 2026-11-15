namespace Sharplight.Llm.Serving;

/// <summary>One of the most likely tokens at a position.</summary>
/// <param name="TokenId">Vocabulary id.</param>
/// <param name="Token">The token's text.</param>
/// <param name="Logprob">Natural-log probability.</param>
public readonly record struct TokenAlternative(int TokenId, string Token, float Logprob);
