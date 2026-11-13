namespace Sharplight.Llm.Serving;

/// <summary>Log-probability of a token and of the most likely alternatives.</summary>
public sealed record TokenLogprob(int TokenId, string Token, float Logprob, IReadOnlyList<TokenAlternative> TopAlternatives);
