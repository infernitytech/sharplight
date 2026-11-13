namespace Sharplight.Llm.Serving;

/// <summary>Log-probability of a token and of the most likely alternatives.</summary>
/// <param name="TokenId">Vocabulary id of the sampled token.</param>
/// <param name="Token">The sampled token's text.</param>
/// <param name="Logprob">Natural-log probability of the sampled token under the model (before sampling filters).</param>
/// <param name="TopAlternatives">The most likely tokens at this position, best first (empty unless requested).</param>
public sealed record TokenLogprob(int TokenId, string Token, float Logprob, IReadOnlyList<TokenAlternative> TopAlternatives);
