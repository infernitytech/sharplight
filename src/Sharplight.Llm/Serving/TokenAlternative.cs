namespace Sharplight.Llm.Serving;

public readonly record struct TokenAlternative(int TokenId, string Token, float Logprob);
