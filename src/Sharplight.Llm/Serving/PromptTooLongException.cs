namespace Sharplight.Llm.Serving;

/// <summary>The prompt leaves no room for a generated token in the context window (code <c>context_length_exceeded</c>).</summary>
/// <param name="tokens">Prompt length.</param>
/// <param name="max">Longest acceptable prompt.</param>
public sealed class PromptTooLongException(int tokens, int max) : EngineException($"The prompt has {tokens} tokens, but the model context holds at most {max} (a prompt needs room for at least one generated token).", "context_length_exceeded")
{
    /// <summary>Length of the rejected prompt.</summary>
    public int PromptTokens { get; } = tokens;
    /// <summary>Longest prompt the engine accepts.</summary>
    public int MaxTokens { get; } = max;
}
