namespace Sharplight.Llm.Serving;

public sealed class PromptTooLongException(int tokens, int max) : EngineException($"The prompt has {tokens} tokens, but the model context holds at most {max} (a prompt needs room for at least one generated token).", "context_length_exceeded")
{
    public int PromptTokens { get; } = tokens;
    public int MaxTokens { get; } = max;
}
