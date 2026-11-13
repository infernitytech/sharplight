namespace Sharplight.Llm.Serving;

public readonly record struct GenerationUsage(int PromptTokens, int CompletionTokens, int CachedTokens)
{
    public int TotalTokens => PromptTokens + CompletionTokens;
}
