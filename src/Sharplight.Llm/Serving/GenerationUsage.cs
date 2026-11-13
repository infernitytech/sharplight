namespace Sharplight.Llm.Serving;

/// <summary>Token accounting of one finished request.</summary>
/// <param name="PromptTokens">Length of the prompt.</param>
/// <param name="CompletionTokens">Generated tokens.</param>
/// <param name="CachedTokens">Prompt tokens served from the prefix cache instead of being computed (part of <paramref name="PromptTokens"/>).</param>
public readonly record struct GenerationUsage(int PromptTokens, int CompletionTokens, int CachedTokens)
{
    /// <summary>Prompt plus completion tokens.</summary>
    public int TotalTokens => PromptTokens + CompletionTokens;
}
