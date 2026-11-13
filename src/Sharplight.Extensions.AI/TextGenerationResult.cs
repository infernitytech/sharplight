using Sharplight.Llm;

namespace Sharplight.Extensions.AI;

/// <summary>Filled in by the backend while a generation runs; complete once the text stream has ended.</summary>
public sealed class TextGenerationResult
{
    /// <summary>Prompt length in tokens (including image tokens).</summary>
    public int PromptTokens { get; set; }
    /// <summary>Generated tokens, including a terminating end-of-generation token.</summary>
    public int GeneratedTokens { get; set; }
    /// <summary>True when generation ended because <see cref="GenerationOptions.MaxNewTokens"/> was reached.</summary>
    public bool ReachedTokenLimit { get; set; }
}
