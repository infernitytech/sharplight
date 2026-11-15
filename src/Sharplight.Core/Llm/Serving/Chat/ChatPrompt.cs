namespace Sharplight.Llm.Serving.Chat;

/// <summary>A rendered and tokenized chat prompt.</summary>
internal sealed record ChatPrompt(string Text, int[] Tokens, bool StartsInReasoning);
