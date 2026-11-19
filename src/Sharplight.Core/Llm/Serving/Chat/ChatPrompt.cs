using Sharplight.Vision;

namespace Sharplight.Llm.Serving.Chat;

/// <summary>A rendered and tokenized chat prompt; <see cref="Images"/> are the decoded images of its image placeholders, in order.</summary>
internal sealed record ChatPrompt(string Text, int[] Tokens, bool StartsInReasoning)
{
    /// <summary>The images of the prompt's placeholders, in order (empty for a text prompt).</summary>
    public IReadOnlyList<ImageInput> Images { get; init; } = [];
}
