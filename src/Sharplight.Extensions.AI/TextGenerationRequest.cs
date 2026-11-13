using Sharplight.Llm;
using Sharplight.Vision;

namespace Sharplight.Extensions.AI;

/// <summary>One generation request: an already chat-templated prompt (control tokens as text), optional images and the generation options.</summary>
public sealed class TextGenerationRequest
{
    /// <summary>The rendered prompt. Image placeholders must appear in the same order as <see cref="Images"/>.</summary>
    public required string Prompt { get; init; }
    /// <summary>Decoded images, in placeholder order; empty for text-only prompts.</summary>
    public IReadOnlyList<ImageInput> Images { get; init; } = [];
    /// <summary>Token budget, sampling and stop strings for this generation.</summary>
    public required GenerationOptions Options { get; init; }
}
