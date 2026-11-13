using Sharplight.Llm;
using Sharplight.Vision;

namespace Sharplight.Extensions.AI;

/// <summary>One generation request: an already chat-templated prompt (control tokens as text), optional images and the generation options.</summary>
public sealed class TextGenerationRequest
{
    /// <summary>The rendered prompt. Image placeholders must appear in the same order as <see cref="Images"/>.</summary>
    public required string Prompt { get; init; }
    public IReadOnlyList<ImageInput> Images { get; init; } = [];
    public required GenerationOptions Options { get; init; }
}
