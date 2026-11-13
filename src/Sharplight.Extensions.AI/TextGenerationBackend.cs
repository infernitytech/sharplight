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

/// <summary>Filled in by the backend while a generation runs; complete once the text stream has ended.</summary>
public sealed class TextGenerationResult
{
    public int PromptTokens { get; set; }
    /// <summary>Generated tokens, including a terminating end-of-generation token.</summary>
    public int GeneratedTokens { get; set; }
    /// <summary>True when generation ended because <see cref="GenerationOptions.MaxNewTokens"/> was reached.</summary>
    public bool ReachedTokenLimit { get; set; }
}

/// <summary>
/// What <see cref="SharplightChatClient"/> needs from an inference engine: the model's chat template and a way to stream
/// text for a rendered prompt. <see cref="SessionGenerationBackend"/> runs one generation at a time on an
/// <see cref="InferenceSession"/>; a continuous-batching serving engine can implement this interface to let many
/// concurrent chat calls share the model (implementations must be safe for concurrent callers).
/// </summary>
public interface ITextGenerationBackend : IDisposable
{
    public string ModelName { get; }
    /// <summary>The Jinja chat template (the <c>tokenizer.chat_template</c> metadata), or null if the model has none.</summary>
    public string? ChatTemplate { get; }
    public string? BosToken { get; }
    public string? EosToken { get; }
    public bool SupportsImages { get; }
    public int ContextLength { get; }

    /// <summary>
    /// Streams the generated text pieces. Stop strings in <see cref="GenerationOptions.StopStrings"/> may be included in the
    /// output (the client trims them). <paramref name="result"/> is complete after the enumeration finishes.
    /// </summary>
    public IAsyncEnumerable<string> GenerateAsync(TextGenerationRequest request, TextGenerationResult result, CancellationToken cancellationToken = default);
}
