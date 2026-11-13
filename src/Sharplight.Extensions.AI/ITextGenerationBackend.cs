using Sharplight.Llm;

namespace Sharplight.Extensions.AI;

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
