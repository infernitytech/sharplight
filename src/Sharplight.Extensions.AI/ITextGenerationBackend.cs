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
    /// <summary>Model name reported in responses (unless <see cref="SharplightChatClientOptions.ModelName"/> overrides it).</summary>
    public string ModelName { get; }
    /// <summary>The Jinja chat template (the <c>tokenizer.chat_template</c> metadata), or null if the model has none.</summary>
    public string? ChatTemplate { get; }
    /// <summary>Text of the beginning-of-sequence token, passed to the chat template; null when the model has none.</summary>
    public string? BosToken { get; }
    /// <summary>Text of the end-of-sequence token, passed to the chat template; null when the model has none.</summary>
    public string? EosToken { get; }
    /// <summary>True when requests may carry images (the model has a vision tower).</summary>
    public bool SupportsImages { get; }
    /// <summary>Context window in tokens (prompt plus generated text); <see cref="int.MaxValue"/> while not yet known.</summary>
    public int ContextLength { get; }

    /// <summary>
    /// Streams the generated text pieces. Stop strings in <see cref="GenerationOptions.StopStrings"/> may be included in the
    /// output (the client trims them). <paramref name="result"/> is complete after the enumeration finishes. Cancelling
    /// <paramref name="cancellationToken"/> (or abandoning the enumeration) stops generation.
    /// </summary>
    /// <exception cref="NotSupportedException">The request carries images but <see cref="SupportsImages"/> is false.</exception>
    /// <exception cref="InvalidOperationException">The prompt does not fit the context window.</exception>
    public IAsyncEnumerable<string> GenerateAsync(TextGenerationRequest request, TextGenerationResult result, CancellationToken cancellationToken = default);
}
