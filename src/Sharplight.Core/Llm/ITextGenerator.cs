using Sharplight.Tasks;

namespace Sharplight.Llm;

/// <summary>
/// Text generation (kind <see cref="Models.ModelKinds.TextGeneration"/>): a prompt in, generated text out. The text-generation
/// instance of the task-client pattern shared by every model kind (<see cref="ITaskClient"/>). Each request starts from an empty
/// context; for multi-turn chat with prefix reuse and continuous batching use <see cref="Serving.InferenceEngine"/> and
/// <see cref="Serving.Chat.ModelService"/>. Created with <see cref="LlmModel.CreateTextGenerator"/>, from a <see cref="TaskModel"/> through
/// <see cref="TaskClientProvider"/>, or resolved from the service container (<c>services.AddTextGenerator(...)</c>). Not thread-safe.
/// </summary>
public interface ITextGenerator : ITaskClient
{
    /// <summary>Generates a whole completion.</summary>
    /// <param name="prompt">The prompt text (special tokens in it are parsed).</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Stops generation between tokens.</param>
    /// <returns>The generated text, its tokens and timings.</returns>
    public Task<TextGenerationResult> GenerateAsync(string prompt, GenerationOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Streams a completion token by token.</summary>
    /// <param name="prompt">The prompt text (special tokens in it are parsed).</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Stops generation between tokens.</param>
    public IAsyncEnumerable<GeneratedToken> GenerateStreamingAsync(string prompt, GenerationOptions? options = null, CancellationToken cancellationToken = default);
}
