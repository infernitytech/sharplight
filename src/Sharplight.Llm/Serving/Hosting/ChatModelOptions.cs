using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Speculative;

namespace Sharplight.Llm.Serving.Hosting;

/// <summary>How a chat model is served: batching slots, engine and chat-service settings.</summary>
public sealed record ChatModelOptions
{
    /// <summary>Concurrent sequences batched per forward pass, each with the load option's context length.</summary>
    public int Slots { get; init; } = 4;
    /// <summary>
    /// Paged KV cache shared by the slots, in tokens (null: every slot reserves the full context). With paging, admission is
    /// bounded by free cache blocks rather than slots, so <see cref="Slots"/> can exceed what fixed per-slot contexts would fit.
    /// </summary>
    public long? KvCacheTokens { get; init; }
    /// <summary>Scheduler settings of the model's engine.</summary>
    public EngineOptions Engine { get; init; } = new();
    /// <summary>Name reported by the API is the registration name; the other settings (chat template override, tool formats, default max tokens) apply.</summary>
    public ModelServiceOptions Service { get; init; } = new();
    /// <summary>
    /// Creates the speculator of each loaded instance (speculative decoding), or null for none. A factory rather than an
    /// instance because the engine owns and disposes its speculator, and a model may be loaded again after eviction.
    /// The decoder is then sized for the verification rows (<see cref="ISpeculator.MaxDraftTokens"/> + 1 per speculating
    /// sequence); models without rollback support (recurrent state) run without speculation.
    /// </summary>
    public Func<LlmModel, ISpeculator?>? Speculator { get; init; }
}
