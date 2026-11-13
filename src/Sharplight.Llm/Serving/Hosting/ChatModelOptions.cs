using Sharplight.Llm.Serving.Chat;

namespace Sharplight.Llm.Serving.Hosting;

/// <summary>How a chat model is served: batching slots, engine and chat-service settings.</summary>
public sealed record ChatModelOptions
{
    /// <summary>Concurrent sequences batched per forward pass, each with the load option's context length.</summary>
    public int Slots { get; init; } = 4;
    /// <summary>Scheduler settings of the model's engine.</summary>
    public EngineOptions Engine { get; init; } = new();
    /// <summary>Name reported by the API is the registration name; the other settings (chat template override, tool formats, default max tokens) apply.</summary>
    public ModelServiceOptions Service { get; init; } = new();
}
