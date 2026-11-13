namespace Sharplight.Llm.Serving.Chat;

public sealed class ChatResult
{
    public IReadOnlyList<ChatChoice> Choices { get; init; } = [];
    /// <summary>Prompt tokens counted once, completion tokens summed over the choices.</summary>
    public GenerationUsage Usage { get; init; }
}
