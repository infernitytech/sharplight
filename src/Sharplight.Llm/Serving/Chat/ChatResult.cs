namespace Sharplight.Llm.Serving.Chat;

/// <summary>The complete outcome of a <see cref="ChatRun"/>.</summary>
public sealed class ChatResult
{
    /// <summary>One entry per requested choice, ordered by <see cref="ChatChoice.Index"/>.</summary>
    public IReadOnlyList<ChatChoice> Choices { get; init; } = [];
    /// <summary>Prompt tokens counted once, completion tokens summed over the choices.</summary>
    public GenerationUsage Usage { get; init; }
}
