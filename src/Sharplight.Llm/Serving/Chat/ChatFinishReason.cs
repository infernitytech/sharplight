namespace Sharplight.Llm.Serving.Chat;

/// <summary>Why a chat completion ended.</summary>
public enum ChatFinishReason
{
    /// <summary>The model finished its answer or a stop string / token was produced.</summary>
    Stop,

    /// <summary>The token budget or the context window was exhausted.</summary>
    Length,

    /// <summary>The model ended its turn with one or more tool calls.</summary>
    ToolCalls,

    /// <summary>The request was cancelled or the engine shut down.</summary>
    Cancelled,

    /// <summary>Execution failed.</summary>
    Error,

    /// <summary>The time budget ran out.</summary>
    Timeout,
}
