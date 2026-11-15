namespace Sharplight.Llm.Serving;

/// <summary>Why a generation ended.</summary>
public enum FinishReason
{
    /// <summary>An end-of-generation token, a requested stop token or a stop string was produced.</summary>
    Stop,

    /// <summary>The token budget or the context window was exhausted.</summary>
    Length,

    /// <summary>The caller cancelled the request, or the engine shut down.</summary>
    Cancelled,

    /// <summary>Execution failed; <see cref="GenerationChunk.Error"/> has the message.</summary>
    Error,

    /// <summary>The request's time budget or queue time ran out.</summary>
    Timeout,
}
