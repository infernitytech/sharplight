namespace Sharplight.Llm.Serving;

/// <summary>Base of the exceptions thrown when a request is not admitted.</summary>
public class EngineException(string message, string code) : Exception(message)
{
    /// <summary>Machine-readable reason: <c>queue_full</c>, <c>context_length_exceeded</c>, <c>shutting_down</c>, <c>invalid_request</c>.</summary>
    public string Code { get; } = code;
}
