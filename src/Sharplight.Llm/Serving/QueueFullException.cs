namespace Sharplight.Llm.Serving;

/// <summary>The waiting queue is full (code <c>queue_full</c>); the caller may retry later.</summary>
/// <param name="capacity">The configured <see cref="EngineOptions.MaxQueue"/>.</param>
public sealed class QueueFullException(int capacity) : EngineException($"The request queue is full ({capacity} waiting requests).", "queue_full");
