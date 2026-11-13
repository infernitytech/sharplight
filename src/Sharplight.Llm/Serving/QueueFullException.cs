namespace Sharplight.Llm.Serving;

public sealed class QueueFullException(int capacity) : EngineException($"The request queue is full ({capacity} waiting requests).", "queue_full");
