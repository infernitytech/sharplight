namespace Sharplight.Llm.Serving;

/// <summary>The engine is draining or disposed and accepts no new requests (code <c>shutting_down</c>).</summary>
public sealed class EngineStoppedException() : EngineException("The engine is shutting down and accepts no new requests.", "shutting_down");
