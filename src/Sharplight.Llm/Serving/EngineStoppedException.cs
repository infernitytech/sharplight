namespace Sharplight.Llm.Serving;

public sealed class EngineStoppedException() : EngineException("The engine is shutting down and accepts no new requests.", "shutting_down");
