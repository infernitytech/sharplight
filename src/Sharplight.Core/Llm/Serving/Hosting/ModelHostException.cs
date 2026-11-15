namespace Sharplight.Llm.Serving.Hosting;

/// <summary>Base of the exceptions raised by <see cref="ModelHost"/> operations.</summary>
public class ModelHostException(string message, string code) : EngineException(message, code);
