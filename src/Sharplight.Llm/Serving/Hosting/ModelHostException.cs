namespace Sharplight.Llm.Serving.Hosting;

public class ModelHostException(string message, string code) : EngineException(message, code);
