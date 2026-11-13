namespace Sharplight.Llm.Serving.Hosting;

public sealed class ModelNotFoundException(string name) : ModelHostException($"The model '{name}' does not exist.", "model_not_found") { public string ModelName { get; } = name; }
