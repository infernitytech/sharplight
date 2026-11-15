namespace Sharplight.Llm.Serving.Hosting;

/// <summary>No model is registered under the requested name.</summary>
public sealed class ModelNotFoundException(string name) : ModelHostException($"The model '{name}' does not exist.", "model_not_found") 
{
    /// <summary>The requested name.</summary>
    public string ModelName { get; } = name;
}
