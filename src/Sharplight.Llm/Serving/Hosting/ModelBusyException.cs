namespace Sharplight.Llm.Serving.Hosting;

/// <summary>The model cannot be unloaded or evicted because requests are still running on it.</summary>
public sealed class ModelBusyException(string name, string detail) : ModelHostException($"The model '{name}' is busy: {detail}", "model_busy");
