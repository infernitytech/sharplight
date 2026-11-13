namespace Sharplight.Llm.Serving.Hosting;

public sealed class ModelBusyException(string name, string detail) : ModelHostException($"The model '{name}' is busy: {detail}", "model_busy");
