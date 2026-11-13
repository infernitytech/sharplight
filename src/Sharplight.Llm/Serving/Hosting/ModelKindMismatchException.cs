namespace Sharplight.Llm.Serving.Hosting;

public sealed class ModelKindMismatchException(string name, ModelKind actual, ModelKind wanted)
    : ModelHostException($"The model '{name}' is a {actual} model and cannot be used for {wanted} requests.", "model_not_found");
