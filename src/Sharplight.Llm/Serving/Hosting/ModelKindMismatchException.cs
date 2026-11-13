namespace Sharplight.Llm.Serving.Hosting;

/// <summary>The named model exists but is of another <see cref="ModelKind"/> than the request needs.</summary>
public sealed class ModelKindMismatchException(string name, ModelKind actual, ModelKind wanted)
    : ModelHostException($"The model '{name}' is a {actual} model and cannot be used for {wanted} requests.", "model_not_found");
