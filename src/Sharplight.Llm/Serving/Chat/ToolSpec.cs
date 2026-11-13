namespace Sharplight.Llm.Serving.Chat;

/// <summary>A function offered to the model; <see cref="ParametersJson"/> is its JSON-schema object.</summary>
public sealed record ToolSpec(string Name, string? Description, string? ParametersJson);
