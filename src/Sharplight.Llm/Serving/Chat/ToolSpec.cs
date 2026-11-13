namespace Sharplight.Llm.Serving.Chat;

/// <summary>A function offered to the model; <see cref="ParametersJson"/> is its JSON-schema object.</summary>
/// <param name="Name">Function name.</param>
/// <param name="Description">What the function does, shown to the model.</param>
/// <param name="ParametersJson">JSON-schema object describing the parameters, or null for none.</param>
/// <param name="Strict">
/// Enforce the call syntax and <paramref name="ParametersJson"/> with constrained decoding even when the model may choose
/// not to call (OpenAI <c>strict</c>); forced calls (<see cref="ToolChoiceMode.Required"/>, <see cref="ToolChoiceMode.Function"/>)
/// are always enforced.
/// </param>
public sealed record ToolSpec(string Name, string? Description, string? ParametersJson, bool Strict = false);
