namespace Sharplight.Llm.Serving.Chat;

/// <summary>A tool/function call made by the model (or replayed in the history). Arguments are a JSON object text.</summary>
/// <param name="Id">Call id that the matching tool-result message refers to.</param>
/// <param name="Name">Name of the called function.</param>
/// <param name="ArgumentsJson">The arguments as a JSON object text.</param>
public sealed record ToolCall(string Id, string Name, string ArgumentsJson);
