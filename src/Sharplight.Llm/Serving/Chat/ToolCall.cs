namespace Sharplight.Llm.Serving.Chat;

/// <summary>A tool/function call made by the model (or replayed in the history). Arguments are a JSON object text.</summary>
public sealed record ToolCall(string Id, string Name, string ArgumentsJson);
