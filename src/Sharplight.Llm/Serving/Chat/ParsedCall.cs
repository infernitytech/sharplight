namespace Sharplight.Llm.Serving.Chat;

/// <summary>A call found by an <see cref="IToolCallFormat"/>.</summary>
/// <param name="Name">Function name.</param>
/// <param name="ArgumentsJson">Arguments as a JSON object text.</param>
public readonly record struct ParsedCall(string Name, string ArgumentsJson);
