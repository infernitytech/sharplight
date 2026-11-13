namespace Sharplight.Templating;

/// <summary>A tool call made by the assistant, as exposed to the template (<c>message.tool_calls</c>).</summary>
/// <param name="Id">Call identifier, or null when the format has none.</param>
/// <param name="Name">Name of the called function.</param>
/// <param name="Arguments">Argument values in the template value model, or null for no arguments.</param>
public sealed record ChatToolCall(string? Id, string Name, IReadOnlyDictionary<string, object?>? Arguments = null);
