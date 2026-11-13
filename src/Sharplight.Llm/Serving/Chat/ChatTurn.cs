namespace Sharplight.Llm.Serving.Chat;

/// <summary>One message. Roles: system, user, assistant, tool. Give <see cref="Content"/> or <see cref="Parts"/>.</summary>
public sealed record ChatTurn(
    string Role,
    string? Content = null,
    IReadOnlyList<ChatPart>? Parts = null,
    string? Reasoning = null,
    IReadOnlyList<ToolCall>? ToolCalls = null,
    string? ToolCallId = null,
    string? Name = null);
