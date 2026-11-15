namespace Sharplight.Llm.Serving.Chat;

/// <summary>One message. Roles: system, user, assistant, tool. Give <see cref="Content"/> or <see cref="Parts"/>.</summary>
/// <param name="Role"><c>system</c>, <c>user</c>, <c>assistant</c> or <c>tool</c>.</param>
/// <param name="Content">Plain text content.</param>
/// <param name="Parts">Multi-part content (text and images) instead of <paramref name="Content"/>.</param>
/// <param name="Reasoning">Reasoning text of an earlier assistant turn, replayed to templates that use it.</param>
/// <param name="ToolCalls">Tool calls of an assistant turn.</param>
/// <param name="ToolCallId">For <c>tool</c> messages: the id of the call this answers.</param>
/// <param name="Name">Optional author / function name.</param>
public sealed record ChatTurn(
    string Role,
    string? Content = null,
    IReadOnlyList<ChatPart>? Parts = null,
    string? Reasoning = null,
    IReadOnlyList<ToolCall>? ToolCalls = null,
    string? ToolCallId = null,
    string? Name = null);
