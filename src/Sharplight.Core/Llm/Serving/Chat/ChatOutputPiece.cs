namespace Sharplight.Llm.Serving.Chat;

/// <summary>A piece of parsed model output from <see cref="ChatOutputParser"/>.</summary>
/// <param name="Kind">Content, reasoning or a tool call.</param>
/// <param name="Text">The text of a content or reasoning piece; empty for a tool call.</param>
/// <param name="Call">The call of a <see cref="ChatOutputPieceKind.ToolCall"/> piece; null otherwise.</param>
public sealed record ChatOutputPiece(ChatOutputPieceKind Kind, string Text = "", ToolCall? Call = null);
