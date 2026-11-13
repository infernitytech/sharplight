namespace Sharplight.Llm.Serving.Chat;

internal sealed record ParsedPiece(PieceKind Kind, string Text = "", ToolCall? Call = null);
