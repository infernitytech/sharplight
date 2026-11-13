namespace Sharplight.Llm.Serving.Chat;

public sealed record ParsedPiece(PieceKind Kind, string Text = "", ToolCall? Call = null);
