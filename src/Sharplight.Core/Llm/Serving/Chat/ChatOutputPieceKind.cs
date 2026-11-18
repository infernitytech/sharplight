namespace Sharplight.Llm.Serving.Chat;

/// <summary>What a <see cref="ChatOutputPiece"/> carries.</summary>
public enum ChatOutputPieceKind
{
    /// <summary>Answer text.</summary>
    Content,

    /// <summary>Reasoning (thinking) text.</summary>
    Reasoning,

    /// <summary>A complete tool call.</summary>
    ToolCall,
}
