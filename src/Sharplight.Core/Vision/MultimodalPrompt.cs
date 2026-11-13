namespace Sharplight.Vision;

/// <summary>
/// A prompt after assembly: one row per language-model position-in-cache. Rows are either text tokens or rows of an
/// image's embeddings (<see cref="ImageIndex"/> ≥ 0). Rope positions are absolute and architecture-defined
/// (M-RoPE image spans advance the text position by less than their row count).
/// </summary>
public sealed class MultimodalPrompt
{
    /// <summary>Token id per row (0 for embedding rows).</summary>
    public required int[] Tokens { get; init; }
    /// <summary>Per row: index of the image whose embeddings fill it, or -1 for a text token.</summary>
    public required int[] ImageIndex { get; init; }
    /// <summary>Per row: which embedding row of that image (row-major over the image's output token grid).</summary>
    public required int[] ImageRow { get; init; }
    /// <summary>Rope positions [rows, PositionAxes].</summary>
    public required int[] Positions { get; init; }
    public required int PositionAxes { get; init; }
    /// <summary>Rope position of the next text token after this prompt.</summary>
    public required int NextPosition { get; init; }
    public int Length => Tokens.Length;
}
