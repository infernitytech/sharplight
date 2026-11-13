namespace Sharplight.Vision;

/// <summary>An image converted to the inputs of the vision-encode graph (see <see cref="VisionParameters"/>).</summary>
public sealed class PreprocessedImage
{
    /// <summary>Size of the resized image in pixels.</summary>
    public required int Width { get; init; }
    /// <summary>Height of the resized image in pixels.</summary>
    public required int Height { get; init; }
    /// <summary>Patch grid (rows × columns of patches).</summary>
    public required int GridHeight { get; init; }
    /// <summary>Number of patch columns.</summary>
    public required int GridWidth { get; init; }
    /// <summary>Output token grid after spatial merging.</summary>
    public required int MergedHeight { get; init; }
    /// <summary>Number of output token columns after spatial merging.</summary>
    public required int MergedWidth { get; init; }
    /// <summary>Normalized patch values [N, PatchDim], merge-window order, each patch flattened in (channel, frame, row, col) order.</summary>
    public required float[] Patches { get; init; }
    /// <summary>(row, col) of each patch in the patch grid [N, 2].</summary>
    public required int[] Positions { get; init; }
    /// <summary>Bilinear taps into the learned position grid [N, 4] (row-major grid index) and their weights [N, 4].</summary>
    public required int[] EmbeddingIndex { get; init; }
    /// <summary>The weights of the taps in <see cref="EmbeddingIndex"/> [N, 4].</summary>
    public required float[] EmbeddingWeight { get; init; }

    /// <summary>Number of patches (N).</summary>
    public int PatchCount => GridHeight * GridWidth;
    /// <summary>Number of embeddings the encoder produces (and language-model positions the image occupies as rows).</summary>
    public int TokenCount => MergedHeight * MergedWidth;
}
