using System.Globalization;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Image prompt assembly of the Qwen3.5 family (canonical arch <c>qwen3_5</c>). The chat template wraps an image as
/// <c>&lt;|vision_start|&gt;&lt;|image_pad|&gt;&lt;|vision_end|&gt;</c>; each <c>&lt;|image_pad|&gt;</c> placeholder expands to one
/// row per merged image patch whose embeddings come from the vision encoder. M-RoPE positions (axes: time, row, column,
/// unused): an image whose output grid is ny × nx starting at position p gives its row r, column c the positions
/// (p, p + r, p + c, 0); the text after the image continues at p + max(nx, ny). Text tokens use the same position on the
/// first three axes. The same layout serves the generic <c>decoder</c> with a vision tower (Qwen2-VL, Qwen2.5-VL, Qwen3-VL and PaddleOCR-VL,
/// whose M-RoPE image positions are those of Qwen3.5; the placeholder is the tower's <see cref="VisionParameters.ImageToken"/>, PaddleOCR-VL's
/// <c>&lt;|IMAGE_PLACEHOLDER|&gt;</c>); DeepStack blocks travel in the image rows' embeddings. Towers with sequential image positions
/// (<see cref="VisionParameters.SequentialImagePositions"/>, HunyuanVL's XD-RoPE) lay an image out as its encoder rows (begin, grid rows each followed
/// by a separator, end): every row takes the next position on the first axis, a grid row (column, row, image ordinal) on the other three, the
/// begin and end rows their position on every axis; text tokens carry their position on all four axes. Towers with grouped image positions
/// (<see cref="VisionParameters.GroupedImagePositions"/>, CogVLM) give the begin row, all grid rows together and the end row one position each;
/// a language model without M-RoPE has a single position axis. With <see cref="VisionParameters.AllButLastMediaWeightRows"/> the prompt marks
/// which image rows use the language model's media weight set (<see cref="MultimodalPrompt.MediaWeights"/>).
/// <c>&lt;|IMAGE_PLACEHOLDER|&gt;</c>); DeepStack blocks travel in the image rows' embeddings. A model with one position axis (DeepSeek-OCR, whose
/// placeholder is <c>&lt;image&gt;</c>) gets consecutive positions for all rows instead, unless its tower has grouped image positions.
/// </summary>
public sealed class Qwen35Multimodal : IMultimodalArchitecture
{
    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => ["qwen3_5", "decoder"];

    // Position axes of an M-RoPE language model.
    private const int MropeAxes = 4;

    /// <inheritdoc/>
    public MultimodalPrompt Assemble(MultimodalContext context, ReadOnlySpan<int> tokens, IReadOnlyList<PreprocessedImage> images, int startPosition)
    {
        var vision = VisionParameters.Of(context.Model);
        string imageToken = vision?.ImageToken ?? VisionParameters.DefaultImageToken;
        bool sequential = vision?.ImagePositions == VisionParameters.SequentialImagePositions;
        bool grouped = vision?.ImagePositions == VisionParameters.GroupedImagePositions;
        int axes = context.Model.Property(Models.WellKnown.PositionAxes, MropeAxes);
        int textAxes = Math.Min(axes, sequential || grouped ? axes : 3);
        bool[]? mediaWeights = null;
        int imagePad = context.RequireToken(imageToken);
        int placeholders = 0;
        foreach (var t in tokens)
        {
            if (t == imagePad)
            {
                placeholders++;
            }
        }

        if (placeholders != images.Count)
        {
            throw new ArgumentException($"The prompt has {placeholders} {imageToken} placeholder(s) but {images.Count} image(s) were given.");
        }

        int rows = tokens.Length - placeholders;
        foreach (var im in images)
        {
            rows += im.TokenCount;
        }

        // Grouped image positions (CogVLM) have their own layout on the single axis.
        if (!grouped && context.Model.Properties.TryGetValue(WellKnown.PositionAxes, out var axesValue) && Convert.ToInt32(axesValue, CultureInfo.InvariantCulture) == 1)
        {
            return Sequential(tokens, images, imagePad, rows, startPosition);
        }

        var toks = new int[rows];
        var imageIndex = new int[rows];
        var imageRow = new int[rows];
        var pos = new int[rows * axes];
        if (vision?.MediaWeightRows == VisionParameters.AllButLastMediaWeightRows)
        {
            mediaWeights = new bool[rows];
        }

        int r = 0;
        int next = startPosition;
        int img = 0;
        foreach (var t in tokens)
        {
            if (t != imagePad)
            {
                toks[r] = t;
                imageIndex[r] = -1;
                imageRow[r] = -1;
                for (int a = 0; a < textAxes; a++)
                {
                    pos[r * axes + a] = next;
                }

                next++;
                r++;
                continue;
            }
            var im = images[img];
            if (mediaWeights is not null)
            {
                mediaWeights.AsSpan(r, im.TokenCount - 1).Fill(true);
            }

            if (grouped)
            {
                for (int i = 0; i < im.TokenCount; i++, r++)
                {
                    imageIndex[r] = img;
                    imageRow[r] = i;
                    pos.AsSpan(r * axes, axes).Fill(next + GroupedOffset(vision!, im, i));
                }

                next += GroupedOffset(vision!, im, im.TokenCount - 1) + 1;
                img++;
                continue;
            }

            if (sequential)
            {
                for (int i = 0; i < im.TokenCount; i++, r++)
                {
                    imageIndex[r] = img;
                    imageRow[r] = i;
                    SequentialPosition(vision!, im, i, img, next + i, pos.AsSpan(r * axes, axes));
                }

                next += im.TokenCount;
                img++;
                continue;
            }

            for (int i = 0; i < im.TokenCount; i++, r++)
            {
                imageIndex[r] = img;
                imageRow[r] = i;
                pos[r * axes] = next;
                pos[r * axes + 1] = next + i / im.MergedWidth;
                pos[r * axes + 2] = next + i % im.MergedWidth;
            }
            next += Math.Max(im.MergedWidth, im.MergedHeight);
            img++;
        }
        return new MultimodalPrompt
        {
            Tokens = toks, ImageIndex = imageIndex, ImageRow = imageRow, Positions = pos, PositionAxes = axes, NextPosition = next, MediaWeights = mediaWeights,
        };
    }

    /// <summary>
    /// Offset from the image's first position of row <paramref name="row"/> with grouped image positions: the begin row 0, every grid row 1 (0
    /// without boundary rows), the end row 2.
    /// </summary>
    private static int GroupedOffset(VisionParameters vision, PreprocessedImage image, int row)
    {
        if (!vision.ImageBoundary)
        {
            return 0;
        }

        return row == 0 ? 0 : row == image.TokenCount - 1 ? 2 : 1;
    }

    /// <summary>
    /// The positions of row <paramref name="row"/> of an image with sequential image positions: <paramref name="sequential"/> on the first axis;
    /// a grid row (or its separator, at column = merged width) has (column, row, image ordinal) on the other axes, a begin / end row its
    /// sequential position on every axis.
    /// </summary>
    private static void SequentialPosition(VisionParameters vision, PreprocessedImage image, int row, int ordinal, int sequential, Span<int> axes)
    {
        int boundary = vision.ImageBoundary ? 1 : 0;
        if (vision.ImageBoundary && (row == 0 || row == image.TokenCount - 1))
        {
            axes.Fill(sequential);
            return;
        }

        int width = image.MergedWidth + (vision.RowSeparator ? 1 : 0);
        int k = row - boundary;
        axes[0] = sequential;
        axes[1] = k % width;
        axes[2] = k / width;
        axes[3] = ordinal;
    }

    /// <summary>One position axis: every row, text or image, takes the next position.</summary>
    private static MultimodalPrompt Sequential(ReadOnlySpan<int> tokens, IReadOnlyList<PreprocessedImage> images, int imagePad, int rows, int startPosition)
    {
        var toks = new int[rows];
        var imageIndex = new int[rows];
        var imageRow = new int[rows];
        var pos = new int[rows];
        int r = 0;
        int img = 0;
        foreach (var t in tokens)
        {
            if (t != imagePad)
            {
                toks[r] = t;
                imageIndex[r] = -1;
                imageRow[r] = -1;
                pos[r] = startPosition + r;
                r++;
                continue;
            }

            for (int i = 0; i < images[img].TokenCount; i++, r++)
            {
                imageIndex[r] = img;
                imageRow[r] = i;
                pos[r] = startPosition + r;
            }

            img++;
        }

        return new MultimodalPrompt { Tokens = toks, ImageIndex = imageIndex, ImageRow = imageRow, Positions = pos, PositionAxes = 1, NextPosition = startPosition + rows };
    }
}
