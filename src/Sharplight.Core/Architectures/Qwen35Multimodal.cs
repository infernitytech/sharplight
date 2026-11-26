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
            return Sequential(tokens, images, imagePad, vision, context, startPosition);
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

    /// <summary>
    /// One position axis: every row, text or image, takes the next position. With <see cref="VisionParameters.ImageBeginToken"/> and
    /// <see cref="VisionParameters.ImageEndToken"/> (Gemma 3, Gemma 4) the image's rows are framed by those tokens, as the model's processor does.
    /// Tiled images (<see cref="VisionParameters.Idefics3TileLayout"/>) interleave their tiles' rows with the layout's text tokens.
    /// </summary>
    private static MultimodalPrompt Sequential(ReadOnlySpan<int> tokens, IReadOnlyList<PreprocessedImage> images, int imagePad, VisionParameters? vision,
        MultimodalContext context, int startPosition)
    {
        int begin = vision is { ImageBeginToken.Length: > 0 } ? context.RequireToken(vision.ImageBeginToken) : -1;
        int end = vision is { ImageEndToken.Length: > 0 } ? context.RequireToken(vision.ImageEndToken) : -1;
        bool tiled = vision?.TileLayout == VisionParameters.Idefics3TileLayout;
        var toks = new List<int>(tokens.Length);
        var imageIndex = new List<int>(tokens.Length);
        var imageRow = new List<int>(tokens.Length);

        void Text(int token)
        {
            toks.Add(token);
            imageIndex.Add(-1);
            imageRow.Add(-1);
        }

        void Rows(int img, int first, int count)
        {
            for (int i = first; i < first + count; i++)
            {
                toks.Add(0);
                imageIndex.Add(img);
                imageRow.Add(i);
            }
        }

        int img = 0;
        foreach (var t in tokens)
        {
            if (t != imagePad)
            {
                Text(t);
                continue;
            }

            var im = images[img];
            if (tiled)
            {
                Idefics3Layout(vision!, context, im, img, Text, Rows);
                img++;
                continue;
            }

            if (begin >= 0)
            {
                Text(begin);
            }

            Rows(img, 0, im.TokenCount);
            if (end >= 0)
            {
                Text(end);
            }

            img++;
        }

        int rows = toks.Count;
        var pos = new int[rows];
        for (int r = 0; r < rows; r++)
        {
            pos[r] = startPosition + r;
        }

        return new MultimodalPrompt
        {
            Tokens = [.. toks], ImageIndex = [.. imageIndex], ImageRow = [.. imageRow], Positions = pos, PositionAxes = 1, NextPosition = startPosition + rows,
        };
    }

    /// <summary>
    /// The Idefics3 / SmolVLM layout of a tiled image (the processor's prompt): per tile <c>&lt;fake_token_around_image&gt;&lt;row_r_col_c&gt;</c>
    /// and its rows, a newline after every tile row but the last, then <c>\n\n&lt;fake_token_around_image&gt;&lt;global-img&gt;</c>, the overview's
    /// rows and <c>&lt;fake_token_around_image&gt;</c>; an image without tiles is only the framed overview.
    /// </summary>
    private static void Idefics3Layout(VisionParameters vision, MultimodalContext context, PreprocessedImage image, int img, Action<int> text,
        Action<int, int, int> rows)
    {
        int fake = context.RequireToken(vision.ImageBeginToken);
        int global = context.RequireToken(vision.TileGlobalToken);
        int perView = image.TokenCount / (image.TileColumns * image.TileRows + 1);
        int row = 0;
        if (image.TileColumns > 0)
        {
            int newline = context.RequireToken("\n");
            for (int r = 0; r < image.TileRows; r++)
            {
                for (int c = 0; c < image.TileColumns; c++)
                {
                    text(fake);
                    text(context.RequireToken($"<row_{r + 1}_col_{c + 1}>"));
                    rows(img, row, perView);
                    row += perView;
                }

                if (r < image.TileRows - 1)
                {
                    text(newline);
                }
            }

            text(context.RequireToken("\n\n"));
        }

        text(fake);
        text(global);
        rows(img, row, perView);
        text(fake);
    }
}
