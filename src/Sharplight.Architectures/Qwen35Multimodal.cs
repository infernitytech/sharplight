using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Image prompt assembly of the Qwen3.5 family (canonical arch <c>qwen3_5</c>). The chat template wraps an image as
/// <c>&lt;|vision_start|&gt;&lt;|image_pad|&gt;&lt;|vision_end|&gt;</c>; each <c>&lt;|image_pad|&gt;</c> placeholder expands to one
/// row per merged image patch whose embeddings come from the vision encoder. M-RoPE positions (axes: time, row, column,
/// unused): an image whose output grid is ny × nx starting at position p gives its row r, column c the positions
/// (p, p + r, p + c, 0); the text after the image continues at p + max(nx, ny). Text tokens use the same position on the
/// first three axes.
/// </summary>
public sealed class Qwen35Multimodal : IMultimodalArchitecture
{
    private const string ImageToken = "<|image_pad|>";
    public IReadOnlyList<string> Architectures => ["qwen3_5"];

    private const int Axes = 4;

    public MultimodalPrompt Assemble(MultimodalContext context, ReadOnlySpan<int> tokens, IReadOnlyList<PreprocessedImage> images, int startPosition)
    {
        int imagePad = context.RequireToken(ImageToken);
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
            throw new ArgumentException($"The prompt has {placeholders} {ImageToken} placeholder(s) but {images.Count} image(s) were given.");
        }

        int rows = tokens.Length - placeholders;
        foreach (var im in images)
        {
            rows += im.TokenCount;
        }

        var toks = new int[rows];
        var imageIndex = new int[rows];
        var imageRow = new int[rows];
        var pos = new int[rows * Axes];

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
                pos[r * Axes] = pos[r * Axes + 1] = pos[r * Axes + 2] = next++;
                r++;
                continue;
            }
            var im = images[img];
            for (int i = 0; i < im.TokenCount; i++, r++)
            {
                imageIndex[r] = img;
                imageRow[r] = i;
                pos[r * Axes] = next;
                pos[r * Axes + 1] = next + i / im.MergedWidth;
                pos[r * Axes + 2] = next + i % im.MergedWidth;
            }
            next += Math.Max(im.MergedWidth, im.MergedHeight);
            img++;
        }
        return new MultimodalPrompt { Tokens = toks, ImageIndex = imageIndex, ImageRow = imageRow, Positions = pos, PositionAxes = Axes, NextPosition = next };
    }
}
