namespace Sharplight.Vision;

/// <summary>
/// Document understanding over two task clients: an <see cref="IObjectDetector"/> that analyses the page layout (boxes, classes and, for layout models, the reading order)
/// and an <see cref="ITextRecognizer"/> that reads the text of each region. The page is cut into its regions, every region is read with the task its class asks for, and the
/// result is the structured page in reading order. Nothing here depends on which models back the clients. The reader does not own the clients; requests run one at a time.
/// </summary>
/// <param name="layout">The layout detector.</param>
/// <param name="recognizer">The text recognizer.</param>
public sealed class DocumentReader(IObjectDetector layout, ITextRecognizer recognizer)
{
    /// <summary>Reads a page image.</summary>
    /// <param name="page">The page, any size.</param>
    /// <param name="options">Settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels the request between regions.</param>
    public async Task<DocumentPage> ReadAsync(ImageInput page, DocumentReaderOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(page);
        options ??= new DocumentReaderOptions();
        var found = await layout.DetectAsync(page, new ObjectDetectionOptions { ScoreThreshold = options.ScoreThreshold, MaxDetections = 0, IncludeMasks = false }, cancellationToken).ConfigureAwait(false);
        var objects = found.Objects.All(o => o.ReadingOrder is not null) ? found.Objects.OrderBy(o => o.ReadingOrder).ToList() : [.. found.Objects.OrderBy(o => o.Box.Y1).ThenBy(o => o.Box.X1)];
        var tasks = recognizer.Tasks;
        var regions = new List<DocumentRegion>(objects.Count);
        foreach (var o in objects)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (options.SkippedLabels.Contains(o.Label) || Crop(page, o.Box, options.Padding) is not { } crop)
            {
                regions.Add(new DocumentRegion(regions.Count, o.Label, o.Box, o.Score, null, null));
                continue;
            }

            string? task = options.TaskByLabel.TryGetValue(o.Label, out var wanted) && tasks.Contains(wanted) ? wanted : tasks.Count > 0 ? tasks[0] : null;
            var read = await recognizer.RecognizeAsync(crop, new TextRecognitionOptions { MaxTokens = options.MaxTokensPerRegion, Task = task }, cancellationToken).ConfigureAwait(false);
            regions.Add(new DocumentRegion(regions.Count, o.Label, o.Box, o.Score, read.Text, task));
        }

        return new DocumentPage(regions);
    }

    // The pixels of the box grown by the padding and clipped to the page; null for an empty box.
    private static ImageInput? Crop(ImageInput page, BoundingBox box, int padding)
    {
        int x0 = Math.Max(0, (int)MathF.Floor(box.X1) - padding);
        int y0 = Math.Max(0, (int)MathF.Floor(box.Y1) - padding);
        int x1 = Math.Min(page.Width, (int)MathF.Ceiling(box.X2) + padding);
        int y1 = Math.Min(page.Height, (int)MathF.Ceiling(box.Y2) + padding);
        if (x1 <= x0 || y1 <= y0)
        {
            return null;
        }

        int width = x1 - x0;
        var rgb = new byte[3 * width * (y1 - y0)];
        for (int y = y0; y < y1; y++)
        {
            page.Rgb.AsSpan(3 * ((y * page.Width) + x0), 3 * width).CopyTo(rgb.AsSpan(3 * (y - y0) * width, 3 * width));
        }

        return new ImageInput(width, y1 - y0, rgb);
    }
}
