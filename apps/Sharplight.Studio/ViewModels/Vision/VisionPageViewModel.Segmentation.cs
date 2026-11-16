using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Vision;

namespace Sharplight.Studio.ViewModels.Vision;

/// <summary>The segmentation part of the Vision page: the prompt (points and a box on the image) and the encoded image that makes each new prompt quick.</summary>
internal sealed partial class VisionPageViewModel
{
    private readonly List<SegmentationPoint> _points = [];
    private BoundingBox? _box;
    private SegmentationImage? _segmentImage;
    private ImageInput? _segmentSource;
    private string _segmentModel = "";

    /// <summary>The prompt's points, in pixels of the image.</summary>
    internal IReadOnlyList<SegmentationPoint> PromptPoints => _points;

    /// <summary>The prompt's box, or null.</summary>
    internal BoundingBox? PromptBox => _box;

    /// <summary>The prompt tool the pointer uses (<see cref="SegmentTool"/>).</summary>
    public string PromptTool => Panel.GetChoice("segtool");

    /// <summary>
    /// A click on the image segments again: the point (as fractions of the image's width and height) joins the prompt, as a point on the object or, with
    /// <paramref name="negative"/>, beside it. Without effect when the page has no segmentation to refine.
    /// </summary>
    /// <param name="x">Horizontal position, 0 to 1.</param>
    /// <param name="y">Vertical position, 0 to 1.</param>
    /// <param name="negative">Whether the point is not on the object.</param>
    public void PlacePoint(double x, double y, bool negative)
    {
        if (!CanPrompt || Source.Image is not { } image)
        {
            return;
        }

        _box = null;
        _points.Add(new SegmentationPoint((float)(Math.Clamp(x, 0, 1) * image.Width), (float)(Math.Clamp(y, 0, 1) * image.Height), !negative));
        RunSegmentation();
    }

    /// <summary>A box dragged on the image (fractions of its width and height) becomes the prompt.</summary>
    /// <param name="x1">Left edge, 0 to 1.</param>
    /// <param name="y1">Top edge, 0 to 1.</param>
    /// <param name="x2">Right edge, 0 to 1.</param>
    /// <param name="y2">Bottom edge, 0 to 1.</param>
    public void PlaceBox(double x1, double y1, double x2, double y2)
    {
        if (!CanPrompt || Source.Image is not { } image)
        {
            return;
        }

        double left = Math.Clamp(Math.Min(x1, x2), 0, 1) * image.Width;
        double right = Math.Clamp(Math.Max(x1, x2), 0, 1) * image.Width;
        double top = Math.Clamp(Math.Min(y1, y2), 0, 1) * image.Height;
        double bottom = Math.Clamp(Math.Max(y1, y2), 0, 1) * image.Height;
        if (right - left < 2 || bottom - top < 2)
        {
            return;
        }

        _points.Clear();
        _box = new BoundingBox((float)left, (float)top, (float)right, (float)bottom);
        RunSegmentation();
    }

    /// <summary>Removes the points and the box, and segments from the middle of the image again.</summary>
    [RelayCommand]
    private void ClearPrompt()
    {
        if (CanPrompt)
        {
            _points.Clear();
            _box = null;
            RunSegmentation();
        }
    }

    private void RunSegmentation()
    {
        if (RunCommand.CanExecute(null))
        {
            RunCommand.Execute(null);
        }
    }

    /// <summary>The prompt of a run: the points and box the user placed, or one point in the middle of the image.</summary>
    private SegmentationPrompt BuildPrompt(ImageInput image)
    {
        if (_points.Count == 0 && _box is null)
        {
            _points.Add(new SegmentationPoint(image.Width / 2f, image.Height / 2f));
        }

        return new SegmentationPrompt { Points = [.. _points], Box = _box };
    }

    /// <summary>The mask parameter: 0 for the best candidate, else the number of the chosen one.</summary>
    private int MaskChoice() => int.TryParse(Panel.GetChoice("mask"), out int number) ? number : 0;

    /// <summary>The index of the mask to show: the chosen number (from 1) when it exists, else the candidate with the best predicted IoU.</summary>
    private static int ChosenMask(IReadOnlyList<SegmentMask> masks, int choice)
    {
        if (choice > 0 && choice <= masks.Count)
        {
            return choice - 1;
        }

        int best = 0;
        for (int i = 1; i < masks.Count; i++)
        {
            best = masks[i].Score > masks[best].Score ? i : best;
        }

        return best;
    }

    /// <summary>Encodes the image (once per image and model) and segments the prompt.</summary>
    private async Task<ImageSegmentation> SegmentAsync(IImageSegmenter segmenter, string modelId, ImageInput image, SegmentationPrompt prompt, SegmentationOptions options,
        RunContext context, CancellationToken token)
    {
        bool cached = _segmentImage is not null && ReferenceEquals(_segmentSource, image) && _segmentModel == modelId;
        if (cached)
        {
            try
            {
                context.Report(null, "Segmenting");
                return await segmenter.SegmentAsync(_segmentImage!, prompt, options, token).ConfigureAwait(false);
            }
            catch (ArgumentException)
            {
                // The model was loaded again since: the encoded image belongs to the old client.
            }
        }

        context.Report(null, "Encoding the image");
        var encoded = await segmenter.EncodeImageAsync(image, token).ConfigureAwait(false);
        _segmentImage = encoded;
        _segmentSource = image;
        _segmentModel = modelId;
        context.Report(null, "Segmenting");
        return await segmenter.SegmentAsync(encoded, prompt, options, token).ConfigureAwait(false);
    }
}
