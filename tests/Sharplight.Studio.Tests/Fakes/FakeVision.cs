using Sharplight.Tasks;
using Sharplight.Vision;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>One fake client for the vision interfaces (a model gets the one of its kind); records the options of every request.</summary>
internal sealed class FakeVision : IImageClassifier, IImageEmbedder, IObjectDetector, IDepthEstimator, IImageSegmenter, ITextRecognizer, IImageUpscaler
{
    public TaskClientMetadata Metadata { get; } = new(Sharplight.Models.ModelKinds.ImageClassification, "fake-vision", "vit");

    public IReadOnlyList<string> Labels { get; } = ["tabby cat", "tiger cat", "Egyptian cat"];

    public int Dimensions => 8;

    public bool SupportsText { get; set; } = true;

    /// <summary>Blocking work per step (a stand-in for a graph run that holds its thread), or zero.</summary>
    public TimeSpan StepWork { get; set; }

    public List<object?> Options { get; } = [];

    public List<IReadOnlyList<string>> Candidates { get; } = [];

    /// <summary>The objects every detection returns, best first.</summary>
    public List<DetectedObject> Objects { get; } =
    [
        new(0, "cat", 0.93f, new BoundingBox(4, 6, 40, 50)),
        new(1, "dog", 0.62f, new BoundingBox(30, 20, 60, 60)),
        new(2, "cup", 0.21f, new BoundingBox(1, 1, 12, 12)),
    ];

    public Task<ImageClassification> ClassifyAsync(ImageInput image, ImageClassificationOptions? options = null, CancellationToken cancellationToken = default)
    {
        Options.Add(options);
        SlowWork.Block(StepWork, cancellationToken);
        return Task.FromResult(new ImageClassification([new(0, "tabby cat", 0.71f, 2f), new(1, "tiger cat", 0.2f, 1f), new(2, "Egyptian cat", 0.09f, 0f)]));
    }

    public Task<float[]> EmbedImageAsync(ImageInput image, ImageEmbeddingOptions? options = null, CancellationToken cancellationToken = default)
    {
        Options.Add(options);
        return Task.FromResult(Enumerable.Range(0, Dimensions).Select(i => i / 10f).ToArray());
    }

    public Task<float[]> EmbedTextAsync(string text, ImageEmbeddingOptions? options = null, CancellationToken cancellationToken = default)
        => Task.FromResult(new float[Dimensions]);

    public Task<ImageClassification> ClassifyZeroShotAsync(ImageInput image, IReadOnlyList<string> candidates, ImageClassificationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        Options.Add(options);
        Candidates.Add(candidates);
        // The last candidate wins, the first is second.
        var scores = candidates.Select((label, i) => new ImageLabelScore(i, label, i == candidates.Count - 1 ? 0.8f : 0.2f / Math.Max(1, candidates.Count - 1), 0f))
            .OrderByDescending(score => score.Score).ToList();
        return Task.FromResult(new ImageClassification(scores));
    }

    public Task<ObjectDetection> DetectAsync(ImageInput image, ObjectDetectionOptions? options = null, CancellationToken cancellationToken = default)
    {
        Options.Add(options);
        float threshold = options?.ScoreThreshold ?? 0.5f;
        return Task.FromResult(new ObjectDetection([.. Objects.Where(item => item.Score >= threshold)]));
    }

    public Task<DepthMap> EstimateAsync(ImageInput image, DepthEstimationOptions? options = null, CancellationToken cancellationToken = default)
    {
        Options.Add(options);
        var values = new float[image.Width * image.Height];
        for (int y = 0; y < image.Height; y++)
        {
            for (int x = 0; x < image.Width; x++)
            {
                values[y * image.Width + x] = 1f + y / (float)image.Height * 4f;
            }
        }

        return Task.FromResult(new DepthMap(image.Width, image.Height, values));
    }

    /// <summary>The prompts every segmentation received, in order.</summary>
    public List<SegmentationPrompt> Prompts { get; } = [];

    /// <summary>How often the image stage ran.</summary>
    public int Encodings { get; private set; }

    /// <summary>The text every recognition returns.</summary>
    public string Text { get; set; } = "Hello Sharplight";

    public int Scale => 2;

    public Task<SegmentationImage> EncodeImageAsync(ImageInput image, CancellationToken cancellationToken = default)
    {
        Encodings++;
        return Task.FromResult(NewSegmentationImage(image));
    }

    public Task<ImageSegmentation> SegmentAsync(SegmentationImage image, SegmentationPrompt prompt, SegmentationOptions? options = null, CancellationToken cancellationToken = default)
    {
        Options.Add(options);
        Prompts.Add(prompt);
        // Three candidates: the left half, the left quarter and the whole image; the first has the best score.
        SegmentMask Mask(int width, float score)
        {
            var pixels = new bool[image.Width * image.Height];
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = i % image.Width < width;
            }

            return new SegmentMask(image.Width, image.Height, pixels, score);
        }

        return Task.FromResult(new ImageSegmentation([Mask(image.Width / 2, 0.95f), Mask(image.Width / 4, 0.8f), Mask(image.Width, 0.6f)], 4f));
    }

    public Task<TextRecognition> RecognizeAsync(ImageInput image, TextRecognitionOptions? options = null, CancellationToken cancellationToken = default)
    {
        Options.Add(options);
        return Task.FromResult(new TextRecognition(Text, [new TextRegion(Text, null)]));
    }

    public Task<ImageInput> UpscaleAsync(ImageInput image, UpscaleOptions? options = null, CancellationToken cancellationToken = default)
    {
        Options.Add(options);
        options?.Progress?.Report(1);
        var rgb = new byte[3 * image.Width * Scale * image.Height * Scale];
        Array.Fill(rgb, (byte)200);
        return Task.FromResult(new ImageInput(image.Width * Scale, image.Height * Scale, rgb));
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose()
    {
    }

    // SegmentationImage has no public constructor (only a segmenter makes them): the fake builds one through its internal one.
    private SegmentationImage NewSegmentationImage(ImageInput image)
        => (SegmentationImage)Activator.CreateInstance(typeof(SegmentationImage), System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic, null,
            [this, image.Width, image.Height, Array.Empty<float>(), Array.Empty<float>(), Array.Empty<float>()], null)!;
}
