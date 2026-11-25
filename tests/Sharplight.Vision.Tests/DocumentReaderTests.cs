using Sharplight.Imaging;
using Sharplight.Tasks;

namespace Sharplight.Vision.Tests;

/// <summary>
/// <see cref="DocumentReader"/>: the generic flow over fake clients (reading order, task per label, skipped figures, crops), and the end-to-end run of PP-DocLayoutV3 with
/// GLM-OCR on the synthetic page of <c>tools/layout-reference.py</c> (skipped when either checkpoint is missing).
/// </summary>
public class DocumentReaderTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    [Fact]
    public async Task Regions_are_read_in_reading_order_with_the_task_of_their_label()
    {
        var page = new ImageInput(100, 100, new byte[3 * 100 * 100]);
        using var layout = new FakeLayout(
        [
            new DetectedObject(1, "table", 0.9f, new BoundingBox(10, 50, 60, 90), null, 1),
            new DetectedObject(0, "text", 0.9f, new BoundingBox(10, 10, 60, 30), null, 0),
            new DetectedObject(2, "image", 0.9f, new BoundingBox(60, 10, 90, 40), null, 2),
        ]);
        using var recognizer = new FakeRecognizer();
        var result = await new DocumentReader(layout, recognizer).ReadAsync(page, null, TestContext.Current.CancellationToken);
        Assert.Equal(["text", "table", "image"], result.Regions.Select(r => r.Label));
        Assert.Equal(["text", "table", null], result.Regions.Select(r => r.Task));
        Assert.Equal(["text:58x28", "table:58x48", null], result.Regions.Select(r => r.Text));
        Assert.Equal("text:58x28\n\ntable:58x48", result.Text);
        Assert.Equal([0, 1, 2], result.Regions.Select(r => r.Order));
    }

    [Fact]
    public async Task Without_a_reading_order_regions_follow_the_page_top_down()
    {
        var page = new ImageInput(100, 100, new byte[3 * 100 * 100]);
        using var layout = new FakeLayout([new DetectedObject(0, "text", 0.9f, new BoundingBox(10, 60, 50, 80)), new DetectedObject(0, "text", 0.9f, new BoundingBox(10, 5, 50, 20))]);
        using var recognizer = new FakeRecognizer();
        var result = await new DocumentReader(layout, recognizer).ReadAsync(page, null, TestContext.Current.CancellationToken);
        Assert.Equal([5f, 60f], result.Regions.Select(r => r.Box.Y1));
    }

    [Fact]
    public async Task Layout_and_glm_ocr_read_a_page()
    {
        string layoutDir = Path.Combine(Home, "models/vision-r10/pp-doclayoutv3");
        string ocrDir = Path.Combine(Home, "models/vision-r10/glm-ocr");
        Assert.SkipUnless(File.Exists(Path.Combine(layoutDir, "sharplight-sample.png")) && File.Exists(Path.Combine(ocrDir, "model.safetensors")), "PP-DocLayoutV3 or GLM-OCR is not available");
        var slot = Env.Vulkan() ?? Env.Cpu();
        var page = PngCodec.Read(Path.Combine(layoutDir, "sharplight-sample.png"));
        using var layoutModel = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = layoutDir, Devices = [slot] });
        using var ocrModel = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = ocrDir, Devices = [slot], ContextLength = 2048 });
        var provider = new TaskClientProvider([.. Env.TaskFactories()]);
        using var layout = provider.Create<IObjectDetector>(layoutModel);
        using var recognizer = provider.Create<ITextRecognizer>(ocrModel);
        var result = await new DocumentReader(layout, recognizer).ReadAsync(page, new DocumentReaderOptions { MaxTokensPerRegion = 160 }, TestContext.Current.CancellationToken);
        TestContext.Current.SendDiagnosticMessage(string.Join("\n", result.Regions.Select(r => $"{r.Order} {r.Label} {r.Task}: {r.Text?.Replace("\n", " ⏎ ")}")));
        Assert.Equal("paragraph_title", result.Regions[0].Label);
        Assert.Contains("Sample Document", result.Regions[0].Text);
        Assert.Contains(result.Regions, r => r.Label == "table" && r.Task == "table" && r.Text!.Contains("<t", StringComparison.Ordinal));
        Assert.Contains(result.Regions, r => r.Label == "image" && r.Text is null);
    }

    private sealed class FakeLayout(DetectedObject[] objects) : IObjectDetector
    {
        public TaskClientMetadata Metadata => new("object-detection", "fake", "fake");

        public IReadOnlyList<string> Labels => [];

        public object? GetService(Type serviceType, object? serviceKey = null) => null;

        public Task<ObjectDetection> DetectAsync(ImageInput image, ObjectDetectionOptions? options = null, CancellationToken cancellationToken = default)
            => Task.FromResult(new ObjectDetection(objects));

        public void Dispose()
        {
        }
    }

    private sealed class FakeRecognizer : ITextRecognizer
    {
        public TaskClientMetadata Metadata => new("ocr", "fake", "fake");

        public IReadOnlyList<string> Tasks => ["text", "table"];

        public object? GetService(Type serviceType, object? serviceKey = null) => null;

        public Task<TextRecognition> RecognizeAsync(ImageInput image, TextRecognitionOptions? options = null, CancellationToken cancellationToken = default)
            => Task.FromResult(new TextRecognition($"{options!.Task}:{image.Width}x{image.Height}", []));

        public void Dispose()
        {
        }
    }
}
