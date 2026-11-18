using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Vision.Tests;

/// <summary>
/// Object detectors through the task layer, against transformers (<c>tools/vision-reference.py --size 640</c> writes the reference next to the
/// checkpoint). Skipped when the checkpoint or its reference is not present.
/// </summary>
public class ObjectDetectorTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    private static readonly string[] Roots = ["models/vision", "models/vision-sweep", "models/vision-r7", "models/hot-b", "models/vision-r10"];

    private static string Model(string name) => Path.Combine(Home, Roots.FirstOrDefault(r => Directory.Exists(Path.Combine(Home, r, name))) ?? Roots[2], name);

    private static bool Available(string name) => File.Exists(Path.Combine(Model(name), "sharplight-reference.json"));

    [Fact]
    public void NonMaxSuppressionKeepsTheBestOfOverlappingBoxes()
    {
        BoundingBox[] boxes = [new(0, 0, 10, 10), new(1, 1, 11, 11), new(50, 50, 60, 60), new(0, 0, 10, 10)];
        float[] scores = [0.9f, 0.8f, 0.7f, 0.95f];
        Assert.Equal([3, 2], BoxUtilities.NonMaxSuppression(boxes, scores, 0.5f));
        Assert.Equal([3, 1, 2], BoxUtilities.NonMaxSuppression(boxes, scores, 0.5f, [0, 1, 0, 0]));
        Assert.Equal(1f, boxes[0].IntersectionOverUnion(boxes[3]), 5);
        Assert.Equal(new BoundingBox(5, 15, 15, 25), BoundingBox.FromCenter(10, 20, 10, 10));
    }

    [Theory]
    [InlineData("detr-resnet-50")]
    [InlineData("rtdetr_r18vd")]
    [InlineData("rtdetr_r50vd")]
    [InlineData("rtdetr_v2_r18vd")]
    [InlineData("rtdetr_r101vd_coco_o365")]
    [InlineData("table-transformer-detection")]
    [InlineData("table-transformer-structure-recognition")]
    [InlineData("yolos-tiny")]
    [InlineData("yolos-small")]
    public async Task DetectorMatchesTransformers(string name)
    {
        if (!Available(name))
        {
            return;
        }

        var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(Model(name), "sharplight-reference.json"))).RootElement;
        var expectedDetections = reference.GetProperty("detections").EnumerateArray().ToArray();
        var expectedLogits = reference.GetProperty("logits").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        var image = PngCodec.Read(Path.Combine(Model(name), "sharplight-sample.png"));
        var backends = new List<(string, DeviceSlot)> { ("cpu", Env.Cpu()) };
        if (Env.Vulkan() is { } vulkan)
        {
            backends.Add(("vulkan", vulkan));
        }

        foreach (var (backend, slot) in backends)
        {
            using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = Model(name), Devices = [slot] });
            Assert.Equal("object-detection", model.Description.Kind);
            using var detector = new TaskClientProvider([.. Env.TaskFactories()]).Create<IObjectDetector>(model);
            var result = await detector.DetectAsync(image, new ObjectDetectionOptions { ScoreThreshold = 0.5f }, TestContext.Current.CancellationToken);
            TestContext.Current.SendDiagnosticMessage($"{name} {backend}: {string.Join("; ", result.Objects.Select(o => $"{o.Label} {o.Score:f3}"))}");
            Assert.Equal(expectedDetections.Length, result.Objects.Count);
            var remaining = result.Objects.ToList();
            foreach (var expected in expectedDetections)
            {
                var box = expected.GetProperty("box").EnumerateArray().Select(v => v.GetSingle()).ToArray();
                var match = remaining.Where(o => o.Label == expected.GetProperty("label").GetString())
                    .OrderBy(o => Math.Abs(o.Box.X1 - box[0]) + Math.Abs(o.Box.Y1 - box[1]) + Math.Abs(o.Box.X2 - box[2]) + Math.Abs(o.Box.Y2 - box[3])).FirstOrDefault();
                Assert.NotNull(match);
                Assert.True(Math.Abs(match.Box.X1 - box[0]) < 2 && Math.Abs(match.Box.Y1 - box[1]) < 2 && Math.Abs(match.Box.X2 - box[2]) < 2 && Math.Abs(match.Box.Y2 - box[3]) < 2,
                    $"{name} {backend}: {match.Label} box {match.Box} vs [{string.Join(", ", box)}]");
                Assert.True(Math.Abs(match.Score - expected.GetProperty("score").GetSingle()) < 0.02, $"{name} {backend}: score {match.Score}");
                remaining.Remove(match);
            }
        }

        _ = expectedLogits;
    }

    [Theory]
    [InlineData("depth-anything-v2-small")]
    [InlineData("Depth-Anything-V2-Base-hf")]
    [InlineData("depth-anything-base-hf")]
    [InlineData("Depth-Anything-V2-Large-hf")]
    [InlineData("da3-small")]
    public async Task DepthEstimatorMatchesTransformers(string name)
    {
        string Name = name;
        if (!Available(Name))
        {
            return;
        }

        var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(Model(Name), "sharplight-reference.json"))).RootElement;
        var expected = reference.GetProperty("depth").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        var expectedSample = reference.GetProperty("depth_resized_sample").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        var image = PngCodec.Read(Path.Combine(Model(Name), "sharplight-sample.png"));
        var backends = new List<(string, DeviceSlot)> { ("cpu", Env.Cpu()) };
        if (Env.Vulkan() is { } vulkan)
        {
            backends.Add(("vulkan", vulkan));
        }

        foreach (var (backend, slot) in backends)
        {
            using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = Model(Name), Devices = [slot] });
            Assert.Equal("depth-estimation", model.Description.Kind);
            using var estimator = new TaskClientProvider([.. Env.TaskFactories()]).Create<IDepthEstimator>(model);
            var native = await estimator.EstimateAsync(image, new DepthEstimationOptions { ResizeToInput = false }, TestContext.Current.CancellationToken);
            var (maxAbs, _, rms) = Env.Compare(native.Values, expected);
            TestContext.Current.SendDiagnosticMessage($"{Name} {backend}: depth relative rms {rms:e2}, max abs {maxAbs:e2}");
            Assert.True(rms < 5e-3, $"{backend}: depth relative rms {rms}, max abs {maxAbs}");
            var resized = await estimator.EstimateAsync(image, null, TestContext.Current.CancellationToken);
            Assert.Equal(image.Width, resized.Width);
            var sample = Enumerable.Range(0, expectedSample.Length).Select(i => resized.Values[i * 997]).ToArray();
            Assert.True(Env.Compare(sample, expectedSample).Rms < 5e-3, $"{backend}: resized depth");
        }
    }
}
