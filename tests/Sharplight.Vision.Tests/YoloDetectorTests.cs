using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Vision.Tests;

/// <summary>
/// Ultralytics YOLO exports through the task layer, against onnxruntime plus the Ultralytics decoding (<c>tools/yolo-reference.py</c> writes the reference next to
/// <c>model.onnx</c> in <c>~/models/vision-r7/&lt;name&gt;</c>). Skipped when the export or its reference is not present.
/// </summary>
public class YoloDetectorTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    private static string Dir(string name) => Path.Combine(Home, "models/vision-r7", name);

    [Fact]
    public void NamesAreParsedFromThePythonNamesDictionary()
    {
        Assert.Equal(["person", "it's", "car"], Sharplight.Import.Ultralytics.UltralyticsImporter.ParseNames("{0: 'person', 1: \"it's\", 2: 'car'}"));
    }

    [Theory]
    [InlineData("yolov8n", false)]
    [InlineData("yolov8n-seg", false)]
    [InlineData("yolov10n", false)]
    [InlineData("yolov8n-seg", true)]
    public async Task DetectionsMatchTheReference(string name, bool throughPackage)
    {
        string dir = Dir(name);
        if (!File.Exists(Path.Combine(dir, "sharplight-reference.json")))
        {
            return;
        }

        var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "sharplight-reference.json"))).RootElement;
        var expected = reference.GetProperty("detections").EnumerateArray().ToArray();
        double conf = reference.GetProperty("conf").GetDouble();
        var image = PngCodec.Read(Path.Combine(dir, "sharplight-sample.png"));
        string path = Path.Combine(dir, "model.onnx");
        string? package = null;
        if (throughPackage)
        {
            package = Path.Combine(dir, "model.slm");
            TestHost.Converter.Convert(path, package);
            path = package;
        }

        try
        {
            var backends = new List<(string, DeviceSlot)> { ("cpu", Env.Cpu()) };
            if (Env.Vulkan() is { } vulkan)
            {
                backends.Add(("vulkan", vulkan));
            }

            foreach (var (backend, slot) in backends)
            {
                using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = path, Devices = [slot] });
                Assert.Equal("object-detection", model.Description.Kind);
                using var detector = new TaskClientProvider([.. Env.TaskFactories()]).Create<IObjectDetector>(model);
                var result = await detector.DetectAsync(image, new ObjectDetectionOptions { ScoreThreshold = (float)conf, MaxDetections = 300 }, TestContext.Current.CancellationToken);
                TestContext.Current.SendDiagnosticMessage($"{name} {backend}: {string.Join("; ", result.Objects.Select(o => $"{o.Label} {o.Score:f3}"))}");
                Assert.Equal(expected.Length, result.Objects.Count);
                var remaining = result.Objects.ToList();
                foreach (var e in expected)
                {
                    var box = e.GetProperty("box").EnumerateArray().Select(v => v.GetSingle()).ToArray();
                    int cls = e.GetProperty("class").GetInt32();
                    var match = remaining.Where(o => o.ClassIndex == cls)
                        .OrderBy(o => Math.Abs(o.Box.X1 - box[0]) + Math.Abs(o.Box.Y1 - box[1]) + Math.Abs(o.Box.X2 - box[2]) + Math.Abs(o.Box.Y2 - box[3])).FirstOrDefault();
                    Assert.NotNull(match);
                    Assert.True(Math.Abs(match.Box.X1 - box[0]) < 2 && Math.Abs(match.Box.Y1 - box[1]) < 2 && Math.Abs(match.Box.X2 - box[2]) < 2 && Math.Abs(match.Box.Y2 - box[3]) < 2,
                        $"{name} {backend}: {match.Label} box {match.Box} vs [{string.Join(", ", box)}]");
                    Assert.True(Math.Abs(match.Score - e.GetProperty("score").GetSingle()) < 0.01, $"{name} {backend}: score {match.Score} vs {e.GetProperty("score").GetSingle()}");
                    if (e.TryGetProperty("mask_rle", out var rle))
                    {
                        Assert.NotNull(match.Mask);
                        var expectedMask = Convert.FromHexString(rle.GetString()!);
                        long inter = 0;
                        long union = 0;
                        for (int i = 0; i < match.Mask.Pixels.Length; i++)
                        {
                            bool a = match.Mask.Pixels[i];
                            bool b = (expectedMask[i >> 3] & (0x80 >> (i & 7))) != 0;
                            inter += a && b ? 1 : 0;
                            union += a || b ? 1 : 0;
                        }

                        double iou = union == 0 ? 1 : (double)inter / union;
                        TestContext.Current.SendDiagnosticMessage($"{name} {backend}: {match.Label} mask IoU {iou:f4}");
                        Assert.True(iou > 0.95, $"{name} {backend}: {match.Label} mask IoU {iou}");
                    }

                    remaining.Remove(match);
                }
            }
        }
        finally
        {
            if (package is not null)
            {
                File.Delete(package);
            }
        }
    }
}
