using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Vision.Tests;

/// <summary>
/// Promptable segmenters through the task layer, against transformers (<c>tools/segmentation-reference.py</c> writes the reference next to the
/// checkpoint). Skipped when the checkpoint or its reference is not present.
/// </summary>
public class ImageSegmenterTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    private static string Model(string name) => Path.Combine(Home, Directory.Exists(Path.Combine(Home, "models/vision", name)) ? "models/vision" : "models/vision-sweep", name);

    private static bool Available(string name) => File.Exists(Path.Combine(Model(name), "sharplight-reference.json"));

    [Theory]
    [InlineData("sam2.1-hiera-tiny")]
    public async Task SegmenterMatchesTransformers(string name)
    {
        if (!Available(name))
        {
            return;
        }

        var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(Model(name), "sharplight-reference.json"))).RootElement;
        var image = PngCodec.Read(Path.Combine(Model(name), "sharplight-sample.png"));
        var backends = new List<(string, DeviceSlot)> { ("cpu", Env.Cpu()) };
        if (Env.Vulkan() is { } vulkan)
        {
            backends.Add(("vulkan", vulkan));
        }

        foreach (var (backend, slot) in backends)
        {
            using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = Model(name), Devices = [slot] });
            Assert.Equal("image-segmentation", model.Description.Kind);
            using var segmenter = new TaskClientProvider([.. Env.TaskFactories()]).Create<IImageSegmenter>(model);
            var encoded = await segmenter.EncodeImageAsync(image, TestContext.Current.CancellationToken);
            foreach (var expected in reference.GetProperty("prompts").EnumerateArray())
            {
                var prompt = Prompt(expected);
                var result = await segmenter.SegmentAsync(encoded, prompt, null, TestContext.Current.CancellationToken);
                var scores = expected.GetProperty("iou_scores").EnumerateArray().Select(v => v.GetSingle()).ToArray();
                var masks = expected.GetProperty("masks").EnumerateArray().Select(v => Unpack(v.GetString()!, image.Width * image.Height)).ToArray();
                Assert.Equal(masks.Length, result.Masks.Count);
                for (int i = 0; i < masks.Length; i++)
                {
                    double iou = Iou(result.Masks[i].Pixels, masks[i]);
                    TestContext.Current.SendDiagnosticMessage($"{name} {backend} {expected.GetProperty("name").GetString()} mask {i}: IoU {iou:f4}, score {result.Masks[i].Score:f4} vs {scores[i]:f4}");
                    Assert.True(iou > 0.98, $"{backend} {expected.GetProperty("name").GetString()} mask {i}: IoU {iou}");
                    Assert.True(Math.Abs(result.Masks[i].Score - scores[i]) < 0.02, $"{backend}: score {result.Masks[i].Score} vs {scores[i]}");
                }

                Assert.True(result.ObjectScore > 0);
                var single = await segmenter.SegmentAsync(encoded, prompt, new SegmentationOptions { MultipleMasks = false }, TestContext.Current.CancellationToken);
                Assert.Single(single.Masks);
            }
        }
    }

    [Fact]
    public async Task OneShotOverloadSegments()
    {
        const string Name = "sam2.1-hiera-tiny";
        if (!Available(Name))
        {
            return;
        }

        var image = PngCodec.Read(Path.Combine(Model(Name), "sharplight-sample.png"));
        using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = Model(Name), Devices = [Env.Cpu()] });
        using var segmenter = new TaskClientProvider([.. Env.TaskFactories()]).Create<IImageSegmenter>(model);
        var result = await segmenter.SegmentAsync(image, SegmentationPrompt.FromPoint(450, 200), null, TestContext.Current.CancellationToken);
        Assert.Equal(3, result.Masks.Count);
        Assert.All(result.Masks, m => Assert.True(m.Area > 1000));
        await Assert.ThrowsAsync<ArgumentException>(() => segmenter.SegmentAsync(image, new SegmentationPrompt(), null, TestContext.Current.CancellationToken));
    }

    private static SegmentationPrompt Prompt(JsonElement expected)
    {
        if (expected.TryGetProperty("box", out var box))
        {
            var b = box.EnumerateArray().Select(v => v.GetSingle()).ToArray();
            return SegmentationPrompt.FromBox(new BoundingBox(b[0], b[1], b[2], b[3]));
        }

        var points = expected.GetProperty("points").EnumerateArray().Select(p => p.EnumerateArray().Select(v => v.GetSingle()).ToArray()).ToArray();
        var labels = expected.GetProperty("labels").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        return new SegmentationPrompt { Points = [.. points.Select((p, i) => new SegmentationPoint(p[0], p[1], labels[i] == 1))] };
    }

    private static bool[] Unpack(string base64, int count)
    {
        var bytes = Convert.FromBase64String(base64);
        var bits = new bool[count];
        for (int i = 0; i < count; i++)
        {
            bits[i] = (bytes[i / 8] & (0x80 >> (i % 8))) != 0;
        }

        return bits;
    }

    private static double Iou(bool[] a, bool[] b)
    {
        int inter = 0;
        int union = 0;
        for (int i = 0; i < a.Length; i++)
        {
            inter += a[i] && b[i] ? 1 : 0;
            union += a[i] || b[i] ? 1 : 0;
        }

        return union == 0 ? 1 : (double)inter / union;
    }
}
