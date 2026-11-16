using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Vision.Tests;

/// <summary>
/// Per-pixel segmenters through the task layer, against transformers (<c>tools/vision-reference.py</c> writes the class scores and the label map at the input size
/// next to the checkpoint). Skipped when the checkpoint or its reference is not present.
/// </summary>
public class SemanticSegmenterTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    private static string Model(string name)
        => Path.Combine(Home, Directory.Exists(Path.Combine(Home, "models/vision", name)) ? "models/vision" : Directory.Exists(Path.Combine(Home, "models/vision-sweep", name)) ? "models/vision-sweep" : "models/vision-r6", name);

    private static bool Available(string name) => File.Exists(Path.Combine(Model(name), "sharplight-reference.json"));

    [Fact]
    public void ResultDescribesClassesAndMasks()
    {
        var result = new SemanticSegmentation(2, 2, [0, 1, 1, 1], ["wall", "sky"]);
        Assert.Equal("sky", result.NameOf(1));
        Assert.Equal("class 7", result.NameOf(7));
        Assert.Equal([0, 255, 255, 255], result.Mask(1));
        Assert.Equal(("sky", 3), (result.Classes[0].Name, result.Classes[0].Pixels));
    }

    [Theory]
    [InlineData("BiRefNet")]
    [InlineData("RMBG-2.0")]
    public async Task MatteMatchesReference(string name)
    {
        if (!Available(name))
        {
            return;
        }

        var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(Model(name), "sharplight-reference.json"))).RootElement;
        var expected = reference.GetProperty("logits").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        var image = PngCodec.Read(Path.Combine(Model(name), "sharplight-sample.png"));
        var backends = new List<(string, DeviceSlot)>();
        if (Env.Vulkan() is { } vulkan)
        {
            backends.Add(("vulkan", vulkan));
        }

        if (Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_CPU") == "1")
        {
            backends.Add(("cpu", Env.Cpu()));
        }

        foreach (var (backend, slot) in backends)
        {
            using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = Model(name), Devices = [slot] });
            Assert.Equal("image-segmentation-semantic", model.Description.Kind);
            using var segmenter = new TaskClientProvider([.. Env.TaskFactories()]).Create<ISemanticSegmenter>(model);
            var result = await segmenter.SegmentAsync(image, new SemanticSegmentationOptions { IncludeLogits = true }, TestContext.Current.CancellationToken);
            var (maxAbs, _, rms) = Env.Compare(result.Logits!, expected);
            Assert.NotNull(result.Alpha);
            int agree = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                agree += (result.Logits![i] > 0) == (expected[i] > 0) ? 1 : 0;
            }

            double agreement = (double)agree / expected.Length;
            TestContext.Current.SendDiagnosticMessage($"{name} {backend}: matte logits relative rms {rms:e2}, max abs {maxAbs:e2}, foreground agreement {agreement:p3}");
            var (alphaMax, _, alphaRms) = Env.Compare([.. result.Logits!.Select(Sigmoid)], [.. expected.Select(Sigmoid)]);
            TestContext.Current.SendDiagnosticMessage($"{name} {backend}: alpha max abs {alphaMax:e2}, relative rms {alphaRms:e2}");
            int off = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                off += MathF.Abs(Sigmoid(result.Logits![i]) - Sigmoid(expected[i])) > 0.05f ? 1 : 0;
            }

            TestContext.Current.SendDiagnosticMessage($"{name} {backend}: {off} of {expected.Length} pixels with alpha difference above 0.05");
            Assert.True(alphaMax < 0.01f && off == 0, $"{backend}: {off} pixels with an alpha difference above 0.05 (max {alphaMax})");
            Assert.True(agreement > 0.999, $"{backend}: foreground agreement {agreement}");
        }
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    [Theory]
    [InlineData("segformer-b0-finetuned-ade-512-512")]
    [InlineData("segformer-b5-finetuned-ade-640-640")]
    [InlineData("segformer-b2-finetuned-cityscapes-1024-1024")]
    public async Task SegmenterMatchesTransformers(string name)
    {
        if (!Available(name))
        {
            return;
        }

        var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(Model(name), "sharplight-reference.json"))).RootElement;
        var expectedLogits = reference.GetProperty("logits").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        var expectedLabels = reference.GetProperty("label_map").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var image = PngCodec.Read(Path.Combine(Model(name), "sharplight-sample.png"));
        var backends = new List<(string, DeviceSlot)> { ("cpu", Env.Cpu()) };
        if (Env.Vulkan() is { } vulkan)
        {
            backends.Add(("vulkan", vulkan));
        }

        foreach (var (backend, slot) in backends)
        {
            using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = Model(name), Devices = [slot] });
            Assert.Equal("image-segmentation-semantic", model.Description.Kind);
            using var segmenter = new TaskClientProvider([.. Env.TaskFactories()]).Create<ISemanticSegmenter>(model);
            var result = await segmenter.SegmentAsync(image, new SemanticSegmentationOptions { IncludeLogits = true }, TestContext.Current.CancellationToken);
            var (maxAbs, _, rms) = Env.Compare(result.Logits!, expectedLogits);
            int agree = 0;
            for (int i = 0; i < expectedLabels.Length; i++)
            {
                agree += result.Labels[i] == expectedLabels[i] ? 1 : 0;
            }

            double agreement = (double)agree / expectedLabels.Length;
            TestContext.Current.SendDiagnosticMessage($"{name} {backend}: logits relative rms {rms:e2}, max abs {maxAbs:e2}, label agreement {agreement:p3}, top class {result.Classes[0].Name}");
            Assert.Equal(image.Width * image.Height, result.Labels.Length);
            Assert.True(rms < 5e-3, $"{backend}: logits relative rms {rms}, max abs {maxAbs}");
            Assert.True(agreement > 0.995, $"{backend}: label agreement {agreement}");
        }
    }
}
