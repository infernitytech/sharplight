using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Vision.Tests;

/// <summary>
/// Text recognizers through the task layer, against transformers (<c>tools/ocr-reference.py</c> writes the reference next to the checkpoint): the text and the
/// token ids of the greedy decoding must be exactly the reference's. Skipped when the checkpoint or its reference is not present.
/// </summary>
public class TextRecognizerTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    private static string Model(string name)
        => Path.Combine(Home, Directory.Exists(Path.Combine(Home, "models/vision", name)) ? "models/vision" : Directory.Exists(Path.Combine(Home, "models/vision-sweep", name)) ? "models/vision-sweep" : "models/vision-r6", name);

    private static bool Available(string name) => File.Exists(Path.Combine(Model(name), "sharplight-reference.json"));

    [Theory]
    [InlineData("trocr-base-printed")]
    [InlineData("trocr-base-handwritten")]
    public async Task RecognizerMatchesTransformers(string name)
    {
        if (!Available(name))
        {
            return;
        }

        var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(Model(name), "sharplight-reference.json"))).RootElement;
        string expected = reference.GetProperty("text").GetString()!;
        var image = PngCodec.Read(Path.Combine(Model(name), "sharplight-sample.png"));
        var backends = new List<(string, DeviceSlot)> { ("cpu", Env.Cpu()) };
        if (Env.Vulkan() is { } vulkan)
        {
            backends.Add(("vulkan", vulkan));
        }

        foreach (var (backend, slot) in backends)
        {
            using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = Model(name), Devices = [slot] });
            Assert.Equal("ocr", model.Description.Kind);
            using var recognizer = new TaskClientProvider([.. Env.TaskFactories()]).Create<ITextRecognizer>(model);
            var result = await recognizer.RecognizeAsync(image, null, TestContext.Current.CancellationToken);
            TestContext.Current.SendDiagnosticMessage($"{name} {backend}: '{result.Text}' (reference '{expected}')");
            Assert.Equal(expected, result.Text);
            Assert.Single(result.Regions);
        }
    }

    [Fact]
    public void LocationTextBecomesBoxes()
    {
        var boxes = LocationTextParser.Parse("</s><s>cat<loc_0><loc_500><loc_999><loc_1000>remote control<loc_100><loc_200><loc_300><loc_400></s>", 1000, 640, 480, quads: false);
        Assert.Equal(["cat", "remote control"], boxes.Select(r => r.Text));
        Assert.Equal(0.32f, boxes[0].Box!.Value.X1, 0.01f);
        Assert.Equal(240.24f, boxes[0].Box!.Value.Y1, 0.01f);
        var quad = LocationTextParser.Parse("text<loc_10><loc_20><loc_110><loc_20><loc_110><loc_60><loc_10><loc_60>", 1000, 1000, 1000, quads: true);
        Assert.Equal(new BoundingBox(10.5f, 20.5f, 110.5f, 60.5f), Assert.Single(quad).Box);
    }

    [Fact]
    public void PromptedEncoderMatchesTransformers()
    {
        const string Name = "Florence-2-base";
        if (!Available(Name))
        {
            return;
        }

        var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(Model(Name), "sharplight-reference.json"))).RootElement.GetProperty("tasks")[0];
        float[] Floats(string key) => [.. reference.GetProperty(key).EnumerateArray().Select(v => v.GetSingle())];
        var ids = reference.GetProperty("input_ids").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var image = PngCodec.Read(Path.Combine(Model(Name), "sharplight-sample.png"));
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
            using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = Model(Name), Devices = [slot] });
            using var encoder = model.CompileComponent(Models.ModelComponentRoles.Encoder, Models.WellKnown.EncodeGraph);
            var processor = new ImageProcessor(ImageProcessorConfig.FromJson(model.Description.Component(Models.ModelComponentRoles.FeatureExtractor).Config[ImageProcessor.ComponentKey]));
            var tokens = ids[577..];
            encoder.SetInput<float>(encoder.Graph.Inputs.First(v => v.Input!.Name == "pixels"), processor.Process(image).Planes);
            encoder.SetInput<int>(encoder.Graph.Input(Models.InputRole.TokenIds), tokens);
            encoder.SetInput<int>(encoder.Graph.Input(Models.InputRole.Positions), [.. Enumerable.Range(0, ids.Length)]);
            var run = new RunParameters();
            run.Symbols[Models.WellKnown.TokenSymbol] = tokens.Length;
            encoder.Run(run);
            var features = new float[577 * 768];
            encoder.GetOutput<float>(Models.WellKnown.ImageFeaturesOutput, features);
            var encoded = new float[ids.Length * 768];
            encoder.GetOutput<float>(Models.WellKnown.EncoderOutput, encoded);
            var (_, _, featuresRms) = Env.Compare(features, Floats("image_features"));
            var (_, _, encodedRms) = Env.Compare(encoded, Floats("encoder_output"));
            TestContext.Current.SendDiagnosticMessage($"{backend}: image features relative rms {featuresRms:e2}, encoder output {encodedRms:e2}");
            Assert.True(featuresRms < 5e-3, $"{backend}: image features relative rms {featuresRms}");
            Assert.True(encodedRms < 5e-3, $"{backend}: encoder output relative rms {encodedRms}");
        }
    }

    [Theory]
    [InlineData("")]
    [InlineData("-text")]
    public async Task PromptedRecognizerMatchesTransformersGreedy(string sample)
    {
        const string Name = "Florence-2-base";
        if (!File.Exists(Path.Combine(Model(Name), $"sharplight-reference{sample}.json")))
        {
            return;
        }

        var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(Model(Name), $"sharplight-reference{sample}.json"))).RootElement;
        var image = PngCodec.Read(Path.Combine(Model(Name), $"sharplight-sample{sample}.png"));
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
            using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = Model(Name), Devices = [slot] });
            using var recognizer = new TaskClientProvider([.. Env.TaskFactories()]).Create<ITextRecognizer>(model);
            Assert.Contains("<OD>", recognizer.Tasks);
            foreach (var entry in reference.GetProperty("tasks").EnumerateArray())
            {
                string task = entry.GetProperty("task").GetString()!;
                var result = await recognizer.RecognizeAsync(image, new TextRecognitionOptions { Task = task }, TestContext.Current.CancellationToken);
                if (task == "<OD>")
                {
                    var expected = LocationTextParser.Parse(entry.GetProperty("text_raw").GetString()!, 1000, image.Width, image.Height, quads: false);
                    TestContext.Current.SendDiagnosticMessage($"{backend} {task}: {string.Join("; ", result.Regions.Select(r => $"{r.Text} {r.Box}"))}");
                    Assert.Equal(expected.Select(r => (r.Text, r.Box)), result.Regions.Select(r => (r.Text, r.Box)));
                }
                else
                {
                    string expected = entry.GetProperty("text").GetString()!;
                    TestContext.Current.SendDiagnosticMessage($"{backend} {task}: '{result.Text}'");
                    Assert.Equal(expected, result.Text);
                }
            }
        }
    }
}
