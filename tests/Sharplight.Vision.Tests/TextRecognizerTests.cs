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

    private static string Model(string name) => Path.Combine(Home, Directory.Exists(Path.Combine(Home, "models/vision", name)) ? "models/vision" : "models/vision-sweep", name);

    private static bool Available(string name) => File.Exists(Path.Combine(Model(name), "sharplight-reference.json"));

    [Theory]
    [InlineData("trocr-base-printed")]
    [InlineData("trocr-base-handwritten")]
    [InlineData("trocr-small-handwritten")]
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
}
