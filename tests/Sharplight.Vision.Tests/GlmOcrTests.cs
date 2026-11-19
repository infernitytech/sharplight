using System.Text;
using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Templating;

namespace Sharplight.Vision.Tests;

/// <summary>
/// GLM-OCR (<c>zai-org/GLM-OCR</c>: CogViT-style tower with query / key norms and a GLM decoder; its multi-token-prediction layer is not read) as an image-to-text model,
/// against transformers (<c>tools/glm-ocr-reference.py</c> writes <c>sharplight-sample.png</c> and <c>sharplight-reference.json</c> next to the checkpoint): the
/// greedy text of "Text Recognition:" must be exactly the reference's. Skipped when the checkpoint or its reference is not present.
/// </summary>
public class GlmOcrTests
{
    private static readonly string Dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/vision-r10/glm-ocr");

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void Greedy_text_matches_transformers(bool gpu, bool package)
    {
        string refPath = Path.Combine(Dir, "sharplight-reference.json");
        Assert.SkipUnless(File.Exists(refPath), "GLM-OCR or its reference is not available");
        var slot = gpu ? Env.Vulkan() : Env.Cpu();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        var reference = JsonDocument.Parse(File.ReadAllText(refPath)).RootElement;
        string expected = reference.GetProperty("text").GetString()!;
        var image = PngCodec.Read(Path.Combine(Dir, "sharplight-sample.png"));
        string scratch = Path.Combine(Path.GetTempPath(), "sl-slm-" + Guid.NewGuid().ToString("N"));
        try
        {
            string path = Dir;
            if (package)
            {
                Directory.CreateDirectory(scratch);
                path = Path.Combine(scratch, "model.slm");
                TestHost.Converter.Convert(Dir, path);
            }

            Check(path, slot!, gpu, package, expected, image);
        }
        finally
        {
            if (Directory.Exists(scratch))
            {
                Directory.Delete(scratch, recursive: true);
            }
        }
    }

    /// <summary>The same model behind <see cref="ITextRecognizer"/>: "text" (the default) is the reference's greedy text, "formula" and "table" its answers to the other prompts.</summary>
    [Theory]
    [InlineData(false, 64)]
    [InlineData(true, 64)]
    public async Task Recognizer_tasks_match_transformers(bool gpu, int maxTokens)
    {
        string refPath = Path.Combine(Dir, "sharplight-reference.json");
        Assert.SkipUnless(File.Exists(refPath), "GLM-OCR or its reference is not available");
        var slot = gpu ? Env.Vulkan() : Env.Cpu();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        var reference = JsonDocument.Parse(File.ReadAllText(refPath)).RootElement;
        Assert.SkipUnless(reference.TryGetProperty("tasks", out var tasks), "the reference has no other tasks (rerun tools/glm-ocr-reference.py)");
        var image = PngCodec.Read(Path.Combine(Dir, "sharplight-sample.png"));
        using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = Dir, Devices = [slot!], ContextLength = 1024 });
        using var recognizer = new TaskClientProvider([.. Env.TaskFactories()]).Create<ITextRecognizer>(model);
        Assert.Equal(["text", "formula", "table"], recognizer.Tasks);
        var options = new TextRecognitionOptions { MaxTokens = maxTokens };
        var plain = await recognizer.RecognizeAsync(image, options, TestContext.Current.CancellationToken);
        Assert.Equal(reference.GetProperty("text").GetString()!.Trim(), plain.Text);
        foreach (var name in new[] { "formula", "table" })
        {
            options.Task = name;
            var result = await recognizer.RecognizeAsync(image, options, TestContext.Current.CancellationToken);
            Assert.Equal(tasks.GetProperty(name).GetString()!.Trim(), result.Text);
        }
    }

    private static void Check(string path, DeviceSlot slot, bool gpu, bool package, string expected, ImageInput image)
    {
        using var model = LlmModel.Load(TestHost.Loader, path, new LlmLoadOptions { Devices = [slot], ContextLength = 1024 });
        using var vision = model.CreateVisionEncoder();
        using var session = model.CreateSession(WellKnown.DecodeMultimodalGraph);
        var gen = new Generator(session, model.Tokenizer, vision);
        var template = new ChatTemplate(model.ChatTemplate!, "", "");
        var prompt = template.Render([new ChatMessage("user", Parts: [ChatContentPart.Image(), ChatContentPart.FromText("Text Recognition:")])], addGenerationPrompt: true,
            new Dictionary<string, object?>());
        var sb = new StringBuilder();
        var ids = new List<int>();
        foreach (var t in gen.Generate(prompt, [image], new GenerationOptions { MaxNewTokens = 64, Sampling = SamplingOptions.Greedy }))
        {
            sb.Append(t.Text);
            ids.Add(t.Id);
        }

        TestContext.Current.SendDiagnosticMessage($"{(gpu ? "Vulkan" : "CPU")}{(package ? " (.slm)" : "")}: '{sb}' (reference '{expected}')");
        Assert.Equal(expected, sb.ToString().Trim());
    }
}
