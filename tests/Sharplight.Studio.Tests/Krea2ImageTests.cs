using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Image;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Studio.ViewModels.Models;
using Sharplight.Vision;

namespace Sharplight.Studio.Tests;

/// <summary>
/// Krea 2 transformer-only GGUF files in Studio on the real engine runtime (skipped when the models are not on disk): the library recognizes
/// the file as an image model that needs companions, the load panel shows the text encoder and VAE slots and fills them from the diffusers
/// pipeline folder beside the GGUF's folder (its Qwen3-VL <c>text_encoder</c> folder and the Qwen-Image VAE), and the Image page generates.
/// The real model's root can be moved with <c>SHARPLIGHT_KREA2_ROOT</c>.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class Krea2ImageTests
{
    private static readonly string s_home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    private static readonly string s_tiny = Path.Combine(s_home, "models/krea2/tiny-random-krea2");
    private static readonly string s_root = Environment.GetEnvironmentVariable("SHARPLIGHT_KREA2_ROOT") is { Length: > 0 } root ? root : Path.Combine(s_home, "models/krea2");
    private static readonly string s_gguf = Path.Combine(s_root, "gguf", "Krea-2-Turbo-Q8_0.gguf");

    /// <summary>The tiny GGUF in a <c>gguf</c> folder with the tiny diffusers folder as its sibling pipeline folder, in a scratch folder.</summary>
    private static string TinyLayout()
    {
        string root = Directory.CreateTempSubdirectory("sl-studio-krea2-").FullName;
        Directory.CreateDirectory(Path.Combine(root, "gguf"));
        File.Copy(Path.Combine(s_tiny, "single", "krea2-raw-Q8_0.gguf"), Path.Combine(root, "gguf", "krea2-raw-tiny-Q8_0.gguf"));
        string source = Path.Combine(s_tiny, "diffusers");
        foreach (string file in Directory.EnumerateFiles(source, "*", SearchOption.AllDirectories).Where(f => !f.Contains("transformer", StringComparison.Ordinal)))
        {
            string target = Path.Combine(root, "pipeline", Path.GetRelativePath(source, file));
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            File.Copy(file, target);
        }

        return root;
    }

    [Fact]
    public async Task TinyKrea2_LoadPanelFillsTheCompanions_AndTheImagePageGenerates()
    {
        Assert.SkipUnless(File.Exists(Path.Combine(s_tiny, "single", "krea2-raw-Q8_0.gguf")), "tiny-random-krea2 not found");
        string root = TinyLayout();
        try
        {
            var image = await GenerateAsync(Path.Combine(root, "gguf", "krea2-raw-tiny-Q8_0.gguf"), Path.Combine(root, "pipeline"), "128x128", steps: 2, "krea2-tiny");
            Assert.Equal((128, 128), (image.Width, image.Height));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task Krea2Gguf_GeneratesThroughTheImagePage()
    {
        string pipeline = Path.Combine(s_root, "Krea-2-Turbo");
        Assert.SkipUnless(File.Exists(s_gguf) && Directory.Exists(Path.Combine(pipeline, "text_encoder")), "Krea 2 GGUF or its pipeline folder not found");
        var image = await GenerateAsync(s_gguf, pipeline, "1024x1024", steps: 8, "krea2-turbo");
        Assert.Equal((1024, 1024), (image.Width, image.Height));
        string output = Environment.GetEnvironmentVariable("SHARPLIGHT_SD_TEST_OUTPUT") ?? "";
        if (output.Length > 0)
        {
            Directory.CreateDirectory(output);
            File.WriteAllBytes(Path.Combine(output, "krea2-turbo-studio-1024.png"), PngCodec.Encode(image));
        }
    }

    /// <summary>Recognizes, loads (companions from the load panel) and generates one image through the page.</summary>
    private static async Task<ImageInput> GenerateAsync(string file, string pipeline, string size, int steps, string id, bool expectFolderCompanions = true)
    {
        using var fakes = new TestServices();
        using var runtime = new EngineModelRuntime(EngineServices.From(TestHost.Services), () => new ResidencySettings());
        Assert.SkipWhen(runtime.Devices.Count == 0, "no compute device");
        var services = fakes.Services with { Runtime = runtime };
        var ct = TestContext.Current.CancellationToken;

        Assert.Equal(ModelKinds.ImageGeneration, ImageModelProbe.KindOf(file));
        Assert.True(ImageModelProbe.NeedsCompanions(file));
        var entry = new LibraryEntry { Id = id, DisplayName = id, Path = file, Kind = ModelKinds.ImageGeneration, SizeBytes = new FileInfo(file).Length };
        var panel = new LoadOptionsViewModel(new LibraryEntryViewModel(entry), services, _ => { }, () => { });
        Assert.True(panel.NeedsCompanions);
        Assert.Equal([AuxiliaryRoles.TextEncoder, AuxiliaryRoles.Vae], panel.Companions.Select(c => c.Role));
        Assert.Equal("Text encoder (Qwen3-VL-4B)", panel.Companions[0].Label);
        if (expectFolderCompanions)
        {
            Assert.Equal(Path.Combine(pipeline, "text_encoder"), panel.Companions[0].Path);
            Assert.Equal(Path.Combine(pipeline, "vae", "diffusion_pytorch_model.safetensors"), panel.Companions[1].Path);
        }
        else
        {
            // Companion files beside the transformer are found before the pipeline folder.
            Assert.False(string.IsNullOrEmpty(panel.Companions[0].Path));
            Assert.False(string.IsNullOrEmpty(panel.Companions[1].Path));
        }
        var request = panel.ToRequest() with { Settings = new LoadSettings { DeviceId = runtime.Devices[0].Id, ContextLength = 2048, BatchTokens = 256 } };

        var model = await runtime.LoadAsync(request, new SynchronousProgress<LoadStatus>(_ => { }), ct);
        Assert.Equal(ModelKinds.ImageGeneration, model.Kind);
        var page = new ImagePageViewModel(services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        Assert.Equal(id, page.SelectedModel?.Id);
        page.Prompt = "a lighthouse at dusk";
        page.Panel.Parameters.OfType<NumberParameterViewModel>().Single(parameter => parameter.Key == "steps").Value = steps;
        page.Panel.Parameters.OfType<ChoiceParameterViewModel>().Single(parameter => parameter.Key == "size").Deserialize(size);
        page.Panel.Parameters.OfType<TextParameterViewModel>().Single(parameter => parameter.Key == "seed").Value = "7";
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Single();
        await item.Completion.WaitAsync(TimeSpan.FromMinutes(10), ct);
        Assert.True(item.Status == RunStatus.Completed, item.Detail);
        var generated = Assert.Single(page.Images);
        return PngCodec.Decode(generated.Png);
    }
}
