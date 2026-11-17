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
/// FLUX transformer-only files in Studio on the real engine runtime (skipped when the models are not on disk): the library recognizes
/// the file as an image model that needs companions, the load panel fills in the text encoders and the VAE found beside it and
/// passes them as auxiliary files, and the Image page generates through the loaded model.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class FluxImageTests
{
    private static readonly string s_home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    private static readonly string s_tinyFlux = Path.Combine(s_home, "models/flux/tiny-random-flux/schnell");
    private static readonly string s_schnellGguf = Path.Combine(s_home, "models/flux/gguf/flux1-schnell-Q8_0.gguf");

    /// <summary>The tiny transformer-only GGUF with bare companion files (no configuration or tokenizer beside them) in a scratch folder.</summary>
    private static string TinyBareLayout()
    {
        string folder = Directory.CreateTempSubdirectory("sl-studio-flux-").FullName;
        string diffusers = Path.Combine(s_tinyFlux, "diffusers");
        File.Copy(Path.Combine(s_tinyFlux, "single", "flux-Q8_0.gguf"), Path.Combine(folder, "flux-tiny-Q8_0.gguf"));
        File.Copy(Path.Combine(diffusers, "text_encoder", "model.safetensors"), Path.Combine(folder, "clip_l.safetensors"));
        File.Copy(Path.Combine(diffusers, "text_encoder_2", "model.safetensors"), Path.Combine(folder, "t5xxl_fp16.safetensors"));
        File.Copy(Path.Combine(diffusers, "vae", "diffusion_pytorch_model.safetensors"), Path.Combine(folder, "ae.safetensors"));
        return folder;
    }

    [Fact]
    public async Task TinyFlux_LoadPanelFillsTheCompanions_AndTheImagePageGenerates()
    {
        Assert.SkipUnless(File.Exists(Path.Combine(s_tinyFlux, "single", "flux-Q8_0.gguf")), "tiny-random-flux not found");
        string folder = TinyBareLayout();
        try
        {
            string file = Path.Combine(folder, "flux-tiny-Q8_0.gguf");
            var image = await GenerateAsync(file, "128x128", steps: 2, "flux-tiny");
            Assert.Equal((128, 128), (image.Width, image.Height));
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    [Fact]
    public async Task FluxSchnellGguf_GeneratesThroughTheImagePage()
    {
        Assert.SkipUnless(File.Exists(s_schnellGguf), "FLUX.1-schnell GGUF not found");
        var image = await GenerateAsync(s_schnellGguf, "512x512", steps: 4, "flux-schnell");
        Assert.Equal((512, 512), (image.Width, image.Height));
        string output = Environment.GetEnvironmentVariable("SHARPLIGHT_SD_TEST_OUTPUT") ?? "";
        if (output.Length > 0)
        {
            Directory.CreateDirectory(output);
            File.WriteAllBytes(Path.Combine(output, "flux-schnell-studio-512.png"), PngCodec.Encode(image));
        }
    }

    /// <summary>Recognizes, loads (companions from the load panel) and generates one image through the page.</summary>
    private static async Task<ImageInput> GenerateAsync(string file, string size, int steps, string id)
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
        Assert.False(panel.IsLanguageModel);
        string folder = Path.GetDirectoryName(file)!;
        Assert.Equal([AuxiliaryRoles.TextEncoder, AuxiliaryRoles.SecondTextEncoder, AuxiliaryRoles.Vae], panel.Companions.Select(c => c.Role));
        Assert.All(panel.Companions, companion => Assert.Equal(folder, Path.GetDirectoryName(companion.Path)));
        var request = panel.ToRequest() with { Settings = new LoadSettings { DeviceId = runtime.Devices[0].Id, ContextLength = 2048, BatchTokens = 256 } };
        Assert.Equal([AuxiliaryRoles.TextEncoder, AuxiliaryRoles.SecondTextEncoder, AuxiliaryRoles.Vae], request.Auxiliary!.Select(a => a.Role));

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
        Assert.Equal(7, generated.Seed);
        return PngCodec.Decode(generated.Png);
    }
}
