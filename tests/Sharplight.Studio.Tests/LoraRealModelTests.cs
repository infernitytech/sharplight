using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Gallery;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Image;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>A tiny Stable Diffusion pipeline and a tiny LoRA through the Image page and the load panel on the real runtime (skipped when not on disk).</summary>
public sealed class LoraRealModelTests
{
    private static readonly string s_home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    private static readonly string s_tinySd = Path.Combine(s_home, "models/sd/tiny-random-sd/diffusers");
    private static readonly string s_lora = Path.Combine(s_home, "models/lora/tiny-lora/sd1/peft.safetensors");

    private static async Task<byte[]> GenerateAsync(ImagePageViewModel page)
    {
        page.Prompt = "a lighthouse";
        page.Panel.Parameters.OfType<NumberParameterViewModel>().Single(parameter => parameter.Key == "steps").Value = 3;
        page.Panel.Parameters.OfType<ChoiceParameterViewModel>().Single(parameter => parameter.Key == "size").Deserialize("128x128");
        page.Panel.Parameters.OfType<TextParameterViewModel>().Single(parameter => parameter.Key == "seed").Value = "5";
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Last();
        await item.Completion.WaitAsync(TimeSpan.FromMinutes(3), TestContext.Current.CancellationToken);
        Assert.True(item.Status == RunStatus.Completed, item.Detail);
        return page.Images.Single().Png;
    }

    [Fact]
    public async Task TinyPipeline_WithATinyLora_ThroughTheLoadPanelAndThePage()
    {
        Assert.SkipUnless(Directory.Exists(s_tinySd) && File.Exists(s_lora), "tiny-random-sd or the tiny LoRA not found");
        using var fakes = new TestServices();
        using var runtime = new EngineModelRuntime(EngineServices.From(TestHost.Services), () => new ResidencySettings());
        Assert.SkipWhen(runtime.Devices.Count == 0, "no compute device");
        var services = fakes.Services with { Runtime = runtime };
        var ct = TestContext.Current.CancellationToken;
        Assert.Equal(ModelKindCatalog.Lora, ModelInspector.Inspect(s_lora).Kind);

        var entry = new LibraryEntry { Id = "tiny-sd", DisplayName = "tiny-sd", Path = s_tinySd, Kind = ModelKinds.ImageGeneration };
        var lora = new LibraryEntry { Id = "peft", DisplayName = "peft", Path = s_lora, Kind = ModelKindCatalog.Lora };
        var panel = new LoadOptionsViewModel(new LibraryEntryViewModel(entry), services, _ => { }, () => { }, () => [entry, lora]);
        Assert.Equal([lora], panel.Loras.Available);
        var settings = new LoadSettings { DeviceId = runtime.Devices[0].Id, ContextLength = 2048, BatchTokens = 256 };
        await runtime.LoadAsync(panel.ToRequest() with { Settings = settings }, null, ct);

        var page = new ImagePageViewModel(services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        Assert.Empty(page.Loras.Slots);
        byte[] plain = await GenerateAsync(page);

        // Add the LoRA on the page and reload: the engine merges it at load.
        page.Loras.Add(s_lora, 0.8);
        Assert.True(page.LorasPending);
        await page.ApplyLorasCommand.ExecuteAsync(null);
        Assert.False(page.LorasPending);
        Assert.Equal([new ModelLora(s_lora, 0.8f)], LoraRoles.Adapters(runtime.LoadedModels.Single().Auxiliary));
        await page.ActivateAsync();
        Assert.Equal("tiny-sd", page.SelectedModel?.Id);

        byte[] adapted = await GenerateAsync(page);
        Assert.NotEqual(PngCodec.Decode(plain).Rgb, PngCodec.Decode(adapted).Rgb);
        var stored = new ImageGalleryStore(services.Pages.GenerationsDirectory("images")).Load();
        Assert.Equal(["peft"], stored.First().Loras.Select(item => item.Name));
        Assert.Empty(stored.Last().Loras);
        await runtime.UnloadAsync("tiny-sd");
    }
}
