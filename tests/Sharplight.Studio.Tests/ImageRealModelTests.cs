using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Image;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Studio.ViewModels.Vision;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The Image and Vision pages on the real engine runtime and small real models (skipped when they are not on disk): the models are
/// recognized by the library, load through the runtime's residency and run through the pages.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class ImageRealModelTests
{
    private static readonly string s_home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    private static readonly string s_tinySd = Path.Combine(s_home, "models/sd/tiny-random-sd/diffusers");
    private static readonly string s_vit = Path.Combine(s_home, "models/vision/vit-base-patch16-224");

    private static LoadSettings Settings(EngineModelRuntime runtime) => new() { DeviceId = runtime.Devices[0].Id, ContextLength = 2048, BatchTokens = 256 };

    [Fact]
    public async Task ImagePage_GeneratesWithARealDiffusersModel_LoadedThroughTheRuntime()
    {
        Assert.SkipUnless(Directory.Exists(s_tinySd), "tiny-random-sd not found");
        using var fakes = new TestServices();
        using var runtime = new EngineModelRuntime(EngineServices.From(TestHost.Services), () => new ResidencySettings());
        Assert.SkipWhen(runtime.Devices.Count == 0, "no compute device");
        var services = fakes.Services with { Runtime = runtime };
        var ct = TestContext.Current.CancellationToken;

        var inspected = ModelInspector.Inspect(s_tinySd);
        Assert.Equal(ModelKinds.ImageGeneration, inspected.Kind);
        var model = await runtime.LoadAsync(new ModelLoadRequest("tiny-sd", s_tinySd, Settings(runtime), [], Kind: inspected.Kind), new SynchronousProgress<LoadStatus>(_ => { }), ct);
        Assert.Equal(ModelKinds.ImageGeneration, model.Kind);

        var page = new ImagePageViewModel(services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        Assert.Equal("tiny-sd", page.SelectedModel?.Id);
        page.Prompt = "a lighthouse";
        page.Panel.Parameters.OfType<NumberParameterViewModel>().Single(parameter => parameter.Key == "steps").Value = 3;
        page.Panel.Parameters.OfType<ChoiceParameterViewModel>().Single(parameter => parameter.Key == "size").Deserialize("128x128");
        page.Panel.Parameters.OfType<TextParameterViewModel>().Single(parameter => parameter.Key == "seed").Value = "5";
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Single();
        await item.Completion.WaitAsync(TimeSpan.FromMinutes(3), ct);

        Assert.True(item.Status == RunStatus.Completed, item.Detail);
        var image = Assert.Single(page.Images);
        Assert.Equal(5, image.Seed);
        var decoded = PngCodec.Decode(image.Png);
        Assert.Equal((128, 128), (decoded.Width, decoded.Height));
        Assert.Equal("a lighthouse", Studio.Services.Imaging.PngMetadata.Read(image.Png)["Prompt"]);
        Assert.Single(page.Gallery.Items);
    }

    [Fact]
    public async Task VisionPage_ClassifiesWithARealViTModel_LoadedThroughTheRuntime()
    {
        string sample = Path.Combine(s_vit, "sharplight-sample.png");
        Assert.SkipUnless(File.Exists(Path.Combine(s_vit, "model.safetensors")) && File.Exists(sample), "vit-base-patch16-224 not found");
        using var fakes = new TestServices();
        using var runtime = new EngineModelRuntime(EngineServices.From(TestHost.Services), () => new ResidencySettings());
        Assert.SkipWhen(runtime.Devices.Count == 0, "no compute device");
        var services = fakes.Services with { Runtime = runtime };
        var ct = TestContext.Current.CancellationToken;

        var inspected = ModelInspector.Inspect(s_vit);
        Assert.Equal(ModelKinds.ImageClassification, inspected.Kind);
        await runtime.LoadAsync(new ModelLoadRequest("vit", s_vit, Settings(runtime), [], Kind: inspected.Kind), new SynchronousProgress<LoadStatus>(_ => { }), ct);

        var page = new VisionPageViewModel(services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        Assert.Equal(VisionMode.Classify, page.Mode);
        Assert.True(await page.Source.SetFileAsync(sample));
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Single();
        await item.Completion.WaitAsync(TimeSpan.FromMinutes(3), ct);
        await page.Refresh.WaitAsync(ct);

        Assert.True(item.Status == RunStatus.Completed, item.Detail);
        Assert.Equal(5, page.Rows.Count);
        Assert.True(page.Rows[0].Score >= page.Rows[1].Score);
        Assert.NotEmpty(page.Rows[0].Label);
        Assert.Single(page.Gallery.Items);
    }
}
