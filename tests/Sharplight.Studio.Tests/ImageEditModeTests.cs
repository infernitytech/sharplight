using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Image;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Vision;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The Image page's edit mode on the real engine runtime with the tiny random FLUX.2 Klein pipeline (skipped when it is not on disk): the
/// page learns from the model's defaults that it edits, shows the edit input, hides image to image, and an edit run (the prompt as the
/// instruction, the picture as the reference image) gives an image of the picture's size, other than the text-to-image one.
/// </summary>
public sealed class ImageEditModeTests
{
    private static readonly string s_tiny = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/flux2/tiny-random-flux2/diffusers");

    [Fact]
    public async Task EditingModel_ShowsTheEditInput_AndEditsThePicture()
    {
        Assert.SkipUnless(Directory.Exists(s_tiny), "tiny-random-flux2 not found");
        using var fakes = new TestServices();
        using var runtime = new EngineModelRuntime(EngineServices.From(TestHost.Services), () => new ResidencySettings());
        Assert.SkipWhen(runtime.Devices.Count == 0, "no compute device");
        var services = fakes.Services with { Runtime = runtime };
        var ct = TestContext.Current.CancellationToken;
        var settings = new LoadSettings { DeviceId = runtime.Devices[0].Id, ContextLength = 2048, BatchTokens = 256 };
        await runtime.LoadAsync(new ModelLoadRequest("flux2-tiny", s_tiny, settings, [], Kind: ModelKinds.ImageGeneration), null, ct);

        var page = new ImagePageViewModel(services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        await page.ModelDefaults;
        await Task.Delay(100, ct);
        Assert.True(page.SupportsEditing);
        Assert.False(page.IsEditing);
        Assert.True(page.ShowImageToImage);
        page.Prompt = "make it blue";
        page.Panel.Parameters.OfType<TextParameterViewModel>().Single(parameter => parameter.Key == "seed").Value = "7";
        page.Panel.Parameters.OfType<NumberParameterViewModel>().Single(parameter => parameter.Key == "steps").Value = 2;
        page.Panel.Parameters.OfType<ChoiceParameterViewModel>().Single(parameter => parameter.Key == "size").Deserialize("64x64");
        var plain = await RunAsync(page, ct);

        var pixels = new float[3 * 64 * 96];
        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = (i * 37 % 101) / 100f;
        }

        page.EditImage.Set(PlanarImage.FromPlanar(pixels, 96, 64, 0f, 1f), "picture.png");
        Assert.True(page.IsEditing);
        Assert.False(page.ShowImageToImage);
        var edited = await RunAsync(page, ct);
        Assert.NotEqual(plain.Width, edited.Width);
        await runtime.UnloadAsync("flux2-tiny");
    }

    private static async Task<ImageInput> RunAsync(ImagePageViewModel page, CancellationToken ct)
    {
        int before = page.Queue.Items.Count;
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Skip(before).Single();
        await item.Completion.WaitAsync(TimeSpan.FromMinutes(5), ct);
        Assert.True(item.Status == RunStatus.Completed, item.Detail);
        return PngCodec.Decode(Assert.Single(page.Images).Png);
    }
}
