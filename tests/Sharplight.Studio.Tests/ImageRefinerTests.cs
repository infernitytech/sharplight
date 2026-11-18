using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Image;
using Sharplight.Studio.ViewModels.Inference.Parameters;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The Image page's refiner on the real engine runtime with the tiny random SDXL pipeline loaded twice (skipped when it is not on disk): the
/// second resident model is offered as the first one's refiner, choosing it shows the switch point, and a run hands the latents over
/// (the steps of both stages are counted once, the result is recorded with the refiner).
/// </summary>
public sealed class ImageRefinerTests
{
    private static readonly string s_tiny = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/sd/tiny-random-sdxl/diffusers");

    [Fact]
    public async Task SecondImageModel_IsOfferedAsRefiner_AndFinishesTheRun()
    {
        Assert.SkipUnless(Directory.Exists(s_tiny), "tiny-random-sdxl not found");
        using var fakes = new TestServices();
        using var runtime = new EngineModelRuntime(EngineServices.From(TestHost.Services), () => new ResidencySettings());
        Assert.SkipWhen(runtime.Devices.Count == 0, "no compute device");
        var services = fakes.Services with { Runtime = runtime };
        var ct = TestContext.Current.CancellationToken;
        var settings = new LoadSettings { DeviceId = runtime.Devices[0].Id, ContextLength = 2048, BatchTokens = 256 };
        await runtime.LoadAsync(new ModelLoadRequest("xl-base", s_tiny, settings, [], Kind: ModelKinds.ImageGeneration), null, ct);
        await runtime.LoadAsync(new ModelLoadRequest("xl-refiner", s_tiny, settings, [], Kind: ModelKinds.ImageGeneration), null, ct);

        var page = new ImagePageViewModel(services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        page.SelectModel(runtime.LoadedModels.Single(model => model.Id == "xl-base"));
        await page.ModelDefaults;
        Assert.True(page.HasRefiners);
        var refiner = page.Refiners.Single(choice => choice.Model?.Id == "xl-refiner");
        Assert.DoesNotContain(page.Refiners, choice => choice.Model?.Id == "xl-base");
        var start = page.Panel.Parameters.OfType<NumberParameterViewModel>().Single(parameter => parameter.Key == "refinerStart");
        Assert.False(start.IsVisibleNow);
        page.SelectedRefiner = refiner;
        Assert.True(start.IsVisibleNow);

        page.Prompt = "a red cube";
        page.Panel.Parameters.OfType<TextParameterViewModel>().Single(parameter => parameter.Key == "seed").Value = "3";
        page.Panel.Parameters.OfType<NumberParameterViewModel>().Single(parameter => parameter.Key == "steps").Value = 10;
        start.Value = 0.5;
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Single();
        await item.Completion.WaitAsync(TimeSpan.FromMinutes(5), ct);
        await Task.Delay(200, ct);
        Assert.NotNull(page.SelectedImage);
        Assert.Contains("refiner: xl-refiner from 0.5", page.ResultSummary, StringComparison.Ordinal);
        await runtime.UnloadAsync("xl-refiner");
        await runtime.UnloadAsync("xl-base");
    }
}
