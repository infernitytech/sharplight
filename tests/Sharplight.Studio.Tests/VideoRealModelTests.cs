using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Studio.ViewModels.Video;

namespace Sharplight.Studio.Tests;

/// <summary>The Video page on the tiny random Wan model through the real engine runtime on the CPU (skipped when the model is not on disk).</summary>
public sealed class VideoRealModelTests
{
    private static readonly string s_tinyWan = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/video/tiny-random-wan/t2v/diffusers");

    [Fact]
    public async Task VideoPage_GeneratesWithTheTinyWanModel_AndExportsGifAndWebM()
    {
        Assert.SkipUnless(Directory.Exists(s_tinyWan), "tiny-random-wan not found");
        using var fakes = new TestServices();
        using var runtime = new EngineModelRuntime(EngineServices.From(TestHost.Services), () => new ResidencySettings());
        var cpu = runtime.Devices.FirstOrDefault(device => !device.IsGpu);
        Assert.SkipWhen(cpu is null, "no CPU device");
        var services = fakes.Services with { Runtime = runtime };
        var ct = TestContext.Current.CancellationToken;

        var inspected = ModelInspector.Inspect(s_tinyWan);
        Assert.Equal(ModelKinds.VideoGeneration, inspected.Kind);
        var settings = new LoadSettings { DeviceId = cpu!.Id, ContextLength = 2048, BatchTokens = 256 };
        var model = await runtime.LoadAsync(new ModelLoadRequest("tiny-wan", s_tinyWan, settings, [], Kind: inspected.Kind), new SynchronousProgress<LoadStatus>(_ => { }), ct);
        Assert.Equal(ModelKinds.VideoGeneration, model.Kind);

        var page = new VideoPageViewModel(services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        Assert.Equal("tiny-wan", page.SelectedModel?.Id);
        await page.ModelDefaults;

        // The model's defaults (Wan: 81 frames) arrive through the dispatcher; set the small values only after them.
        var deadline = DateTime.UtcNow.AddSeconds(30);
        while (Number(page, "frames").IntValue == 33 && DateTime.UtcNow < deadline)
        {
            await Task.Delay(50, ct);
        }

        await page.ModelDefaults;
        await Task.Delay(200, ct);
        page.Prompt = "a red fox";
        Number(page, "width").Value = 48;
        Number(page, "height").Value = 32;
        Number(page, "frames").Value = 9;
        Number(page, "steps").Value = 2;
        page.Panel.Parameters.OfType<TextParameterViewModel>().Single(parameter => parameter.Key == "seed").Value = "3";
        Assert.Equal(9, Number(page, "frames").IntValue);
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Single();
        await item.Completion.WaitAsync(TimeSpan.FromMinutes(10), ct);
        await Task.Delay(200, ct);

        Assert.True(item.Status == RunStatus.Completed, item.Detail);
        Assert.Equal(3, page.Result!.Seed);
        Assert.Equal(9, page.Result.Frames.Count);
        Assert.Equal(9, page.Player.Frames.Count);
        Assert.Equal((48, 32), (page.Result.Frames[0].Width, page.Result.Frames[0].Height));

        string folder = Path.Combine(fakes.Directory, "export");
        Directory.CreateDirectory(folder);
        await page.ExportToAsync(Path.Combine(folder, "clip.gif"));
        Assert.True(new FileInfo(Path.Combine(folder, "clip.gif")).Length > 0);
        if (page.Export.Formats.Single(format => format.Format == "webm").IsAvailable)
        {
            await page.ExportToAsync(Path.Combine(folder, "clip.webm"));
            Assert.True(new FileInfo(Path.Combine(folder, "clip.webm")).Length > 0);
        }

        Assert.Equal("", page.Notice);
    }

    private static NumberParameterViewModel Number(VideoPageViewModel page, string key) => page.Panel.Parameters.OfType<NumberParameterViewModel>().Single(parameter => parameter.Key == key);
}
