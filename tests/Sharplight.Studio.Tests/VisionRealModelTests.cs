using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Vision;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The segmentation, text recognition and upscaling modes of the Vision page on the real engine runtime and the small real models
/// (skipped when they are not on disk): each is recognized by the library, loads through the runtime and runs through the page.
/// </summary>
public sealed class VisionRealModelTests
{
    private static readonly string s_vision = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/vision");

    [Fact]
    public Task Segmentation_RunsWithARealSam2Model() => RunAsync("sam2.1-hiera-tiny", "model.safetensors", ModelKinds.ImageSegmentation, VisionMode.Segment, page =>
    {
        Assert.NotEmpty(page.Rows);
        Assert.True(page.Rows.Max(row => row.Score) > 0.3);
        Assert.NotNull(page.MainPng);
    });

    [Fact]
    public Task Ocr_RunsWithARealTrOcrModel() => RunAsync("trocr-base-printed", "model.safetensors", ModelKinds.Ocr, VisionMode.Ocr, page => Assert.NotEmpty(page.ResultText.Trim()));

    [Fact]
    public Task Upscaling_RunsWithARealRealEsrganModel() => RunAsync("realesrgan-x4plus", "RealESRGAN_x4plus.pth", ModelKinds.ImageUpscaling, VisionMode.Upscale, page =>
    {
        Assert.NotNull(page.AfterPng);
        var input = PngCodec.Decode(page.MainPng!);
        var output = PngCodec.Decode(page.AfterPng!);
        Assert.Equal((input.Width * 4, input.Height * 4), (output.Width, output.Height));
    });

    private static async Task RunAsync(string name, string weights, string kind, VisionMode mode, Action<VisionPageViewModel> check)
    {
        string folder = Path.Combine(s_vision, name);
        string sample = Path.Combine(folder, "sharplight-sample.png");
        Assert.SkipUnless(File.Exists(Path.Combine(folder, weights)) && File.Exists(sample), $"{name} not found");
        using var fakes = new TestServices();
        using var runtime = new EngineModelRuntime(EngineServices.From(TestHost.Services), () => new ResidencySettings());
        Assert.SkipWhen(runtime.Devices.Count == 0, "no compute device");
        var services = fakes.Services with { Runtime = runtime };
        var ct = TestContext.Current.CancellationToken;

        var inspected = ModelInspector.Inspect(folder);
        Assert.Equal(kind, inspected.Kind);
        var settings = new LoadSettings { DeviceId = runtime.Devices[0].Id, ContextLength = 2048, BatchTokens = 256 };
        await runtime.LoadAsync(new ModelLoadRequest(name, folder, settings, [], Kind: inspected.Kind), new SynchronousProgress<LoadStatus>(_ => { }), ct);

        var page = new VisionPageViewModel(services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        Assert.Equal(mode, page.Mode);
        Assert.True(await page.Source.SetFileAsync(sample));
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Single();
        await item.Completion.WaitAsync(TimeSpan.FromMinutes(5), ct);
        await page.Refresh.WaitAsync(ct);

        Assert.True(item.Status == RunStatus.Completed, item.Detail);
        check(page);
        Assert.Single(page.Gallery.Items);
    }
}
