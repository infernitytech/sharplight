using Sharplight.Hub;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Image;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;

namespace Sharplight.Studio.Tests;

/// <summary>
/// Downloads a tiny real Diffusers repository from the Hub as one model (planned file set, converted to a package), then generates
/// through the Image page. Runs only when <c>SHARPLIGHT_HUB_NETWORK_TESTS=1</c> (about 9 MB).
/// </summary>
public sealed class DiffusersDownloadRealTests
{
    private const string EnableVariable = "SHARPLIGHT_HUB_NETWORK_TESTS";
    private const string Repository = "hf-internal-testing/tiny-sd-pipe";

    [Fact]
    public async Task TinyDiffusersRepository_IsDownloadedAsOneModel_AndGeneratesOnTheImagePage()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable(EnableVariable) == "1", $"set {EnableVariable}=1 to run network tests");
        var ct = TestContext.Current.CancellationToken;
        using var directory = new TempDir();
        using var http = HubClientOptions.CreateHttpClient();
        var options = HubClientOptions.FromEnvironment();
        var store = new ModelStore(directory.Path);
        var puller = new ModelPuller(new HubClient(http, options), new FileDownloader(http, options), store, EngineServices.From(TestHost.Services).Converter);

        var result = await puller.PullAsync(ModelReference.Parse(Repository), cancellationToken: ct);

        Assert.True(File.Exists(result.Manifest.PackagePath));
        // The planned set only: no README, no flax or .bin copies; the package replaced the sources.
        Assert.False(Directory.Exists(Path.Combine(result.Manifest.Directory, ModelStore.SourceDirectoryName)));
        Assert.Contains(result.Manifest.SourceFiles, file => file.Path == "unet/diffusion_pytorch_model.safetensors");
        Assert.DoesNotContain(result.Manifest.SourceFiles, file => file.Path == "README.md");

        using var runtime = new EngineModelRuntime(EngineServices.From(TestHost.Services), () => new ResidencySettings());
        Assert.SkipWhen(runtime.Devices.Count == 0, "no compute device");
        using var fakes = new TestServices();
        var services = fakes.Services with { Runtime = runtime };
        var inspected = ModelInspector.Inspect(result.Manifest.PackagePath);
        Assert.Equal(ModelKinds.ImageGeneration, inspected.Kind);
        var settings = new LoadSettings { DeviceId = runtime.Devices[0].Id, ContextLength = 2048, BatchTokens = 256 };
        await runtime.LoadAsync(new ModelLoadRequest("tiny-sd-pipe", result.Manifest.PackagePath, settings, [], Kind: inspected.Kind), new SynchronousProgress<LoadStatus>(_ => { }), ct);

        var page = new ImagePageViewModel(services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        page.Prompt = "a lighthouse";
        page.Panel.Parameters.OfType<NumberParameterViewModel>().Single(parameter => parameter.Key == "steps").Value = 2;
        page.Panel.Parameters.OfType<ChoiceParameterViewModel>().Single(parameter => parameter.Key == "size").Deserialize("128x128");
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Single();
        await item.Completion.WaitAsync(TimeSpan.FromMinutes(3), ct);

        Assert.True(item.Status == RunStatus.Completed, item.Detail);
        Assert.Single(page.Images);
    }
}
