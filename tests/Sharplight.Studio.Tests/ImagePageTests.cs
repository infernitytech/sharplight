using Sharplight.ImageGeneration;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Studio.Services.Imaging;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Image;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;

namespace Sharplight.Studio.Tests;

/// <summary>The Image page over a fake image generator.</summary>
public sealed class ImagePageTests
{
    private static FakeModel Generator(TestServices services, out FakeImageGenerator client)
    {
        client = new FakeImageGenerator();
        var model = new FakeModel("sd-tiny") { Kind = ModelKinds.ImageGeneration };
        model.Clients[typeof(IImageGenerator)] = client;
        services.Runtime.Add(model);
        return model;
    }

    private static TextParameterViewModel Text(ImagePageViewModel page, string key) => page.Panel.Parameters.OfType<TextParameterViewModel>().Single(parameter => parameter.Key == key);

    private static NumberParameterViewModel Number(ImagePageViewModel page, string key) => page.Panel.Parameters.OfType<NumberParameterViewModel>().Single(parameter => parameter.Key == key);

    private static ChoiceParameterViewModel Choice(ImagePageViewModel page, string key) => page.Panel.Parameters.OfType<ChoiceParameterViewModel>().Single(parameter => parameter.Key == key);

    private static string WritePng(TestServices services, string name, int size = 32)
    {
        string path = services.Directory + "/" + name;
        PngCodec.Write(path, FakeImageGenerator.Gradient(size, size, 3));
        return path;
    }

    [Fact]
    public async Task Generate_ShowsPreviewsAndSteps_AndKeepsPngsWithMetadata()
    {
        using var services = new TestServices();
        Generator(services, out var client);
        var page = new ImagePageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        Assert.True(page.HasModel);
        Assert.False(page.RunCommand.CanExecute(null));
        Assert.Contains("Type a prompt", page.RunHint, StringComparison.Ordinal);

        page.Prompt = "a lighthouse at dusk";
        page.NegativePrompt = "blurry";
        Number(page, "steps").Value = 6;
        Number(page, "guidance").Value = 5;
        Number(page, "count").Value = 2;
        Text(page, "seed").Value = "7";
        Choice(page, "sampler").Deserialize("euler");
        Choice(page, "size").Deserialize("128x128");
        var previews = new List<byte[]>();
        var steps = new List<string>();
        page.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(ImagePageViewModel.PreviewPng) && page.PreviewPng is { } preview)
            {
                previews.Add(preview);
            }

            if (args.PropertyName == nameof(ImagePageViewModel.StepText) && page.StepText.Length > 0)
            {
                steps.Add(page.StepText);
            }
        };
        Assert.True(page.RunCommand.CanExecute(null));
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        Assert.Equal(RunStatus.Completed, page.Queue.Items[0].Status);
        var (request, options) = Assert.Single(client.Requests);
        Assert.Equal("a lighthouse at dusk", request.Prompt);
        Assert.Equal("blurry", request.NegativePrompt);
        Assert.Null(request.Image);
        Assert.Equal(6, options.Steps);
        Assert.Equal(5f, options.GuidanceScale);
        Assert.Equal(7, options.Seed);
        Assert.Equal(2, options.Count);
        Assert.Equal(Samplers.Euler, options.Scheduler);
        Assert.Equal((128, 128), (options.Width, options.Height));
        Assert.True(options.PreviewInterval > 0);
        Assert.NotEmpty(previews);
        Assert.Contains("Step 6 of 6", steps);

        Assert.Equal(2, page.Images.Count);
        Assert.Equal([7L, 8L], page.Images.Select(image => image.Seed));
        Assert.True(page.HasSeveral);
        Assert.Null(page.PreviewPng);
        Assert.Equal(128, page.SelectedImage!.Image.Width);
        var text = PngMetadata.Read(page.SelectedImage.Png);
        Assert.Equal("a lighthouse at dusk", text["Prompt"]);
        Assert.Equal("blurry", text["Negative prompt"]);
        Assert.Equal("7", text["Seed"]);
        Assert.Contains("Steps: 6, Sampler: euler, CFG scale: 5.0, Seed: 7, Size: 128x128, Model: sd-tiny", text["parameters"], StringComparison.Ordinal);
        Assert.Equal(128, PngCodec.Decode(page.SelectedImage.Png).Width);

        var output = Assert.Single(page.Gallery.Items);
        Assert.Equal(["image-1.png", "image-2.png", "parameters.json"], output.Files.Select(file => file.Name));
        Assert.True(output.HasThumbnail);
        Assert.True(output.CanReuse);
        Assert.Contains("seed: 7", output.Facts);
        Assert.Contains("size: 128 × 128", output.Facts);
        Assert.Equal("a lighthouse at dusk", Assert.Single(page.History.Items).Title);
    }

    [Fact]
    public async Task RandomSeed_IsReportedAndReused_ByTheParametersOfTheRun()
    {
        using var services = new TestServices();
        Generator(services, out var client);
        var page = new ImagePageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        page.Prompt = "a red bicycle";
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
        Assert.Null(client.Requests[0].Options.Seed);
        Assert.Equal(424242, page.SelectedImage!.Seed);

        // "Reuse seed" puts the seed into the parameters; the next run is the same picture.
        page.ReuseSeedCommand.Execute(null);
        Assert.Equal("424242", Text(page, "seed").Value);

        // The history keeps the seed that was used, so the run can be repeated after the page was closed.
        page.Prompt = "something else";
        Text(page, "seed").Value = "";
        var reopened = new ImagePageViewModel(services.Services, new FakeNavigator());
        reopened.Panel.ResetToDefaultsCommand.Execute(null);
        reopened.History.RestoreCommand.Execute(reopened.History.Items[0]);
        Assert.Equal("a red bicycle", reopened.Prompt);
        Assert.Equal("424242", Text(reopened, "seed").Value);
        Assert.Single(reopened.Images);
        Assert.True(reopened.HasResult);

        // The gallery's "Reuse parameters" does the same.
        reopened.Prompt = "";
        reopened.Gallery.Items[0].ReuseCommand.Execute(null);
        Assert.Equal("a red bicycle", reopened.Prompt);
    }

    [Fact]
    public async Task InputImage_MakesImageToImage_AndAMaskInpainting()
    {
        using var services = new TestServices();
        Generator(services, out var client);
        var page = new ImagePageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        page.Prompt = "make it autumn";
        var strength = page.Panel.Parameters.Single(parameter => parameter.Key == "strength");
        Assert.False(strength.IsVisibleNow);

        Assert.True(await page.Source.SetFileAsync(WritePng(services, "photo.png", 48)));
        Assert.True(strength.IsVisibleNow);
        Assert.True(page.UsesInputImage);
        Number(page, "strength").Value = 0.4;
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
        var (first, _) = client.Requests[0];
        Assert.Equal(48, first.Image!.Width);
        Assert.Equal(0.4f, first.Strength, 3);
        Assert.Null(first.Mask);
        Assert.Contains("mode: image to image", page.Gallery.Items[0].Facts);

        Assert.True(await page.Mask.SetFileAsync(WritePng(services, "mask.png", 48)));
        page.RunCommand.Execute(null);
        await page.Queue.Items[1].Completion.WaitAsync(TestContext.Current.CancellationToken);
        Assert.NotNull(client.Requests[1].Request.Mask);
        Assert.Contains("mode: inpainting", page.Gallery.Items[0].Facts);

        // A generated image becomes the next input.
        page.UseAsInputCommand.Execute(null);
        Assert.Equal(page.SelectedImage!.Image, page.Source.Image);
        page.Source.Clear();
        Assert.False(strength.IsVisibleNow);
    }

    [Fact]
    public async Task Cancelling_StopsTheRun_AndKeepsNothing()
    {
        using var services = new TestServices();
        Generator(services, out var client);
        client.Gate = new TaskCompletionSource();
        var page = new ImagePageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        page.Prompt = "a very slow picture";
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Single();
        for (int i = 0; i < 400 && client.Requests.Count == 0; i++)
        {
            await Task.Delay(10, TestContext.Current.CancellationToken);
        }

        page.CancelCommand.Execute(null);
        await item.Completion.WaitAsync(TestContext.Current.CancellationToken);
        Assert.Equal(RunStatus.Cancelled, item.Status);
        Assert.Empty(page.History.Items);
        Assert.Empty(page.Images);
        Assert.Null(page.PreviewPng);
    }

    [Fact]
    public async Task Presets_SetStepsGuidanceAndSampler_AndASeedThatIsNotANumberIsRefused()
    {
        using var services = new TestServices();
        Generator(services, out var client);
        var page = new ImagePageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        Assert.Equal(["Fast", "Quality"], page.Panel.Presets.Select(preset => preset.Name));
        page.Panel.SelectedPreset = page.Panel.Presets.Single(preset => preset.Name == "Quality");
        Assert.Equal(35, page.Panel.Options.Steps);
        Assert.Equal(Samplers.DpmPlusPlus2MKarras, page.Panel.Options.Scheduler);
        page.Panel.SelectedPreset = page.Panel.Presets.Single(preset => preset.Name == "Fast");
        Assert.Equal(12, page.Panel.Options.Steps);

        page.Prompt = "x";
        Text(page, "seed").Value = "abc";
        page.RunCommand.Execute(null);
        Assert.Contains("seed must be a whole number", page.Notice, StringComparison.Ordinal);
        Assert.Empty(page.Queue.Items);
        Assert.Empty(client.Requests);
    }

    [Fact]
    public async Task AFailingModel_ShowsTheErrorAndIsKeptInTheHistory()
    {
        using var services = new TestServices();
        var model = new FakeModel("broken") { Kind = ModelKinds.ImageGeneration };
        services.Runtime.Add(model);
        var page = new ImagePageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        page.Prompt = "anything";
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
        Assert.Equal(RunStatus.Failed, page.Queue.Items[0].Status);
        Assert.Contains("no IImageGenerator", page.Notice, StringComparison.Ordinal);
        Assert.Equal(Sharplight.Studio.Services.Pages.RunOutcome.Failed, Assert.Single(page.History.Items).Record.Outcome);
        Assert.Empty(page.Gallery.Items);
    }

    [Fact]
    public async Task SavePng_WritesTheImageWithItsMetadata()
    {
        using var services = new TestServices();
        Generator(services, out _);
        var page = new ImagePageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        page.Prompt = "a lighthouse";
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
        string target = services.Directory + "/out.png";
        services.Platform.SaveTo = target;
        await page.SaveImageCommand.ExecuteAsync(null);
        var written = await File.ReadAllBytesAsync(target, TestContext.Current.CancellationToken);
        Assert.Equal("a lighthouse", PngMetadata.Read(written)["Prompt"]);
        Assert.Equal(64, PngCodec.Decode(written).Width);
    }
}
