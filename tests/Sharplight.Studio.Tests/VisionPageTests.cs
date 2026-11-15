using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Studio.ViewModels.Vision;
using Sharplight.Vision;

namespace Sharplight.Studio.Tests;

/// <summary>The Vision page over fake vision clients: the mode follows the kind of the loaded model.</summary>
public sealed class VisionPageTests
{
    private static FakeModel Vision(TestServices services, string kind, Type client, out FakeVision fake, string id = "vision")
    {
        fake = new FakeVision();
        var model = new FakeModel(id) { Kind = kind };
        model.Clients[client] = fake;
        services.Runtime.Add(model);
        return model;
    }

    private static async Task<VisionPageViewModel> OpenAsync(TestServices services, string imageName = "street.png")
    {
        var page = new VisionPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        string path = services.Directory + "/" + imageName;
        PngCodec.Write(path, FakeImageGenerator.Gradient(80, 64, 2));
        Assert.True(await page.Source.SetFileAsync(path));
        return page;
    }

    private static async Task RunAsync(VisionPageViewModel page)
    {
        Assert.True(page.RunCommand.CanExecute(null), page.RunHint);
        page.RunCommand.Execute(null);
        await page.Queue.Items[^1].Completion.WaitAsync(TestContext.Current.CancellationToken);
        await page.Refresh.WaitAsync(TestContext.Current.CancellationToken);
    }

    private static NumberParameterViewModel Number(VisionPageViewModel page, string key) => page.Panel.Parameters.OfType<NumberParameterViewModel>().Single(parameter => parameter.Key == key);

    [Fact]
    public async Task TheModeFollowsTheKindOfTheLoadedModel()
    {
        using var services = new TestServices();
        var page = new VisionPageViewModel(services.Services, new FakeNavigator());
        Assert.Equal(VisionMode.None, page.Mode);
        Assert.Contains("Load an image classification", page.RunHint, StringComparison.Ordinal);

        var classifier = Vision(services, ModelKinds.ImageClassification, typeof(IImageClassifier), out _, "vit");
        var detector = Vision(services, ModelKinds.ObjectDetection, typeof(IObjectDetector), out _, "detr");
        Vision(services, ModelKinds.DepthEstimation, typeof(IDepthEstimator), out _, "depth");
        Vision(services, ModelKinds.ImageEmbedding, typeof(IImageEmbedder), out _, "clip");
        await page.ActivateAsync();
        Assert.Equal(4, page.Picker.Models.Count);

        page.SelectModel(classifier);
        Assert.Equal((VisionMode.Classify, "Classify"), (page.Mode, page.RunText));
        Assert.True(Visible(page, "topk"));
        Assert.False(Visible(page, "threshold"));
        page.SelectModel(detector);
        Assert.Equal((VisionMode.Detect, "Detect objects"), (page.Mode, page.RunText));
        Assert.True(Visible(page, "threshold"));
        Assert.False(Visible(page, "colormap"));
        page.Picker.SelectedModel = page.Picker.Models.Single(option => option.Id == "depth");
        Assert.Equal(VisionMode.Depth, page.Mode);
        Assert.True(Visible(page, "colormap"));
        page.Picker.SelectedModel = page.Picker.Models.Single(option => option.Id == "clip");
        Assert.Equal(VisionMode.Embed, page.Mode);
        Assert.True(page.ShowCandidates);
        page.Candidates = "cat, dog";
        Assert.Equal("Score labels", page.RunText);
    }

    private static bool Visible(VisionPageViewModel page, string key) => page.Panel.Parameters.Single(parameter => parameter.Key == key).IsVisibleNow;

    [Fact]
    public async Task Classification_ListsTheTopClasses()
    {
        using var services = new TestServices();
        Vision(services, ModelKinds.ImageClassification, typeof(IImageClassifier), out var client);
        var page = await OpenAsync(services);
        Number(page, "topk").Value = 3;
        await RunAsync(page);

        Assert.Equal(3, ((ImageClassificationOptions)client.Options[0]!).TopK);
        Assert.Equal(["tabby cat", "tiger cat", "Egyptian cat"], page.Rows.Select(row => row.Label));
        Assert.Equal("71%", page.Rows[0].ScoreText);
        Assert.Equal("9.0%", page.Rows[2].ScoreText);
        Assert.True(page.ShowRows);
        Assert.NotNull(page.MainPng);
        var output = Assert.Single(page.Gallery.Items);
        Assert.Equal(["result.json"], output.Files.Select(file => file.Name));
        Assert.Contains("top: tabby cat 71%", output.Facts);
        Assert.StartsWith("tabby cat 71%", output.Preview, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ZeroShot_ScoresTheCandidates_AndWithoutLabelsEmbedsTheImage()
    {
        using var services = new TestServices();
        Vision(services, ModelKinds.ImageEmbedding, typeof(IImageEmbedder), out var client);
        var page = await OpenAsync(services);

        page.Candidates = "a cat\nthe sea, a tree; a cat";
        await RunAsync(page);
        Assert.Equal(["a photo of a cat", "a photo of the sea", "a photo of a tree"], client.Candidates[0]);
        Assert.Equal("a tree", page.Rows[0].Label);
        Assert.Equal("zero-shot", page.ResultMode);
        Assert.Equal(["a cat", "a tree", "the sea"], page.Rows.Select(row => row.Label).Order());

        page.Candidates = "";
        await RunAsync(page);
        Assert.Equal("embedding", page.ResultMode);
        Assert.True(page.ShowEmbedding);
        Assert.StartsWith("8 dimensions. First values: 0.000, 0.100", page.EmbeddingText, StringComparison.Ordinal);
        Assert.Contains("dimensions: 8", page.Gallery.Items[0].Facts);
        Assert.True(((ImageEmbeddingOptions)client.Options[1]!).Normalize);
    }

    [Fact]
    public async Task AModelWithoutATextTower_IgnoresLabelsWithANote()
    {
        using var services = new TestServices();
        Vision(services, ModelKinds.ImageEmbedding, typeof(IImageEmbedder), out var client);
        client.SupportsText = false;
        var page = await OpenAsync(services);
        page.Candidates = "cat";
        await RunAsync(page);
        Assert.Equal("embedding", page.ResultMode);
        Assert.Contains("no text tower", page.Notice, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Detection_FiltersByTheThreshold_AfterTheRun_AndExports()
    {
        using var services = new TestServices();
        Vision(services, ModelKinds.ObjectDetection, typeof(IObjectDetector), out var client);
        var page = await OpenAsync(services);
        await RunAsync(page);

        // The model is asked with a low floor, the page shows what the threshold keeps.
        Assert.Equal(VisionPageViewModel.DetectionFloor, ((ObjectDetectionOptions)client.Options[0]!).ScoreThreshold);
        Assert.Equal(["cat", "dog"], page.Rows.Select(row => row.Label));
        Assert.Equal("93%", page.Rows[0].ScoreText);
        Assert.True(page.Rows[0].HasColor);
        byte[] atHalf = page.MainPng!;

        Number(page, "threshold").Value = 0.8;
        await page.Refresh.WaitAsync(TestContext.Current.CancellationToken);
        Assert.Equal(["cat"], page.Rows.Select(row => row.Label));
        Assert.Contains("more below the threshold", page.RowsHint, StringComparison.Ordinal);
        Assert.NotEqual(atHalf, page.MainPng);
        Number(page, "threshold").Value = 0.1;
        await page.Refresh.WaitAsync(TestContext.Current.CancellationToken);
        Assert.Equal(["cat", "dog", "cup"], page.Rows.Select(row => row.Label));
        Number(page, "threshold").Value = 1;
        await page.Refresh.WaitAsync(TestContext.Current.CancellationToken);
        Assert.Empty(page.Rows);
        Assert.Contains("No objects above", page.RowsHint, StringComparison.Ordinal);

        // Exports follow the threshold: the annotated PNG has the image's size, the JSON the shown objects.
        Number(page, "threshold").Value = 0.5;
        await page.Refresh.WaitAsync(TestContext.Current.CancellationToken);
        string png = services.Directory + "/annotated.png";
        services.Platform.SaveTo = png;
        await page.ExportImageCommand.ExecuteAsync(null);
        var decoded = PngCodec.Decode(await File.ReadAllBytesAsync(png, TestContext.Current.CancellationToken));
        Assert.Equal((80, 64), (decoded.Width, decoded.Height));
        string json = services.Directory + "/result.json";
        services.Platform.SaveTo = json;
        await page.ExportJsonCommand.ExecuteAsync(null);
        using var document = JsonDocument.Parse(await File.ReadAllBytesAsync(json, TestContext.Current.CancellationToken));
        Assert.Equal(2, document.RootElement.GetProperty("Boxes").GetArrayLength());
        Assert.Equal("detection", document.RootElement.GetProperty("Mode").GetString());

        var output = Assert.Single(page.Gallery.Items);
        Assert.Equal(["annotated.png", "result.json"], output.Files.Select(file => file.Name));
        Assert.True(output.HasThumbnail);
        Assert.Contains("objects: 2", output.Facts);
    }

    [Fact]
    public async Task Depth_ShowsTheMapBesideTheImage_AndRecolorsWithoutANewRun()
    {
        using var services = new TestServices();
        Vision(services, ModelKinds.DepthEstimation, typeof(IDepthEstimator), out var client);
        var page = await OpenAsync(services);
        await RunAsync(page);

        Assert.True(page.ShowDepth);
        Assert.NotNull(page.MainPng);
        Assert.NotNull(page.DepthPng);
        var turbo = PngCodec.Decode(page.DepthPng!);
        Assert.Equal((80, 64), (turbo.Width, turbo.Height));
        Assert.NotEqual(turbo.Rgb[0], turbo.Rgb[2]);

        page.Panel.Parameters.OfType<ChoiceParameterViewModel>().Single(parameter => parameter.Key == "colormap").Deserialize("gray");
        await page.Refresh.WaitAsync(TestContext.Current.CancellationToken);
        var gray = PngCodec.Decode(page.DepthPng!);
        Assert.Equal(gray.Rgb[0], gray.Rgb[2]);
        Assert.Equal(0, gray.Rgb[0]);
        Assert.Equal(255, gray.Rgb[^1]);
        Assert.Single(client.Options);

        string target = services.Directory + "/depth.png";
        services.Platform.SaveTo = target;
        await page.ExportImageCommand.ExecuteAsync(null);
        Assert.Equal(gray.Rgb, PngCodec.Decode(await File.ReadAllBytesAsync(target, TestContext.Current.CancellationToken)).Rgb);
        Assert.Equal(["depth.png", "result.json"], page.Gallery.Items[0].Files.Select(file => file.Name));
    }

    [Fact]
    public async Task History_RestoresTheResultAndParameters_AlsoWithoutTheOriginalImage()
    {
        using var services = new TestServices();
        Vision(services, ModelKinds.ObjectDetection, typeof(IObjectDetector), out _);
        var page = await OpenAsync(services, "kitchen.png");
        Number(page, "threshold").Value = 0.6;
        await RunAsync(page);
        Assert.Equal(["cat", "dog"], page.Rows.Select(row => row.Label));

        File.Delete(services.Directory + "/kitchen.png");
        var reopened = new VisionPageViewModel(services.Services, new FakeNavigator());
        await reopened.ActivateAsync();
        reopened.Panel.ResetToDefaultsCommand.Execute(null);
        reopened.History.RestoreCommand.Execute(reopened.History.Items[0]);
        await reopened.Refresh.WaitAsync(TestContext.Current.CancellationToken);
        Assert.Equal(0.6, Number(reopened, "threshold").Value, 3);
        Assert.Equal(["cat", "dog"], reopened.Rows.Select(row => row.Label));
        Assert.NotNull(reopened.MainPng);
        Assert.Equal("detection", reopened.ResultMode);
    }

    [Fact]
    public async Task AFileThatIsNotAnImage_IsRefusedWithANotice()
    {
        using var services = new TestServices();
        Vision(services, ModelKinds.ImageClassification, typeof(IImageClassifier), out _);
        var page = await OpenAsync(services);
        string text = services.Directory + "/notes.txt";
        await File.WriteAllTextAsync(text, "hello", TestContext.Current.CancellationToken);
        Assert.False(await page.Source.SetFileAsync(text));
        Assert.Contains("not an image file", page.Notice, StringComparison.Ordinal);
        string broken = services.Directory + "/broken.png";
        await File.WriteAllBytesAsync(broken, [1, 2, 3, 4], TestContext.Current.CancellationToken);
        Assert.False(await page.Source.SetFileAsync(broken));
        Assert.Contains("cannot be read", page.Notice, StringComparison.Ordinal);
        Assert.True(page.Source.HasImage);
    }
}
