using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Studio.Tests.Fakes;
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
    public async Task Semantic_ShowsAnOverlayWithALegend_AndExportsIt()
    {
        using var services = new TestServices();
        Vision(services, ModelKinds.SemanticSegmentation, typeof(ISemanticSegmenter), out var client);
        var page = await OpenAsync(services);
        Assert.Equal(VisionMode.Semantic, page.Mode);
        await RunAsync(page);

        Assert.Equal("semantic", page.ResultMode);
        Assert.Equal(["sky", "road"], page.Rows.Select(row => row.Label));
        Assert.All(page.Rows, row => Assert.True(row.HasColor));
        Assert.NotEqual(page.Rows[0].Color, page.Rows[1].Color);
        Assert.Equal(0.5f, page.Rows[0].Score, 2);
        var overlay = PngCodec.Decode(page.MainPng!);
        Assert.Equal((80, 64), (overlay.Width, overlay.Height));
        Assert.NotEqual(overlay.Rgb[0..3], overlay.Rgb[(3 * 79)..(3 * 79 + 3)]);
        Assert.Single(client.Options);

        string target = services.Directory + "/semantic.png";
        services.Platform.SaveTo = target;
        await page.ExportImageCommand.ExecuteAsync(null);
        Assert.Equal(overlay.Rgb, PngCodec.Decode(await File.ReadAllBytesAsync(target, TestContext.Current.CancellationToken)).Rgb);
    }

    [Fact]
    public async Task Semantic_OfAMattingModel_CutsTheSubjectOutWithAnAlphaChannel()
    {
        using var services = new TestServices();
        Vision(services, ModelKinds.SemanticSegmentation, typeof(ISemanticSegmenter), out var client);
        client.Matte = true;
        var page = await OpenAsync(services);
        await RunAsync(page);

        Assert.Equal("semantic", page.ResultMode);
        var png = page.MainPng!;
        Assert.Equal(6, png[25]);
        Assert.Contains("cut out", page.RowsHint, StringComparison.Ordinal);
        Assert.All(page.Rows, row => Assert.False(row.HasColor));
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

    private static async Task WaitAsync(VisionPageViewModel page)
    {
        await page.Queue.Items[^1].Completion.WaitAsync(TestContext.Current.CancellationToken);
        await page.Refresh.WaitAsync(TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task Segmentation_StartsInTheMiddle_ThenFollowsClicksAndBoxesOfOneEncodedImage()
    {
        using var services = new TestServices();
        Vision(services, ModelKinds.ImageSegmentation, typeof(IImageSegmenter), out var client);
        var page = await OpenAsync(services);
        Assert.Equal((VisionMode.Segment, "Segment"), (page.Mode, page.RunText));
        Assert.True(Visible(page, "segtool"));
        await RunAsync(page);

        Assert.Equal("segmentation", page.ResultMode);
        Assert.True(page.CanPrompt);
        Assert.Equal(new SegmentationPoint(40, 32), client.Prompts[0].Points.Single());
        Assert.Equal(["Mask 1", "Mask 2", "Mask 3"], page.Rows.Select(row => row.Label));
        var best = PngCodec.Decode(page.MainPng!);
        Assert.Equal((80, 64), (best.Width, best.Height));

        // A click adds a point on the object; the image is not encoded again.
        page.PlacePoint(0.25, 0.5, negative: false);
        await WaitAsync(page);
        Assert.Equal([new SegmentationPoint(40, 32), new SegmentationPoint(20, 32)], client.Prompts[1].Points);
        page.PlacePoint(0.9, 0.5, negative: true);
        await WaitAsync(page);
        Assert.False(client.Prompts[2].Points[2].IsPositive);
        Assert.Equal(1, client.Encodings);

        // A box replaces the points.
        page.PlaceBox(0.5, 0.75, 0.25, 0.25);
        await WaitAsync(page);
        Assert.Empty(client.Prompts[3].Points);
        Assert.Equal(new BoundingBox(20, 16, 40, 48), client.Prompts[3].Box);

        // The mask choice redraws without a new run.
        var before = PngCodec.Decode(page.MainPng!).Rgb;
        page.Panel.Parameters.OfType<ChoiceParameterViewModel>().Single(parameter => parameter.Key == "mask").Deserialize("3");
        await page.Refresh.WaitAsync(TestContext.Current.CancellationToken);
        Assert.NotEqual(before, PngCodec.Decode(page.MainPng!).Rgb);
        Assert.Equal(4, client.Prompts.Count);

        page.ClearPromptCommand.Execute(null);
        await WaitAsync(page);
        Assert.Equal(new SegmentationPoint(40, 32), client.Prompts[4].Points.Single());
        Assert.Null(client.Prompts[4].Box);
        Assert.Equal(["overlay.png", "result.json"], page.Gallery.Items[0].Files.Select(file => file.Name));
    }

    [Fact]
    public async Task Ocr_ShowsTheRecognizedText()
    {
        using var services = new TestServices();
        Vision(services, ModelKinds.Ocr, typeof(ITextRecognizer), out var client);
        var page = await OpenAsync(services);
        Assert.Equal((VisionMode.Ocr, "Read text"), (page.Mode, page.RunText));
        Number(page, "maxtokens").Value = 64;
        await RunAsync(page);

        Assert.True(page.ShowText);
        Assert.Equal("Hello Sharplight", page.ResultText);
        Assert.Equal(64, ((TextRecognitionOptions)client.Options[0]!).MaxTokens);
        await page.CopyCommand.ExecuteAsync(null);
        Assert.Equal("Hello Sharplight", services.Platform.Copied[^1]);
        Assert.Contains("Hello Sharplight", page.History.Items[0].Summary, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Upscaling_ShowsBeforeAndAfter_AndExportsTheEnlargedImage()
    {
        using var services = new TestServices();
        Vision(services, ModelKinds.ImageUpscaling, typeof(IImageUpscaler), out var client);
        var page = await OpenAsync(services);
        Assert.Equal((VisionMode.Upscale, "Upscale"), (page.Mode, page.RunText));
        Number(page, "tile").Value = 96;
        Number(page, "tilepad").Value = 8;
        await RunAsync(page);

        Assert.True(page.ShowAfter);
        var options = (ImageUpscalingOptions)client.Options[0]!;
        Assert.Equal((96, 8), (options.TileSize, options.TilePadding));
        var after = PngCodec.Decode(page.AfterPng!);
        Assert.Equal((160, 128), (after.Width, after.Height));
        Assert.Equal((80, 64), (PngCodec.Decode(page.MainPng!).Width, PngCodec.Decode(page.MainPng!).Height));
        string target = services.Directory + "/up.png";
        services.Platform.SaveTo = target;
        await page.ExportImageCommand.ExecuteAsync(null);
        Assert.Equal(160, PngCodec.Decode(await File.ReadAllBytesAsync(target, TestContext.Current.CancellationToken)).Width);
        Assert.Equal(["upscaled.png", "result.json"], page.Gallery.Items[0].Files.Select(file => file.Name));
    }
}
