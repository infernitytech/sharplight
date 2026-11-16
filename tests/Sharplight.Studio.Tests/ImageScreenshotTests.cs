using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.VisualTree;
using Avalonia.Logging;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using Sharplight.ImageGeneration;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Platform;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Studio.Views;
using Sharplight.Vision;

namespace Sharplight.Studio.Tests;

/// <summary>
/// Renders the Image and Vision pages with their shared blocks and checks that the views bind cleanly. Set
/// <c>SHARPLIGHT_STUDIO_SCREENSHOTS</c> to a directory to also save each page as a PNG.
/// </summary>
public sealed class ImageScreenshotTests
{
    private static readonly string? s_screenshots = Environment.GetEnvironmentVariable("SHARPLIGHT_STUDIO_SCREENSHOTS");

    [Fact]
    public Task ImageAndVisionPages_RenderWithoutBindingErrors() => HeadlessApp.RunAsync(async () =>
    {
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
        var generator = new FakeImageGenerator { NativeSize = 384 };
        var sd = new FakeModel("sd-1.5-q8") { Kind = ModelKinds.ImageGeneration };
        sd.Clients[typeof(IImageGenerator)] = generator;
        var vision = new FakeVision();
        var detr = new FakeModel("detr-resnet-50") { Kind = ModelKinds.ObjectDetection };
        detr.Clients[typeof(IObjectDetector)] = vision;
        var depth = new FakeModel("depth-anything-v2-small") { Kind = ModelKinds.DepthEstimation };
        depth.Clients[typeof(IDepthEstimator)] = vision;
        var vit = new FakeModel("vit-base-patch16-224") { Kind = ModelKinds.ImageClassification };
        vit.Clients[typeof(IImageClassifier)] = vision;
        var clip = new FakeModel("clip-vit-base-patch32") { Kind = ModelKinds.ImageEmbedding };
        clip.Clients[typeof(IImageEmbedder)] = vision;
        var sam = new FakeModel("sam2.1-hiera-tiny") { Kind = ModelKinds.ImageSegmentation };
        sam.Clients[typeof(IImageSegmenter)] = vision;
        var trocr = new FakeModel("trocr-base-printed") { Kind = ModelKinds.Ocr };
        trocr.Clients[typeof(ITextRecognizer)] = vision;
        var esrgan = new FakeModel("realesrgan-x4plus") { Kind = ModelKinds.ImageUpscaling };
        esrgan.Clients[typeof(IImageUpscaler)] = vision;
        string photo = services.Directory + "/street.png";
        PngCodec.Write(photo, Scene(480, 360));
        new ThemeService(Application.Current!).Apply(AppTheme.Dark, "Violet");
        var viewModel = new MainWindowViewModel(services.Services);
        var window = new MainWindow { DataContext = viewModel, Width = 1440, Height = 900 };
        window.Show();
        try
        {
            await viewModel.NavigateAsync(viewModel.Image);
            await HeadlessApp.RenderAsync(window);
            Assert.False(viewModel.Image.HasModels);
            Save(window, "image-empty");

            services.Runtime.Add(sd);
            viewModel.OpenKind(ModelKinds.ImageGeneration, sd);
            viewModel.Image.Prompt = "a lighthouse on a cliff at dusk, oil painting, dramatic clouds";
            viewModel.Image.NegativePrompt = "blurry, low quality";
            viewModel.Image.Panel.Parameters.OfType<NumberParameterViewModel>().Single(parameter => parameter.Key == "count").Value = 3;
            viewModel.Image.RunCommand.Execute(null);
            await viewModel.Image.Queue.Items.Single().Completion;
            await HeadlessApp.WaitForAsync(window, () => viewModel.Image.HasResult);
            await HeadlessApp.RenderAsync(window);
            Assert.Equal(3, viewModel.Image.Images.Count);
            // The gallery strip below the result loads its thumbnails on worker threads.
            await HeadlessApp.WaitForAsync(window, () => viewModel.Image.Generations.Visible.Count == 3 && viewModel.Image.Generations.Visible.All(item => item.Thumbnail is not null));
            await HeadlessApp.RenderAsync(window);
            Save(window, "image");
            // Scrolled down to the gallery of stored images, with one of them selected.
            viewModel.Image.Generations.SelectedItem = viewModel.Image.Generations.Visible[1];
            await HeadlessApp.RenderAsync(window);
            foreach (var scroller in window.GetVisualDescendants().OfType<ScrollViewer>().Where(scroller => scroller.Extent.Height > scroller.Viewport.Height + 1 && scroller.Viewport.Width > 600))
            {
                scroller.ScrollToEnd();
            }

            await HeadlessApp.RenderAsync(window);
            Save(window, "image-gallery");
            viewModel.Image.SideTab = 2;
            await HeadlessApp.RenderAsync(window);
            Save(window, "image-outputs");
            viewModel.Image.SideTab = 0;

            await viewModel.Image.Source.SetFileAsync(photo);
            await HeadlessApp.RenderAsync(window);
            Save(window, "image-img2img");

            // Vision: detection first, then the other modes.
            services.Runtime.Add(detr);
            services.Runtime.Add(depth);
            services.Runtime.Add(vit);
            services.Runtime.Add(clip);
            viewModel.OpenKind(ModelKinds.ObjectDetection, detr);
            await viewModel.Vision.Source.SetFileAsync(photo);
            viewModel.Vision.RunCommand.Execute(null);
            await viewModel.Vision.Queue.Items.Single().Completion;
            await viewModel.Vision.Refresh;
            await HeadlessApp.RenderAsync(window);
            Save(window, "vision-detection");

            viewModel.Vision.SelectModel(depth);
            viewModel.Vision.RunCommand.Execute(null);
            await viewModel.Vision.Queue.Items[^1].Completion;
            await viewModel.Vision.Refresh;
            await HeadlessApp.RenderAsync(window);
            Save(window, "vision-depth");

            viewModel.Vision.SelectModel(clip);
            viewModel.Vision.Candidates = "a street, a beach, a forest";
            viewModel.Vision.RunCommand.Execute(null);
            await viewModel.Vision.Queue.Items[^1].Completion;
            await viewModel.Vision.Refresh;
            await HeadlessApp.RenderAsync(window);
            Save(window, "vision-zero-shot");

            services.Runtime.Add(sam);
            services.Runtime.Add(trocr);
            services.Runtime.Add(esrgan);
            viewModel.Vision.SelectModel(sam);
            viewModel.Vision.RunCommand.Execute(null);
            await viewModel.Vision.Queue.Items[^1].Completion;
            await viewModel.Vision.Refresh;
            await HeadlessApp.RenderAsync(window);
            Save(window, "vision-segmentation");

            viewModel.Vision.SelectModel(trocr);
            viewModel.Vision.RunCommand.Execute(null);
            await viewModel.Vision.Queue.Items[^1].Completion;
            await viewModel.Vision.Refresh;
            await HeadlessApp.RenderAsync(window);
            Save(window, "vision-ocr");

            viewModel.Vision.SelectModel(esrgan);
            viewModel.Vision.RunCommand.Execute(null);
            await viewModel.Vision.Queue.Items[^1].Completion;
            await viewModel.Vision.Refresh;
            await HeadlessApp.RenderAsync(window);
            Save(window, "vision-upscale");

            viewModel.Vision.SideTab = 2;
            await HeadlessApp.RenderAsync(window);
            Save(window, "vision-outputs");
            viewModel.Vision.SideTab = 0;

            await viewModel.NavigateAsync(viewModel.Tasks);
            await HeadlessApp.RenderAsync(window);
            Assert.Contains(viewModel.Tasks.Kinds, card => card.Info.Id == ModelKinds.ImageGeneration && card.IsRunnable && card.OpenText == "Open image");
            Save(window, "tasks-runnable");
        }
        finally
        {
            window.Close();
            Dispatcher.UIThread.RunJobs();
            Logger.Sink = null;
        }

        Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages.Distinct()));
    });

    private static ImageInput Scene(int width, int height)
    {
        var rgb = new byte[3 * width * height];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int i = 3 * (y * width + x);
                bool sky = y < height / 2;
                rgb[i] = (byte)(sky ? 90 + x * 80 / width : 70);
                rgb[i + 1] = (byte)(sky ? 130 + y * 60 / height : 110 + x * 40 / width);
                rgb[i + 2] = (byte)(sky ? 200 : 80);
            }
        }

        return new ImageInput(width, height, rgb);
    }

    private static void Save(MainWindow window, string name)
    {
        using var frame = window.CaptureRenderedFrame();
        Assert.NotNull(frame);
        if (s_screenshots is null)
        {
            return;
        }

        Directory.CreateDirectory(s_screenshots);
        frame.Save(Path.Combine(s_screenshots, $"studio-{name}.png"), new PngBitmapEncoderOptions());
    }
}
