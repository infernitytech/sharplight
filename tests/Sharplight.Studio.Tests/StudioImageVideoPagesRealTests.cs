using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Studio.ViewModels.Image;
using Sharplight.Studio.ViewModels.Video;
using Sharplight.Video;
using Sharplight.Vision;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The Image and Video pages end to end on the tiny random pipelines (skipped when a model is missing), driven through the view models on
/// the real engine runtime: text to image with seed reuse and the gallery kept across page instances, image to image and inpainting with the
/// input and mask pictures, and the tiny Wan and LTX video models with the MP4 export of the native video layer.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class StudioImageVideoPagesRealTests
{
    private static ImageInput Gradient(int width, int height, int phase)
    {
        byte[] rgb = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int i = (y * width + x) * 3;
                rgb[i] = (byte)(x * 255 / width);
                rgb[i + 1] = (byte)(y * 255 / height);
                rgb[i + 2] = (byte)((x + y + phase) % 256);
            }
        }

        return new ImageInput(width, height, rgb);
    }

    private static ImageInput HalfMask(int width, int height)
    {
        byte[] rgb = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
        {
            for (int x = width / 2; x < width; x++)
            {
                int i = (y * width + x) * 3;
                rgb[i] = rgb[i + 1] = rgb[i + 2] = 255;
            }
        }

        return new ImageInput(width, height, rgb);
    }

    private static async Task<byte[]> GenerateAsync(ImagePageViewModel page)
    {
        int before = page.Queue.Items.Count;
        page.RunCommand.Execute(null);
        Assert.Equal(before + 1, page.Queue.Items.Count);
        await RealStudioSession.FinishAsync(page, 3);
        return page.Images[0].Png;
    }

    [Fact]
    public async Task Image_ReusingTheSeedReproducesThePicture_AndTheGalleryOutlivesThePage()
    {
        string sd = RealStudioSession.Require("sd/tiny-random-sd/diffusers");
        using var session = new RealStudioSession();
        await session.LoadAsync("tiny-sd", sd, ModelKinds.ImageGeneration, TestContext.Current.CancellationToken);

        var page = new ImagePageViewModel(session.Services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        page.Prompt = "a lighthouse at dusk";
        page.Panel.Apply(new Dictionary<string, string> { ["steps"] = "3", ["size"] = "128x128", ["seed"] = "" });
        byte[] first = await GenerateAsync(page);
        long seed = page.Images[0].Seed;
        Assert.InRange(seed, 0, long.MaxValue);
        Assert.Equal((128, 128), (PngCodec.Decode(first).Width, PngCodec.Decode(first).Height));

        // A random seed gives another picture; reusing the shown picture's seed gives the same one.
        byte[] second = await GenerateAsync(page);
        Assert.NotEqual(seed, page.Images[0].Seed);
        Assert.NotEqual(PngCodec.Decode(first).Rgb, PngCodec.Decode(second).Rgb);
        page.ReuseSeedCommand.Execute(null);
        Assert.Equal(page.Images[0].Seed.ToString(System.Globalization.CultureInfo.InvariantCulture), page.Panel.GetText("seed"));
        page.Panel.Apply(new Dictionary<string, string> { ["seed"] = seed.ToString(System.Globalization.CultureInfo.InvariantCulture) });
        byte[] third = await GenerateAsync(page);
        Assert.Equal(PngCodec.Decode(first).Rgb, PngCodec.Decode(third).Rgb);

        // The gallery lives on disk: a new page instance lists the three pictures, newest first, and takes one back.
        var reopened = new ImagePageViewModel(session.Services, new Fakes.FakeNavigator());
        await reopened.ActivateAsync();
        Assert.Equal(3, reopened.Gallery.Items.Count);
        reopened.Gallery.Items[0].ReuseCommand.Execute(null);
        Assert.Equal("a lighthouse at dusk", reopened.Prompt);
    }

    [Fact]
    public async Task Image_InputPictureMakesImageToImage_AndAMaskMakesInpainting()
    {
        string sd = RealStudioSession.Require("sd/tiny-random-sd/diffusers");
        using var session = new RealStudioSession();
        await session.LoadAsync("tiny-sd", sd, ModelKinds.ImageGeneration, TestContext.Current.CancellationToken);

        var page = new ImagePageViewModel(session.Services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        page.Prompt = "make it autumn";
        page.Panel.Apply(new Dictionary<string, string> { ["steps"] = "4", ["seed"] = "11" });
        byte[] text = await GenerateAsync(page);

        page.Source.Set(Gradient(128, 64, 0), "photo.png");
        Assert.True(page.UsesInputImage);
        page.Panel.Apply(new Dictionary<string, string> { ["strength"] = "0.6" });
        byte[] edited = await GenerateAsync(page);
        Assert.Contains("mode: image to image", page.Gallery.Items[0].Facts);
        var editedPicture = PngCodec.Decode(edited);
        Assert.Equal((128, 64), (editedPicture.Width, editedPicture.Height));
        Assert.NotEqual(PngCodec.Decode(text).Rgb, editedPicture.Rgb);

        page.Mask.Set(HalfMask(128, 64), "mask.png");
        byte[] inpainted = await GenerateAsync(page);
        Assert.Contains("mode: inpainting", page.Gallery.Items[0].Facts);
        var inpaintedPicture = PngCodec.Decode(inpainted);
        Assert.Equal((128, 64), (inpaintedPicture.Width, inpaintedPicture.Height));
        Assert.NotEqual(editedPicture.Rgb, inpaintedPicture.Rgb);

        // A generated picture goes back in as the next input.
        page.UseAsInputCommand.Execute(null);
        Assert.Equal(page.SelectedImage!.Image, page.Source.Image);
    }

    [Theory]
    [InlineData("video/tiny-random-wan/t2v/diffusers", "tiny-wan")]
    [InlineData("video/tiny-random-ltx/diffusers", "tiny-ltx")]
    public async Task Video_GeneratesFrames_AndExportsAnMp4(string relative, string id)
    {
        string path = RealStudioSession.Require(relative);
        using var session = new RealStudioSession();
        var ct = TestContext.Current.CancellationToken;
        await session.LoadAsync(id, path, ModelKinds.VideoGeneration, ct);

        var page = new VideoPageViewModel(session.Services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        Assert.Equal(id, page.SelectedModel?.Id);
        await page.ModelDefaults;
        await Task.Delay(300, ct);
        page.Prompt = "a red fox";
        page.Panel.Apply(new Dictionary<string, string> { ["width"] = "64", ["height"] = "64", ["frames"] = "9", ["steps"] = "2", ["seed"] = "3" });
        page.RunCommand.Execute(null);
        await RealStudioSession.FinishAsync(page, 10);
        await Task.Delay(200, ct);

        Assert.Equal(3, page.Result!.Seed);
        Assert.Equal(9, page.Result.Frames.Count);
        Assert.Equal(9, page.Player.Frames.Count);
        Assert.All(page.Result.Frames, frame => Assert.Equal((64, 64), (frame.Width, frame.Height)));
        Assert.NotEqual(page.Result.Frames[0].Rgb, page.Result.Frames[^1].Rgb);

        var mp4 = page.Export.Formats.Single(format => format.Format == VideoFormats.Mp4);
        Assert.SkipUnless(mp4.IsAvailable, mp4.Tip);
        string file = Path.Combine(session.Directory, "clip.mp4");
        await page.ExportToAsync(file);
        Assert.Equal("", page.Notice);
        byte[] head = new byte[12];
        await using (var stream = File.OpenRead(file))
        {
            await stream.ReadExactlyAsync(head, ct);
        }

        Assert.Equal("ftyp", System.Text.Encoding.ASCII.GetString(head, 4, 4));
        Assert.True(new FileInfo(file).Length > 500);
    }
}
