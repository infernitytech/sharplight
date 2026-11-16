using Sharplight.Imaging;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Vision.Tests;

/// <summary>
/// Image upscalers through the task layer, against the reference implementation (<c>tools/upscale-reference.py</c> writes the input and the 8-bit output next to
/// the checkpoint): the whole-image output must be within 40 dB PSNR of it, and the tiled output (tiles that cover the image several times over, blended)
/// close to the whole-image one. Skipped when the checkpoint or its reference is not present.
/// </summary>
public class ImageUpscalerTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    private static string Model(string name) => Path.Combine(Home, "models/vision", name);

    private static bool Available(string name) => File.Exists(Path.Combine(Model(name), "sharplight-reference.png")) && Directory.Exists(Model(name));

    [Fact]
    public async Task UpscalerMatchesTheReference()
    {
        const string Name = "realesrgan-x4plus";
        if (!Available(Name))
        {
            return;
        }

        var input = PngCodec.Read(Path.Combine(Model(Name), "sharplight-sample.png"));
        var expected = PngCodec.Read(Path.Combine(Model(Name), "sharplight-reference.png"));
        var backends = new List<(string, DeviceSlot)>();
        if (Env.Vulkan() is { } vulkan)
        {
            backends.Add(("vulkan", vulkan));
        }

        backends.Add(("cpu", Env.Cpu()));
        foreach (var (backend, slot) in backends)
        {
            using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = Path.Combine(Model(Name), "RealESRGAN_x4plus.pth"), Devices = [slot] });
            Assert.Equal("image-upscaling", model.Description.Kind);
            using var upscaler = new TaskClientProvider([.. Env.TaskFactories()]).Create<IImageUpscaler>(model);
            Assert.Equal(4, upscaler.Scale);
            var whole = await upscaler.UpscaleAsync(input, new UpscaleOptions { TileSize = 0 }, TestContext.Current.CancellationToken);
            Assert.Equal((expected.Width, expected.Height), (whole.Width, whole.Height));
            double psnr = Psnr(whole, expected);
            TestContext.Current.SendDiagnosticMessage($"{backend}: whole image {psnr:f1} dB from the reference");
            Assert.True(psnr > 40, $"{backend}: {psnr} dB");
            if (backend == "cpu")
            {
                continue;
            }

            var progress = new List<double>();
            var tiled = await upscaler.UpscaleAsync(input, new UpscaleOptions { TileSize = 64, TilePadding = 16, Progress = new Progress<double>(progress.Add) }, TestContext.Current.CancellationToken);
            double tiledPsnr = Psnr(tiled, expected);
            TestContext.Current.SendDiagnosticMessage($"{backend}: tiled {tiledPsnr:f1} dB from the reference, {progress.Count} tiles");
            Assert.True(tiledPsnr > 40, $"{backend}: tiled {tiledPsnr} dB");
            Assert.True(progress.Count > 1);
        }
    }

    private static double Psnr(ImageInput a, ImageInput b)
    {
        double sum = 0;
        for (int i = 0; i < a.Rgb.Length; i++)
        {
            double d = a.Rgb[i] - b.Rgb[i];
            sum += d * d;
        }

        double mse = sum / a.Rgb.Length;
        return mse == 0 ? 99 : 10 * Math.Log10(255.0 * 255.0 / mse);
    }
}
