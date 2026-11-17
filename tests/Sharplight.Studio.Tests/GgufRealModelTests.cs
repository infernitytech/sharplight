using System.Text;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Compatibility;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Image;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Studio.ViewModels.Speech;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The GGUF and whisper.cpp files people download from the Hub for image generation and speech recognition, through Studio's own code
/// paths: library recognition (<see cref="ModelInspector"/>), loading on the engine runtime and a run on the Image / Transcribe pages.
/// Each test skips when its file is not on disk. Diffusion transformers other than FLUX (SD3, ...) are recognized and named as not supported yet.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class GgufRealModelTests
{
    private static readonly string s_home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    private static readonly string s_jfk = Path.Combine(s_home, "models/audio/jfk.wav");

    private static LoadSettings Settings(EngineModelRuntime runtime) => new() { DeviceId = runtime.Devices[0].Id, ContextLength = 2048, BatchTokens = 256 };

    /// <summary>transcribe.cpp GGUF (handy-computer/whisper-*-gguf), and a whisper.cpp ggml file published with a .gguf extension (oxide-lab).</summary>
    public static TheoryData<string> WhisperFiles =>
    [
        "whisper-gguf/handy-computer/whisper-tiny-Q8_0.gguf",
        "whisper-gguf/handy-computer/whisper-tiny-F16.gguf",
        "whisper-gguf/handy-computer/whisper-base-Q4_K_M.gguf",
        "whisper-gguf/oxide-lab/whisper.cpp/whisper-large-v3-turbo-q5_0.gguf",
        "whisper.cpp/ggml-large-v3-turbo-q5_0.bin",
    ];

    [Theory]
    [MemberData(nameof(WhisperFiles))]
    public async Task TranscribePage_RunsWhisperFiles_RecognizedByTheLibrary(string file)
    {
        string path = Path.Combine(s_home, "models", file);
        Assert.SkipUnless(File.Exists(path) && File.Exists(s_jfk), $"{file} or jfk.wav not found");
        using var fakes = new TestServices();
        using var runtime = new EngineModelRuntime(EngineServices.From(TestHost.Services), () => new ResidencySettings());
        Assert.SkipWhen(runtime.Devices.Count == 0, "no compute device");
        var services = fakes.Services with { Runtime = runtime };
        var ct = TestContext.Current.CancellationToken;

        var inspected = ModelInspector.Inspect(path);
        Assert.Equal(ModelKinds.SpeechRecognition, inspected.Kind);
        var model = await runtime.LoadAsync(new ModelLoadRequest("whisper", path, Settings(runtime), [], Kind: inspected.Kind), null, ct);
        Assert.Equal(ModelKinds.SpeechRecognition, model.Kind);

        var page = new TranscribePageViewModel(services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        Assert.True(await page.SetAudioAsync(s_jfk));
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Single();
        await item.Completion.WaitAsync(TimeSpan.FromMinutes(5), ct);

        Assert.True(item.Status == RunStatus.Completed, item.Detail);
        Assert.Contains("ask not what your country can do for you", page.TranscriptText, StringComparison.OrdinalIgnoreCase);
        await runtime.UnloadAsync("whisper");
    }

    /// <summary>
    /// stable-diffusion.cpp GGUF files: SDXL-Turbo as Studio pulls it as <c>sdxl-turbo:q8_0</c> (OlegSkutte/sdxl-turbo-GGUF: both text
    /// encoders under sd.cpp's names, 512×512, the page switching to the distilled defaults of 4 steps without guidance), SD 1.5
    /// (second-state, the usual 20 steps with guidance 7.5).
    /// </summary>
    public static TheoryData<string, int, int> ImageFiles => new()
    {
        { "sd/gguf/sd_xl_turbo_1.0.q8_0.gguf", 512, 4 },
        { "sd/gguf/stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf", 512, 20 },
    };

    [Theory]
    [MemberData(nameof(ImageFiles))]
    public async Task ImagePage_GeneratesWithStableDiffusionCppGguf(string file, int size, int steps)
    {
        string path = Path.Combine(s_home, "models", file);
        Assert.SkipUnless(File.Exists(path), $"{file} not found");
        using var fakes = new TestServices();
        using var runtime = new EngineModelRuntime(EngineServices.From(TestHost.Services), () => new ResidencySettings());
        Assert.SkipWhen(runtime.Devices.Count == 0 || !runtime.Devices[0].IsGpu, "no GPU");
        var services = fakes.Services with { Runtime = runtime };
        var ct = TestContext.Current.CancellationToken;

        var inspected = ModelInspector.Inspect(path);
        Assert.Equal(ModelKinds.ImageGeneration, inspected.Kind);
        await runtime.LoadAsync(new ModelLoadRequest("sd", path, Settings(runtime), [], Kind: inspected.Kind), null, ct);

        var page = new ImagePageViewModel(services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        Assert.Equal("sd", page.SelectedModel?.Id);
        await page.ModelDefaults;
        Assert.Equal(steps, page.Panel.GetInt("steps"));
        Assert.Equal(steps == 4 ? 0 : 7.5, page.Panel.GetNumber("guidance"));
        page.Prompt = "a lighthouse on a cliff at sunset";
        page.Panel.Parameters.OfType<TextParameterViewModel>().Single(parameter => parameter.Key == "seed").Value = "7";
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Single();
        await item.Completion.WaitAsync(TimeSpan.FromMinutes(10), ct);

        Assert.True(item.Status == RunStatus.Completed, item.Detail);
        var decoded = PngCodec.Decode(Assert.Single(page.Images).Png);
        Assert.Equal((size, size), (decoded.Width, decoded.Height));
        Assert.True(Contrast(decoded.Rgb) > 0.05, "the image is flat");
        await runtime.UnloadAsync("sd");
    }

    [Fact]
    public async Task Sd3Gguf_IsListedAsAnImageModel_AndLoadingSaysNotSupportedYet()
    {
        using var dir = new TempDir();
        string city96 = dir.File("sd3.5_large-Q4_0.gguf");
        WriteGguf(city96, "sd3", "joint_blocks.0.x_block.attn.qkv.weight");
        string sdcpp = dir.File("sd3_medium-q8_0.gguf");
        WriteGguf(sdcpp, null, "model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight");
        using var fakes = new TestServices();
        using var runtime = new EngineModelRuntime(EngineServices.From(TestHost.Services), () => new ResidencySettings());
        Assert.SkipWhen(runtime.Devices.Count == 0, "no compute device");
        var ct = TestContext.Current.CancellationToken;

        foreach (string path in new[] { city96, sdcpp })
        {
            Assert.Equal(ModelKinds.ImageGeneration, ModelInspector.Inspect(path).Kind);
            var error = await Assert.ThrowsAnyAsync<Exception>(() =>
                runtime.LoadAsync(new ModelLoadRequest("flux", path, Settings(runtime), [], Kind: ModelKinds.ImageGeneration), null, ct));
            string shown = ImportFailure.Describe(error, error.GetBaseException().Message, path);
            Assert.StartsWith("Not supported yet: Stable Diffusion 3 (diffusion transformer)", shown, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void LibraryInference_RecognizesWhisperGgufFiles()
    {
        using var dir = new TempDir();
        string transcribeCpp = dir.File("whisper-small-Q5_K_M.gguf");
        WriteGguf(transcribeCpp, "whisper", "enc.conv.0.weight");
        string ggmlNamedGguf = dir.File("whisper-large-v3-turbo-q5_0.gguf");
        File.WriteAllBytes(ggmlNamedGguf, [0x6c, 0x6d, 0x67, 0x67, 1, 2, 3, 4]);
        string llm = dir.File("qwen3-0.6b-Q8_0.gguf");
        WriteGguf(llm, "qwen3", "token_embd.weight");

        Assert.Equal(ModelKinds.SpeechRecognition, ModelInspector.Inspect(transcribeCpp).Kind);
        Assert.Equal(ModelKinds.SpeechRecognition, ModelInspector.Inspect(ggmlNamedGguf).Kind);
        Assert.Equal(ModelKinds.TextGeneration, ModelInspector.Inspect(llm).Kind);
    }

    /// <summary>Standard deviation of the pixel values in [0, 1].</summary>
    private static double Contrast(ReadOnlySpan<byte> rgb)
    {
        double sum = 0;
        double squares = 0;
        foreach (byte value in rgb)
        {
            sum += value / 255.0;
            squares += value / 255.0 * (value / 255.0);
        }

        double mean = sum / rgb.Length;
        return Math.Sqrt(Math.Max(0, squares / rgb.Length - mean * mean));
    }

    /// <summary>A minimal GGUF v3 file: an optional <c>general.architecture</c> and one 4-element f32 tensor.</summary>
    private static void WriteGguf(string path, string? architecture, string tensor)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream, Encoding.UTF8);
        void String(string text)
        {
            var bytes = Encoding.UTF8.GetBytes(text);
            writer.Write((ulong)bytes.Length);
            writer.Write(bytes);
        }

        writer.Write("GGUF"u8);
        writer.Write(3u);
        writer.Write(1UL);
        writer.Write(architecture is null ? 0UL : 1UL);
        if (architecture is not null)
        {
            String("general.architecture");
            writer.Write(8u);
            String(architecture);
        }

        String(tensor);
        writer.Write(1u);
        writer.Write(4UL);
        writer.Write(0u);
        writer.Write(0UL);
        while (stream.Position % 32 != 0)
        {
            writer.Write((byte)0);
        }

        writer.Write(new byte[16]);
    }
}
