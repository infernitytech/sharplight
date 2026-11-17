using System.CommandLine;
using Sharplight.ImageGeneration;
using Sharplight.Samples;

namespace Sharplight.GenerateVideo;

/// <summary>The <c>sharplight-generate-video</c> command line.</summary>
internal sealed class GenerateVideoCommand
{
    private readonly Option<string> _model = new("--model", "-m") { Description = "Model: a diffusers folder (Wan 2.1 T2V, Wan 2.2 TI2V), a transformer-only Wan .gguf / .safetensors (with --companions) or an .slm package.", Required = true };
    private readonly Option<string?> _companions = new("--companions") { Description = "For a transformer-only checkpoint: a Wan diffusers folder of the same family that provides the text encoder, tokenizer and VAE." };
    private readonly Option<string?> _transformer2 = new("--transformer-2") { Description = "For a two-expert model given as transformer-only files (Wan 2.2 A14B GGUF): the low-noise expert; --model is the high-noise one." };
    private readonly Option<string?> _imageEncoder = new("--image-encoder") { Description = "For a transformer-only Wan 2.1 I2V / FLF2V checkpoint: a diffusers folder with its image_encoder/ and image_processor/ (the configuration folder suffices)." };
    private readonly Option<string> _prompt = new("--prompt", "-p") { Description = "What to generate.", Required = true };
    private readonly Option<string?> _negative = new("--negative-prompt", "-n") { Description = "What the video should not show (default: the model's)." };
    private readonly Option<string?> _image = new("--image", "-i") { Description = "PNG start frame for image-to-video (models that support it, e.g. Wan 2.2 TI2V)." };
    private readonly Option<string?> _lastImage = new("--last-image") { Description = "PNG last frame for first-and-last-frame generation (Wan 2.1 FLF2V; Wan 2.2 I2V-A14B optionally)." };
    private readonly Option<string> _output = new("--output", "-o") { Description = "Output: a video file whose extension picks the encoder (.webm, .gif, .apng / .png, .mp4 with a system ffmpeg), or a folder of PNG frames (no extension).", DefaultValueFactory = _ => "video.webm" };
    private readonly Option<int?> _width = new("--width", "-W") { Description = "Width in pixels (default: the model's)." };
    private readonly Option<int?> _height = new("--height", "-H") { Description = "Height in pixels (default: the model's)." };
    private readonly Option<int?> _frames = new("--frames", "-f") { Description = "Frame count, 1 + 4k for Wan (default: the model's)." };
    private readonly Option<double?> _fps = new("--fps") { Description = "Playback rate of the written video (default: the model's)." };
    private readonly Option<int?> _steps = new("--steps") { Description = "Denoising steps (default: the model's)." };
    private readonly Option<float?> _guidance = new("--guidance", "--cfg-scale") { Description = "Classifier-free guidance scale (default: the model's)." };
    private readonly Option<float?> _guidance2 = new("--guidance-2") { Description = "Guidance scale of the low-noise expert of two-expert models (default: the model's)." };
    private readonly Option<long?> _seed = new("--seed", "-s") { Description = "Seed (default: random)." };
    private readonly Option<string?> _sampler = new("--sampler") { Description = $"Sampler: {string.Join(", ", Samplers.Flow)} (default: the model's)." };
    private readonly Option<string?> _noise = new("--noise") { Description = "Raw little-endian f32 file with the initial latents [C, T, h, w] (reproduces another implementation's run)." };
    private readonly Option<bool> _cpu = new("--cpu") { Description = "Run on the CPU instead of the GPU." };
    private readonly Option<int> _device = new("--device") { Description = "Device index of the provider." };
    private readonly Option<bool> _reducedPrecision = new("--reduced-precision") { Description = "Opt-in: f16 products where the default keeps near-f32 accuracy." };

    public Command Build()
    {
        var command = new RootCommand("Generates videos with a Sharplight video-generation model (Wan) through the DI task API.");
        foreach (var option in new Option[] { _model, _companions, _transformer2, _imageEncoder, _prompt, _negative, _image, _lastImage, _output, _width, _height, _frames, _fps, _steps, _guidance, _guidance2, _seed, _sampler, _noise, _cpu, _device, _reducedPrecision })
        {
            command.Options.Add(option);
        }

        command.SetAction(CommandRunner.Guard((parseResult, cancellationToken) => new VideoGenerationRunner(Settings(parseResult)).RunAsync(cancellationToken)));
        return command;
    }

    private GenerateVideoSettings Settings(ParseResult result) => new()
    {
        Model = result.GetValue(_model)!,
        Companions = result.GetValue(_companions),
        SecondTransformer = result.GetValue(_transformer2),
        ImageEncoder = result.GetValue(_imageEncoder),
        Guidance2 = result.GetValue(_guidance2),
        Prompt = result.GetValue(_prompt)!,
        NegativePrompt = result.GetValue(_negative),
        Image = result.GetValue(_image),
        LastImage = result.GetValue(_lastImage),
        Output = result.GetValue(_output)!,
        Width = result.GetValue(_width),
        Height = result.GetValue(_height),
        Frames = result.GetValue(_frames),
        FramesPerSecond = result.GetValue(_fps),
        Steps = result.GetValue(_steps),
        Guidance = result.GetValue(_guidance),
        Seed = result.GetValue(_seed),
        Sampler = result.GetValue(_sampler),
        Noise = result.GetValue(_noise),
        Cpu = result.GetValue(_cpu),
        Device = result.GetValue(_device),
        ReducedPrecision = result.GetValue(_reducedPrecision),
    };
}
