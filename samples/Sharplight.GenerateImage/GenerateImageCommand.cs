using System.CommandLine;
using Sharplight.ImageGeneration;
using Sharplight.Samples;

namespace Sharplight.GenerateImage;

/// <summary>The <c>sharplight-generate-image</c> command line.</summary>
internal sealed class GenerateImageCommand
{
    private readonly Option<string> _model = new("--model", "-m") { Description = "Model: a diffusers folder, a single-file .safetensors / .gguf checkpoint or an .slm package.", Required = true };
    private readonly Option<string> _prompt = new("--prompt", "-p") { Description = "What to generate.", Required = true };
    private readonly Option<string?> _negative = new("--negative-prompt", "-n") { Description = "What the image should not show." };
    private readonly Option<string> _output = new("--output", "-o") { Description = "Output PNG; with --count > 1, '-<index>' is inserted before the extension.", DefaultValueFactory = _ => "output.png" };
    private readonly Option<int?> _width = new("--width", "-W") { Description = "Width in pixels (default: the model's native size)." };
    private readonly Option<int?> _height = new("--height", "-H") { Description = "Height in pixels (default: the model's native size)." };
    private readonly Option<int?> _steps = new("--steps") { Description = "Denoising steps (default: the model's)." };
    private readonly Option<float?> _guidance = new("--guidance", "--cfg-scale") { Description = "Classifier-free guidance scale (default: the model's)." };
    private readonly Option<long?> _seed = new("--seed", "-s") { Description = "Seed; image i uses seed + i (default: random)." };
    private readonly Option<int> _count = new("--count") { Description = "Number of images.", DefaultValueFactory = _ => 1 };
    private readonly Option<string?> _sampler = new("--sampler") { Description = $"Sampler: {string.Join(", ", Samplers.All)} (default: the model's)." };
    private readonly Option<string?> _initImage = new("--init-image") { Description = "PNG input for image-to-image." };
    private readonly Option<string?> _mask = new("--mask") { Description = "PNG mask for inpainting (white = regenerate); needs --init-image." };
    private readonly Option<float> _strength = new("--strength") { Description = "Image-to-image strength in [0, 1].", DefaultValueFactory = _ => 0.75f };
    private readonly Option<string?> _noise = new("--noise") { Description = "Raw little-endian f32 file with the initial noise (reproduces another implementation's run)." };
    private readonly Option<int> _preview = new("--preview-every") { Description = "Write a latent preview PNG every N steps (0: none)." };
    private readonly Option<bool> _cpu = new("--cpu") { Description = "Run on the CPU instead of the GPU." };
    private readonly Option<int> _device = new("--device") { Description = "Device index of the provider." };

    public Command Build()
    {
        var command = new RootCommand("Generates images with a Sharplight image-generation model (Stable Diffusion 1.x / 2.x) through the DI task API.");
        foreach (var option in new Option[] { _model, _prompt, _negative, _output, _width, _height, _steps, _guidance, _seed, _count, _sampler, _initImage, _mask, _strength, _noise, _preview, _cpu, _device })
        {
            command.Options.Add(option);
        }

        command.SetAction(CommandRunner.Guard((parseResult, cancellationToken) => new ImageGenerationRunner(Settings(parseResult)).RunAsync(cancellationToken)));
        return command;
    }

    private GenerateImageSettings Settings(ParseResult result) => new()
    {
        Model = result.GetValue(_model)!,
        Prompt = result.GetValue(_prompt)!,
        NegativePrompt = result.GetValue(_negative),
        Output = result.GetValue(_output)!,
        Width = result.GetValue(_width),
        Height = result.GetValue(_height),
        Steps = result.GetValue(_steps),
        Guidance = result.GetValue(_guidance),
        Seed = result.GetValue(_seed),
        Count = result.GetValue(_count),
        Sampler = result.GetValue(_sampler),
        InitImage = result.GetValue(_initImage),
        Mask = result.GetValue(_mask),
        Strength = result.GetValue(_strength),
        Noise = result.GetValue(_noise),
        PreviewInterval = result.GetValue(_preview),
        Cpu = result.GetValue(_cpu),
        Device = result.GetValue(_device),
    };
}
