using System.CommandLine;
using Sharplight.Samples;
using Sharplight.VideoEnhancement;

namespace Sharplight.EnhanceVideo;

/// <summary>The <c>sharplight-enhance-video</c> command line.</summary>
internal sealed class EnhanceVideoCommand
{
    private readonly Option<string> _model = new("--model", "-m") { Description = "Model: a SeedVR2 folder (seedvr2_ema_3b.pth or seedvr2_ema_7b.pth with ema_vae.pth and pos_emb.pt) or an .slm package.", Required = true };
    private readonly Option<string> _input = new("--input", "-i") { Description = "Input: a video file (decoded with the bundled FFmpeg) or a PNG image.", Required = true };
    private readonly Option<string> _output = new("--output", "-o") { Description = "Output: a video file written as the batches finish, with the input's sound (the extension picks the format: .mp4, .webm, .mkv, .mov), a .png for an image input, or a folder of PNG frames (no extension).", DefaultValueFactory = _ => "enhanced.mp4" };
    private readonly Option<int?> _width = new("--width", "-W") { Description = "Output width in pixels (with only one of --width / --height the other keeps the aspect ratio)." };
    private readonly Option<int?> _height = new("--height", "-H") { Description = "Output height in pixels." };
    private readonly Option<float?> _scale = new("--scale", "-s") { Description = "Enlargement when no size is given (default: the model's, 2)." };
    private readonly Option<int?> _batch = new("--batch", "-b") { Description = "Frames restored together (1 + 4k; default: the whole clip, long clips in the model's longest batches). Smaller batches need less memory." };
    private readonly Option<VideoColorCorrection?> _color = new("--color-fix") { Description = "Colour correction against the input: Wavelet (default), AdaIn or None." };
    private readonly Option<long?> _seed = new("--seed") { Description = "Seed of the noise (default: random)." };
    private readonly Option<double> _start = new("--start") { Description = "Start of the part of the video to enhance, in seconds." };
    private readonly Option<int?> _maxFrames = new("--frames", "-f") { Description = "Frames to enhance from --start (default: all)." };
    private readonly Option<bool> _cpu = new("--cpu") { Description = "Run on the CPU instead of the GPU." };
    private readonly Option<int> _device = new("--device") { Description = "Device index of the provider." };
    private readonly Option<bool> _reducedPrecision = new("--reduced-precision") { Description = "Opt-in: f16 products where the default keeps near-f32 accuracy." };

    public Command Build()
    {
        var command = new RootCommand("Restores and upscales videos and images with a Sharplight video-enhancement model (SeedVR2) through the DI task API.");
        foreach (var option in new Option[] { _model, _input, _output, _width, _height, _scale, _batch, _color, _seed, _start, _maxFrames, _cpu, _device, _reducedPrecision })
        {
            command.Options.Add(option);
        }

        command.SetAction(CommandRunner.Guard((parseResult, cancellationToken) => new VideoEnhancementRunner(Settings(parseResult)).RunAsync(cancellationToken)));
        return command;
    }

    private EnhanceVideoSettings Settings(ParseResult result) => new()
    {
        Model = result.GetValue(_model)!,
        Input = result.GetValue(_input)!,
        Output = result.GetValue(_output)!,
        Width = result.GetValue(_width),
        Height = result.GetValue(_height),
        Scale = result.GetValue(_scale),
        FramesPerBatch = result.GetValue(_batch),
        ColorCorrection = result.GetValue(_color),
        Seed = result.GetValue(_seed),
        Start = result.GetValue(_start),
        MaxFrames = result.GetValue(_maxFrames),
        Cpu = result.GetValue(_cpu),
        Device = result.GetValue(_device),
        ReducedPrecision = result.GetValue(_reducedPrecision),
    };
}
