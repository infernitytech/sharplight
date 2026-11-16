using System.CommandLine;
using Sharplight.Samples;

namespace Sharplight.Vision.Sample;

/// <summary>The command line of <c>sharplight-vision</c>: the sub-commands <c>classify</c>, <c>detect</c>, <c>depth</c>, <c>segment</c>, <c>ocr</c> and <c>upscale</c>.</summary>
public sealed class VisionOptionSet
{
    /// <summary>The model argument.</summary>
    public Argument<string> Model { get; } = new("model") { Description = "Vision model: .slm package or Hugging Face directory." };

    /// <summary>The image argument.</summary>
    public Argument<string> Image { get; } = new("image") { Description = "PNG image." };

    /// <summary><c>--output</c>.</summary>
    public Option<string?> Output { get; } = new("--output", "-o") { Description = "Write the annotated image to this PNG file.", HelpName = "file" };

    /// <summary><c>--cpu</c>.</summary>
    public Option<bool> Cpu { get; } = new("--cpu") { Description = "Run on the CPU provider instead of a GPU." };

    /// <summary><c>--device</c>.</summary>
    public Option<int> Device { get; } = OptionFactory.Integer("--device", "Index of the device to use (see sharplight --devices).", 0, minimum: 0, helpName: "index");

    /// <summary><c>--threshold</c>.</summary>
    public Option<float> Threshold { get; } = new("--threshold", "-t") { Description = "Detections scoring below this are dropped.", DefaultValueFactory = _ => 0.5f, HelpName = "score" };

    /// <summary><c>--top</c>.</summary>
    public Option<int> Top { get; } = OptionFactory.Integer("--top", "Number of classes to print.", 5, minimum: 1, helpName: "count");

    /// <summary><c>--label</c>.</summary>
    public Option<string[]> Label { get; } = new("--label") { Description = "Zero-shot class prompt (repeatable; for CLIP or SigLIP models).", HelpName = "prompt", AllowMultipleArgumentsPerToken = false };

    /// <summary><c>--point</c>.</summary>
    public Option<string[]> Point { get; } = new("--point") { Description = "Segmentation: a point on the object, x,y in pixels (repeatable).", HelpName = "x,y", AllowMultipleArgumentsPerToken = false };

    /// <summary><c>--not</c>.</summary>
    public Option<string[]> Negative { get; } = new("--not") { Description = "Segmentation: a point beside the object, x,y in pixels (repeatable).", HelpName = "x,y", AllowMultipleArgumentsPerToken = false };

    /// <summary><c>--box</c>.</summary>
    public Option<string?> Box { get; } = new("--box") { Description = "Segmentation: a box around the object, x1,y1,x2,y2 in pixels.", HelpName = "x1,y1,x2,y2" };

    /// <summary><c>--tile</c>.</summary>
    public Option<int> Tile { get; } = OptionFactory.Integer("--tile", "Upscaling: tile side in input pixels (0: the whole image at once).", 192, minimum: 0, helpName: "pixels");

    /// <summary><c>--tile-padding</c>.</summary>
    public Option<int> TilePadding { get; } = OptionFactory.Integer("--tile-padding", "Upscaling: context shared by neighbouring tiles, in pixels.", 16, minimum: 0, helpName: "pixels");

    /// <summary><c>--verbose</c>.</summary>
    public Option<bool> Verbose { get; } = new("--verbose", "-v") { Description = "Print load and compile messages to standard error." };

    /// <summary>Builds the root command.</summary>
    /// <param name="classify">Runs <c>classify</c>.</param>
    /// <param name="detect">Runs <c>detect</c>.</param>
    /// <param name="depth">Runs <c>depth</c>.</param>
    /// <param name="segment">Runs <c>segment</c>.</param>
    /// <param name="ocr">Runs <c>ocr</c>.</param>
    /// <param name="upscale">Runs <c>upscale</c>.</param>
    public RootCommand BuildCommand(Func<VisionSettings, CancellationToken, Task<int>> classify, Func<VisionSettings, CancellationToken, Task<int>> detect,
        Func<VisionSettings, CancellationToken, Task<int>> depth, Func<VisionSettings, CancellationToken, Task<int>> segment, Func<VisionSettings, CancellationToken, Task<int>> ocr,
        Func<VisionSettings, CancellationToken, Task<int>> upscale)
    {
        var root = new RootCommand("Image classification, detection, depth, segmentation, text recognition and upscaling with Sharplight vision models.");
        root.Subcommands.Add(Create("classify", "Classifies an image (ViT, DINOv2 heads) or scores it against --label prompts (CLIP, SigLIP).", classify));
        root.Subcommands.Add(Create("detect", "Detects objects (DETR, RT-DETR) and draws the boxes into --output.", detect));
        root.Subcommands.Add(Create("depth", "Estimates a depth map (Depth Anything) and writes it as a grayscale PNG to --output (near is bright).", depth));
        root.Subcommands.Add(Create("segment", "Segments the object at --point / --box (SAM 2) and draws the mask into --output.", segment));
        root.Subcommands.Add(Create("ocr", "Reads the text of an image (TrOCR).", ocr));
        root.Subcommands.Add(Create("upscale", "Enlarges an image (Real-ESRGAN .pth or .slm) into --output, tiled and blended for large images.", upscale));
        return root;
    }

    private Command Create(string name, string description, Func<VisionSettings, CancellationToken, Task<int>> run)
    {
        var command = new Command(name, description);
        command.Arguments.Add(Model);
        command.Arguments.Add(Image);
        foreach (Option option in new Option[] { Output, Cpu, Device, Threshold, Top, Label, Point, Negative, Box, Tile, TilePadding, Verbose })
        {
            command.Options.Add(option);
        }

        command.SetAction(CommandRunner.Guard((parseResult, cancellationToken) => run(Read(parseResult), cancellationToken)));
        return command;
    }

    private VisionSettings Read(ParseResult parseResult)
        => new(parseResult.GetValue(Model)!, parseResult.GetValue(Image)!, parseResult.GetValue(Output), parseResult.GetValue(Cpu), parseResult.GetValue(Device),
            parseResult.GetValue(Threshold), parseResult.GetValue(Top), parseResult.GetValue(Label) ?? [], parseResult.GetValue(Verbose), parseResult.GetValue(Point) ?? [],
            parseResult.GetValue(Negative) ?? [], parseResult.GetValue(Box), parseResult.GetValue(Tile), parseResult.GetValue(TilePadding));
}
