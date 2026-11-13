using System.CommandLine;
using Sharplight.Samples;

namespace Sharplight.OnnxRun;

/// <summary>Declares the command line of the ONNX runner and binds it to <see cref="OnnxRunSettings"/>.</summary>
internal sealed class OnnxRunCommand
{
    private const int DefaultRuns = 5;
    private const string ProviderCpu = "cpu";
    private const string ProviderVulkan = "vulkan";

    private readonly Argument<string> _model = new("model")
    {
        Description = "The ONNX model file, or a package (.slm) converted from one with sharplight-convert (its precompiled kernels are used).",
    };

    private readonly Option<string> _provider = new("--provider")
    {
        Description = "Execution provider.",
        DefaultValueFactory = _ => ProviderCpu,
    };

    private readonly Option<int> _device = OptionFactory.Integer("--device", "Device index within the provider.", 0, minimum: 0, helpName: "index");

    private readonly Option<int> _batch = OptionFactory.Integer("--batch", "Value of the symbolic batch dimension.", 1, helpName: "size");

    private readonly Option<int> _runs = OptionFactory.Integer("--runs", "Timed runs after one warm-up run.", DefaultRuns, helpName: "count");

    private readonly Option<string[]> _shapes = new("--shape")
    {
        HelpName = "NAME=DIMxDIM...",
        Description = "Full shape of an input with dynamic dimensions, e.g. pixel_values=-1x3x224x224; a dimension <= 0 is the batch axis. Repeatable.",
        DefaultValueFactory = _ => [],
    };

    private readonly Option<string?> _input = new("--input")
    {
        HelpName = "file",
        Description = "Raw float32 data for the first input (little endian, exactly the input's element count); other inputs get random data.",
    };

    private readonly Option<string?> _output = new("--out")
    {
        HelpName = "file",
        Description = "Store the first output as raw float32.",
    };

    private readonly Option<bool> _dumpGraph = new("--dump-graph")
    {
        Description = "Print the imported compute graph.",
    };

    public OnnxRunCommand()
    {
        _provider.AcceptOnlyFromAmong(ProviderCpu, ProviderVulkan);
        _shapes.Validators.Add(ValidateShapes);
    }

    /// <summary>Creates the root command; <paramref name="action"/> receives the validated settings.</summary>
    public RootCommand Build(Func<OnnxRunSettings, CancellationToken, Task<int>> action)
    {
        var command = new RootCommand("Imports an ONNX model, runs it on the CPU or Vulkan provider with random or file input and prints timings and the top-5 of the first output.");
        command.Arguments.Add(_model);
        command.Options.Add(_provider);
        command.Options.Add(_device);
        command.Options.Add(_batch);
        command.Options.Add(_runs);
        command.Options.Add(_shapes);
        command.Options.Add(_input);
        command.Options.Add(_output);
        command.Options.Add(_dumpGraph);
        command.SetAction(CommandRunner.Guard((parseResult, cancellationToken) => action(Read(parseResult), cancellationToken)));
        return command;
    }

    private void ValidateShapes(System.CommandLine.Parsing.OptionResult result)
    {
        foreach (var text in result.GetValueOrDefault<string[]>() ?? [])
        {
            if (!InputShapeParser.TryParse(text, out _, out _, out var error))
            {
                result.AddError(error!);
            }
        }
    }

    private OnnxRunSettings Read(ParseResult parseResult)
    {
        var shapes = new Dictionary<string, long[]>(StringComparer.Ordinal);
        foreach (var text in parseResult.GetValue(_shapes) ?? [])
        {
            InputShapeParser.TryParse(text, out var name, out var dimensions, out _);
            shapes[name] = dimensions;
        }

        var provider = parseResult.GetValue(_provider) == ProviderVulkan ? OnnxProvider.Vulkan : OnnxProvider.Cpu;
        return new OnnxRunSettings(
            parseResult.GetValue(_model)!,
            provider,
            parseResult.GetValue(_device),
            parseResult.GetValue(_batch),
            parseResult.GetValue(_runs),
            shapes,
            parseResult.GetValue(_input),
            parseResult.GetValue(_output),
            parseResult.GetValue(_dumpGraph));
    }
}
