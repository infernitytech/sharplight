using System.CommandLine;
using System.CommandLine.Parsing;
using Sharplight.Samples;
using Sharplight.Tensors;

namespace Sharplight.Convert;

/// <summary><c>convert &lt;src&gt; &lt;dst.slm&gt;</c>: builds a model from any supported source and writes it as a package.</summary>
internal sealed class ConvertCommand
{
    private readonly Argument<string> _source = new("src")
    {
        Description = "Source model: a .gguf file, a Hugging Face checkpoint directory, an .onnx file or an .slm package.",
    };

    private readonly Argument<string> _destination = new("dst") { Description = "The package to write (.slm); replaced atomically." };

    private readonly Option<bool> _noOptimize = new("--no-optimize")
    {
        Description = "Store the graphs as built, without the device-independent graph optimization.",
    };

    private readonly Option<string> _kvType = new("--kv-type")
    {
        Description = "KV cache element type built into LLM graphs: f16, f32, q8_0 or q4_0.",
        DefaultValueFactory = _ => "f16",
        HelpName = "type",
    };

    private readonly Option<int> _context = OptionFactory.Integer("--ctx", "Context length LLM graphs are built for (0 = the model's own).", 0, minimum: 0, helpName: "tokens");

    private readonly Option<string?> _mmproj = new("--mmproj")
    {
        Description = "Vision projector (GGUF mmproj) to build into the package with a GGUF language model.",
        HelpName = "file",
    };

    private readonly Option<string[]> _shapes = new("--shape")
    {
        HelpName = "NAME=DIMxDIM...",
        Description = "ONNX sources: full shape of an input with dynamic dimensions, e.g. pixel_values=-1x3x224x224 (a dimension <= 0 is the batch axis). Repeatable.",
        DefaultValueFactory = _ => [],
    };

    public ConvertCommand()
    {
        _kvType.AcceptOnlyFromAmong("f16", "f32", "q8_0", "q4_0");
        _shapes.Validators.Add(ValidateShapes);
    }

    public Command Build()
    {
        var command = new Command("convert", "Builds a model (GGUF, Hugging Face, ONNX or package) and writes it as a package without precompiled artifacts.");
        command.Arguments.Add(_source);
        command.Arguments.Add(_destination);
        command.Options.Add(_noOptimize);
        command.Options.Add(_kvType);
        command.Options.Add(_context);
        command.Options.Add(_mmproj);
        command.Options.Add(_shapes);
        command.SetAction(CommandRunner.Guard((parseResult, _) =>
        {
            new ModelConverter(Console.Out).Convert(Read(parseResult));
            return Task.FromResult(ExitCodes.Success);
        }));
        return command;
    }

    private void ValidateShapes(OptionResult result)
    {
        foreach (var text in result.GetValueOrDefault<string[]>() ?? [])
        {
            if (!InputShapeParser.TryParse(text, out _, out _, out var error))
            {
                result.AddError(error!);
            }
        }
    }

    private ConvertSettings Read(ParseResult parseResult)
    {
        var shapes = new Dictionary<string, long[]>(StringComparer.Ordinal);
        foreach (var text in parseResult.GetValue(_shapes) ?? [])
        {
            InputShapeParser.TryParse(text, out var name, out var dimensions, out _);
            shapes[name] = dimensions;
        }

        return new ConvertSettings(
            parseResult.GetValue(_source)!,
            parseResult.GetValue(_destination)!,
            !parseResult.GetValue(_noOptimize),
            DataType.Get(parseResult.GetValue(_kvType)!),
            parseResult.GetValue(_context),
            parseResult.GetValue(_mmproj),
            shapes);
    }
}
