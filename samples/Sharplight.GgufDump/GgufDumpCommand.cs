using System.CommandLine;
using Sharplight.Gguf;
using Sharplight.Samples;

namespace Sharplight.GgufDump;

/// <summary>The <c>GgufDump</c> command line: one GGUF file and an optional tensor listing.</summary>
internal sealed class GgufDumpCommand
{
    private readonly Argument<string> _file = new("file")
    {
        Description = "GGUF file to describe; for a sharded model the first shard.",
    };

    private readonly Option<bool> _tensors = new("--tensors")
    {
        Description = "Also list every tensor with its type, shape, size and file offset.",
    };

    public RootCommand Build()
    {
        var command = new RootCommand("Prints the metadata, tensors and tensor-type statistics of a GGUF model file.");
        command.Arguments.Add(_file);
        command.Options.Add(_tensors);
        command.SetAction(CommandRunner.Guard((parseResult, _) =>
        {
            Run(parseResult.GetValue(_file)!, parseResult.GetValue(_tensors));
            return Task.FromResult(ExitCodes.Success);
        }));
        return command;
    }

    private static void Run(string path, bool includeTensors)
    {
        if (!File.Exists(path))
        {
            throw new SampleException($"File '{path}' does not exist.");
        }

        using var model = GgufModelFiles.Open(path);
        new GgufReport(model, Console.Out).Write(includeTensors);
    }
}
