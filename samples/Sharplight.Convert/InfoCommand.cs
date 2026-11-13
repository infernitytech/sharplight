using System.CommandLine;
using Sharplight.Package;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary><c>info &lt;slm&gt;</c>: prints the structure of a package (<see cref="ModelPackageReader"/>).</summary>
internal sealed class InfoCommand
{
    private readonly Argument<string> _package = PackageArgument.Create();

    private readonly Option<bool> _entries = new("--entries") { Description = "List every entry of every artifact set with its size." };

    private readonly Option<bool> _weights = new("--weights") { Description = "List every weight with type, shape and size." };

    private readonly Option<string?> _nodes = new("--nodes")
    {
        Description = "Print the node table of this graph.",
        HelpName = "graph",
    };

    public Command Build()
    {
        var command = new Command("info", "Prints a package's version, sections, graphs (op histogram, inputs, outputs), properties, weights and artifact sets.");
        command.Arguments.Add(_package);
        command.Options.Add(_entries);
        command.Options.Add(_weights);
        command.Options.Add(_nodes);
        command.SetAction(CommandRunner.Guard((parseResult, _) =>
        {
            var info = Open(PackageArgument.Require(parseResult.GetValue(_package)!));
            var report = new PackageReport(info, Console.Out);
            report.Write(parseResult.GetValue(_weights), parseResult.GetValue(_entries));
            if (parseResult.GetValue(_nodes) is { } graph)
            {
                report.WriteNodes(graph);
            }

            return Task.FromResult(ExitCodes.Success);
        }));
        return command;
    }

    private static PackageInfo Open(string path)
    {
        try
        {
            return ModelPackageReader.Open(path);
        }
        catch (InvalidDataException error)
        {
            throw new SampleException($"Cannot read '{path}': {error.Message}", ExitCodes.InvalidInput);
        }
    }
}
