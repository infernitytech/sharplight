using System.CommandLine;
using Sharplight.Package;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary>
/// <c>optimize &lt;slm&gt; [-o dst.slm]</c>: runs the device-independent graph optimization over a package's stored definition,
/// in place or into a new package.
/// </summary>
internal sealed class OptimizeCommand
{
    private readonly Argument<string> _package = PackageArgument.Create();

    private readonly Option<string?> _output = new("-o", "--output")
    {
        Description = "Write the optimized package here instead of replacing the input in place.",
        HelpName = "dst.slm",
    };

    private readonly Option<bool> _force = new("--force")
    {
        Description = "Optimize even when the package already carries the current pipeline id (by default it is left unchanged).",
    };

    private readonly Option<bool> _dropArtifacts = new("--drop-artifacts")
    {
        Description = "Do not copy the package's precompiled artifact sets (by default they are kept; code entries are content addressed and stay valid).",
    };

    private readonly Option<bool> _passes = new("--passes")
    {
        Description = "Also print what each optimization pass did to each graph.",
    };

    public Command Build()
    {
        var command = new Command("optimize", "Runs the device-independent graph optimization (identity removal, view chain simplification, constant "
            + "folding, common-subexpression and dead-node elimination) over a package's stored graphs and rewrites it, for packages converted by an "
            + "older pipeline or with --no-optimize; loading the result skips the phase. Weights, description, attachments and artifact sets are kept.");
        command.Arguments.Add(_package);
        command.Options.Add(_output);
        command.Options.Add(_force);
        command.Options.Add(_dropArtifacts);
        command.Options.Add(_passes);
        command.SetAction(CommandRunner.Guard((parseResult, _) =>
        {
            Run(parseResult, Console.Out);
            return Task.FromResult(ExitCodes.Success);
        }));
        return command;
    }

    private void Run(ParseResult parseResult, TextWriter output)
    {
        var path = PackageArgument.Require(parseResult.GetValue(_package)!);
        var options = new PackageOptimizeOptions
        {
            Force = parseResult.GetValue(_force),
            KeepArtifacts = !parseResult.GetValue(_dropArtifacts),
            Log = output.WriteLine,
        };
        var result = PackageOptimizer.Optimize(path, parseResult.GetValue(_output), options);
        if (!result.Optimized)
        {
            output.WriteLine($"{path} is already optimized ({result.PipelineId}); use --force to run the pipeline again.");
            return;
        }

        if (parseResult.GetValue(_passes))
        {
            foreach (var s in result.Statistics)
            {
                output.WriteLine($"  {s.Graph} round {s.Round} {s.Pass}: {s.NodesBefore} -> {s.NodesAfter} nodes{(s.Changed ? "" : " (unchanged)")} [{s.Elapsed.TotalMilliseconds:F0} ms]");
            }
        }

        var graphs = string.Join(", ", result.NodesBefore.Select(kv => $"{kv.Key} {kv.Value} -> {result.NodesAfter[kv.Key]} nodes"));
        output.WriteLine($"Optimized ({result.PipelineId}): {graphs} [{result.OptimizeTime.TotalMilliseconds:F0} ms]");
        output.WriteLine($"Saved {result.PackagePath}: {ByteSize.Format(result.PackageBytes)}, {result.ArtifactSets} artifact set(s) [{result.SaveTime.TotalMilliseconds:F0} ms]");
    }
}
