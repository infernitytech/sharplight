using System.CommandLine;
using System.Diagnostics;
using Sharplight.Package;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary><c>compact &lt;slm&gt;</c>: rewrites a package without dead space.</summary>
internal sealed class CompactCommand
{
    private readonly Argument<string> _package = PackageArgument.Create();

    public Command Build()
    {
        var command = new Command("compact", "Rewrites a package without the space of replaced or stripped artifact sets (weights are copied unchanged).");
        command.Arguments.Add(_package);
        command.SetAction(CommandRunner.Guard((parseResult, _) =>
        {
            Compact(PackageArgument.Require(parseResult.GetValue(_package)!), Console.Out);
            return Task.FromResult(ExitCodes.Success);
        }));
        return command;
    }

    public static void Compact(string path, TextWriter output)
    {
        var stopwatch = Stopwatch.StartNew();
        long saved = ModelPackage.Compact(path);
        output.WriteLine($"Compacted {path}: {ByteSize.Format(saved)} removed, now {ByteSize.Format(new FileInfo(path).Length)} [{stopwatch.ElapsedMilliseconds} ms]");
    }
}
