using System.CommandLine;
using Sharplight.Package;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary><c>strip &lt;slm&gt; [--provider] [--tier]</c>: removes artifact sets from a package.</summary>
internal sealed class StripCommand
{
    private readonly Argument<string> _package = PackageArgument.Create();

    private readonly Option<string?> _provider = new("--provider")
    {
        Description = "Only remove sets of this provider (e.g. vulkan); default: every provider.",
        HelpName = "name",
    };

    private readonly Option<string?> _tier = new("--tier")
    {
        Description = "Only remove sets of this tier (code, binary, or mixed for version 1 kernel sets); default: every tier.",
        HelpName = "tier",
    };

    private readonly Option<bool> _compact = new("--compact") { Description = "Also rewrite the file without the removed bytes (see compact)." };

    public Command Build()
    {
        var command = new Command("strip", "Removes precompiled artifact sets (all, or by provider and tier). Only the section table is rewritten unless --compact is given.");
        command.Arguments.Add(_package);
        command.Options.Add(_provider);
        command.Options.Add(_tier);
        command.Options.Add(_compact);
        command.SetAction(CommandRunner.Guard((parseResult, _) =>
        {
            var path = PackageArgument.Require(parseResult.GetValue(_package)!);
            int removed = ModelPackage.Strip(path, parseResult.GetValue(_provider), parseResult.GetValue(_tier));
            Console.WriteLine($"Removed {removed} artifact set(s) from {path}.");
            if (parseResult.GetValue(_compact))
            {
                CompactCommand.Compact(path, Console.Out);
            }

            return Task.FromResult(ExitCodes.Success);
        }));
        return command;
    }
}
