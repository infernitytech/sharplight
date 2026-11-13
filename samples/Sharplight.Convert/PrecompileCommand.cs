using System.CommandLine;
using System.CommandLine.Parsing;
using System.Globalization;
using Sharplight.Runtime;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary><c>precompile &lt;slm&gt; --device provider:index [--tier code|binary|all]</c>: compiles every graph and appends the artifact sets.</summary>
internal sealed class PrecompileCommand
{
    private const string AllTiers = "all";
    private const char BoundSeparator = '=';

    private readonly Argument<string> _package = PackageArgument.Create();

    private readonly Option<string> _device = new("--device")
    {
        Description = "Device to compile on, as provider:index (e.g. vulkan:0).",
        HelpName = "provider:index",
        Required = true,
    };

    private readonly Option<string> _tier = new("--tier")
    {
        Description = "Artifact tiers to store: code (driver-independent kernels), binary (driver pipeline cache) or all.",
        DefaultValueFactory = _ => AllTiers,
        HelpName = "tier",
    };

    private readonly Option<int> _context = OptionFactory.Integer("--ctx", "Context length LLM graphs are compiled for (use the value the model will run with).",
        RuntimeOptionSet.DefaultContextLength, helpName: "tokens");

    private readonly Option<int> _batch = OptionFactory.Integer("--batch", "Rows per forward pass (use the value the model will run with).",
        RuntimeOptionSet.DefaultBatchTokens, helpName: "rows");

    private readonly Option<string[]> _bounds = new("--bound")
    {
        Description = "Non-LLM graphs: upper bound of a symbolic dimension, e.g. batch=1. Repeatable.",
        HelpName = "SYMBOL=VALUE",
        DefaultValueFactory = _ => [],
    };

    private readonly Option<bool> _userCache = new("--use-kernel-cache")
    {
        Description = "Compile with the per-user kernel and pipeline cache. Faster, but the stored pipeline cache then also holds pipelines of other models.",
    };

    public PrecompileCommand()
    {
        _tier.AcceptOnlyFromAmong(ArtifactTiers.Code, ArtifactTiers.Binary, AllTiers);
        _device.Validators.Add(result =>
        {
            if (!DeviceSpec.TryParse(result.GetValueOrDefault<string>() ?? "", out _))
            {
                result.AddError("--device expects provider:index, e.g. vulkan:0.");
            }
        });
        _bounds.Validators.Add(ValidateBounds);
    }

    public Command Build()
    {
        var command = new Command("precompile", "Compiles every graph of a package on a device and appends the resulting artifact sets (kernel code and/or driver binaries).");
        command.Arguments.Add(_package);
        command.Options.Add(_device);
        command.Options.Add(_tier);
        command.Options.Add(_context);
        command.Options.Add(_batch);
        command.Options.Add(_bounds);
        command.Options.Add(_userCache);
        command.SetAction(CommandRunner.Guard((parseResult, _) =>
        {
            new PackagePrecompiler(Console.Out).Run(Read(parseResult));
            return Task.FromResult(ExitCodes.Success);
        }));
        return command;
    }

    private void ValidateBounds(OptionResult result)
    {
        foreach (var text in result.GetValueOrDefault<string[]>() ?? [])
        {
            if (!TryParseBound(text, out _, out _))
            {
                result.AddError($"--bound expects SYMBOL=VALUE with a positive integer value, got '{text}'.");
            }
        }
    }

    private static bool TryParseBound(string text, out string symbol, out long value)
    {
        int separator = text.IndexOf(BoundSeparator);
        symbol = separator > 0 ? text[..separator] : "";
        value = 0;
        return separator > 0 && long.TryParse(text[(separator + 1)..], NumberStyles.None, CultureInfo.InvariantCulture, out value) && value > 0;
    }

    private PrecompileSettings Read(ParseResult parseResult)
    {
        DeviceSpec.TryParse(parseResult.GetValue(_device)!, out var device);
        var tier = parseResult.GetValue(_tier)!;
        var bounds = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var text in parseResult.GetValue(_bounds) ?? [])
        {
            TryParseBound(text, out var symbol, out long value);
            bounds[symbol] = value;
        }

        return new PrecompileSettings(
            PackageArgument.Require(parseResult.GetValue(_package)!),
            device,
            tier == AllTiers ? null : new HashSet<string>(StringComparer.Ordinal) { tier },
            parseResult.GetValue(_context),
            parseResult.GetValue(_batch),
            bounds,
            parseResult.GetValue(_userCache));
    }
}
