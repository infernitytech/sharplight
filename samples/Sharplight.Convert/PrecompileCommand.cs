using System.CommandLine;
using System.CommandLine.Parsing;
using System.Globalization;
using Sharplight.Runtime;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary>
/// <c>precompile &lt;slm&gt; (--device provider:index | --profile file|name) [--tier code|binary|all]</c>: compiles every graph and
/// appends the artifact sets.
/// </summary>
internal sealed class PrecompileCommand
{
    private const string AllTiers = "all";
    private const char BoundSeparator = '=';
    /// <summary>The <c>sharplight-server --slots</c> default, so a served model finds its batch decoder's kernels.</summary>
    private const int DefaultServeSlots = 4;

    private readonly Argument<string> _package = PackageArgument.Create();

    private readonly Option<string?> _device = new("--device")
    {
        Description = "Device to compile on, as provider:index (e.g. vulkan:0); produces the code and binary tiers.",
        HelpName = "provider:index",
    };

    private readonly Option<string?> _profile = new("--profile")
    {
        Description = "Device profile to compile for without the device (a JSON file from 'profile', or a well-known name from 'profile --list'); produces the code tier.",
        HelpName = "file|name",
    };

    private readonly Option<string> _tier = new("--tier")
    {
        Description = "Artifact tiers to store: code (driver-independent kernels), binary (driver pipeline cache) or all.",
        DefaultValueFactory = _ => AllTiers,
        HelpName = "tier",
    };

    private readonly Option<string[]> _disable = OptionFactory.CommaSeparatedList("--disable-features",
        "Comma-separated device features not to use, as when the model runs with --disable-features.", "features");

    private readonly Option<int> _context = OptionFactory.Integer("--ctx", "Context length LLM graphs are compiled for (use the value the model will run with).",
        RuntimeOptionSet.DefaultContextLength, helpName: "tokens");

    private readonly Option<int> _batch = OptionFactory.Integer("--batch", "Rows per forward pass (use the value the model will run with).",
        RuntimeOptionSet.DefaultBatchTokens, helpName: "rows");

    private readonly Option<int> _slots = OptionFactory.Integer("--slots",
        "Sequences of the serving batch decoder to compile, as sharplight-server --slots (1 skips it).", DefaultServeSlots, helpName: "count");

    private readonly Option<long?> _kvTokens = OptionFactory.Size("--kv-tokens",
        "Paged KV cache of the serving batch decoder in tokens, as sharplight-server --kv-tokens (K/M suffixes).", helpName: "tokens");

    private readonly Option<string[]> _bounds = new("--bound")
    {
        Description = "Non-LLM graphs: upper bound of a symbolic dimension, e.g. batch=1 (ONNX batch defaults to 1). Repeatable.",
        HelpName = "SYMBOL=VALUE",
        DefaultValueFactory = _ => [],
    };

    private readonly Option<bool> _userCache = new("--use-kernel-cache")
    {
        Description = "Compile with the per-user kernel and pipeline cache. Faster; the stored sets still hold only this model's kernels and pipelines.",
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
        var command = new Command("precompile", "Compiles every graph of a package on a device, or for a device profile without the device, and appends the "
            + "resulting artifact sets (kernel code and/or driver binaries).");
        command.Arguments.Add(_package);
        command.Options.Add(_device);
        command.Options.Add(_profile);
        command.Options.Add(_tier);
        command.Options.Add(_disable);
        command.Options.Add(_context);
        command.Options.Add(_batch);
        command.Options.Add(_slots);
        command.Options.Add(_kvTokens);
        command.Options.Add(_bounds);
        command.Options.Add(_userCache);
        command.Validators.Add(result =>
        {
            bool device = result.GetValue(_device) is not null;
            bool profile = result.GetValue(_profile) is not null;
            if (device == profile)
            {
                result.AddError("Give exactly one of --device and --profile.");
            }
            else if (profile && result.GetValue(_tier) == ArtifactTiers.Binary)
            {
                result.AddError("A profile produces only the code tier; driver binaries need the device (--device).");
            }
        });
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
        DeviceSpec? device = parseResult.GetValue(_device) is { } deviceText && DeviceSpec.TryParse(deviceText, out var spec) ? spec : null;
        var profile = parseResult.GetValue(_profile) is { } profileText ? WellKnownProfiles.Resolve(profileText) : null;
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
            profile,
            tier == AllTiers ? null : new HashSet<string>(StringComparer.Ordinal) { tier },
            parseResult.GetValue(_disable) ?? [],
            parseResult.GetValue(_context),
            parseResult.GetValue(_batch),
            parseResult.GetValue(_slots),
            parseResult.GetValue(_kvTokens),
            bounds,
            parseResult.GetValue(_userCache));
    }
}
