using System.CommandLine;
using Sharplight.Runtime;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary>
/// <c>profile --device provider:index [-o file] [--disable-features]</c>: exports what code generation for a device depends on
/// (a <see cref="DeviceProfile"/>) as JSON, for <c>precompile --profile</c> on machines without that device;
/// <c>profile --list</c> lists the shipped profiles and <c>profile --show name</c> prints one.
/// </summary>
internal sealed class ProfileCommand
{
    private readonly Option<string?> _device = new("--device")
    {
        Description = "Device to export, as provider:index (e.g. vulkan:0).",
        HelpName = "provider:index",
    };

    private readonly Option<string?> _output = new("--output", "-o")
    {
        Description = "File to write the profile to (default: standard output).",
        HelpName = "file",
    };

    private readonly Option<string[]> _disable = OptionFactory.CommaSeparatedList("--disable-features",
        "Comma-separated device features the profile leaves out, as when the device is opened with --disable-features (e.g. matrix_f16,matrix_bf16).",
        "features");

    private readonly Option<bool> _list = new("--list") { Description = "List the well-known profiles that ship with this tool." };

    private readonly Option<string?> _show = new("--show")
    {
        Description = "Print a well-known profile.",
        HelpName = "name",
    };

    public ProfileCommand()
    {
        _device.Validators.Add(result =>
        {
            if (!DeviceSpec.TryParse(result.GetValueOrDefault<string>() ?? "", out _))
            {
                result.AddError("--device expects provider:index, e.g. vulkan:0.");
            }
        });
    }

    public Command Build()
    {
        var command = new Command("profile", "Exports a device profile (what kernel code generation depends on) for offline precompilation, or lists the shipped ones.");
        command.Options.Add(_device);
        command.Options.Add(_output);
        command.Options.Add(_disable);
        command.Options.Add(_list);
        command.Options.Add(_show);
        command.Validators.Add(result =>
        {
            int modes = (result.GetValue(_device) is null ? 0 : 1) + (result.GetValue(_list) ? 1 : 0) + (result.GetValue(_show) is null ? 0 : 1);
            if (modes != 1)
            {
                result.AddError("Give exactly one of --device, --list or --show.");
            }
        });
        command.SetAction(CommandRunner.Guard((parseResult, _) =>
        {
            Run(parseResult);
            return Task.FromResult(ExitCodes.Success);
        }));
        return command;
    }

    private void Run(ParseResult parseResult)
    {
        if (parseResult.GetValue(_list))
        {
            foreach (var name in WellKnownProfiles.Names)
            {
                Console.WriteLine($"{name}: {WellKnownProfiles.Get(name)}");
            }

            return;
        }

        var profile = parseResult.GetValue(_show) is { } shown ? WellKnownProfiles.Get(shown) : Export(parseResult);
        if (parseResult.GetValue(_output) is { } path)
        {
            File.WriteAllText(path, profile.ToJson() + Environment.NewLine);
            Console.Error.WriteLine($"Wrote {profile} to {path}");
        }
        else
        {
            Console.WriteLine(profile.ToJson());
        }
    }

    private DeviceProfile Export(ParseResult parseResult)
    {
        DeviceSpec.TryParse(parseResult.GetValue(_device)!, out var spec);
        var options = new DeviceOptions();
        foreach (var feature in parseResult.GetValue(_disable) ?? [])
        {
            options.DisabledFeatures.Add(feature);
        }

        using var device = DeviceOpener.Open(spec, options);
        return device.CreateProfile()
            ?? throw new SampleException($"Provider '{spec.Provider}' cannot compile offline, so its devices have no profile.", ExitCodes.InvalidInput);
    }
}
