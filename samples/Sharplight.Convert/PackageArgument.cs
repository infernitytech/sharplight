using System.CommandLine;
using Sharplight.Package;
using Sharplight.Samples;

namespace Sharplight.Convert;

/// <summary>The <c>&lt;slm&gt;</c> argument of the subcommands that work on an existing package.</summary>
internal static class PackageArgument
{
    public static Argument<string> Create() => new("slm") { Description = "The model package (.slm)." };

    /// <summary>Returns <paramref name="path"/> when it is a package.</summary>
    /// <exception cref="SampleException">The file does not exist or is not a package.</exception>
    public static string Require(string path)
    {
        if (!File.Exists(path))
        {
            throw new SampleException($"Package '{path}' does not exist.", ExitCodes.InvalidInput);
        }

        if (!ModelPackage.IsPackage(path))
        {
            throw new SampleException($"'{path}' is not a Sharplight model package.", ExitCodes.InvalidInput);
        }

        return path;
    }
}
