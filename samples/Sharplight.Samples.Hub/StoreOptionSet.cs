using System.CommandLine;
using Sharplight.Hub;

namespace Sharplight.Samples.Hub;

/// <summary><c>--models-dir</c>: the model store, given once on the root command and valid for every subcommand.</summary>
public sealed class StoreOptionSet
{
    /// <summary><c>--models-dir</c> (recursive: valid after any subcommand).</summary>
    public Option<string?> ModelsDirectory { get; } = new("--models-dir")
    {
        HelpName = "dir",
        Description = $"Model store for pulled models (default: ${ModelStore.EnvironmentVariable}, else 'models' next to the executable if it exists, else {ModelStore.UserRoot}).",
        Recursive = true,
    };

    /// <summary>The store directory the command line selects.</summary>
    public string ReadRoot(ParseResult parseResult) => ModelStore.ResolveRoot(parseResult.GetValue(ModelsDirectory));
}
