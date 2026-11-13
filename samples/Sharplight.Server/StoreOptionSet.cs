using System.CommandLine;
using Sharplight.Hub;

namespace Sharplight.Server;

/// <summary><c>--models-dir</c>: the model store, given once on the root command and valid for every subcommand.</summary>
internal sealed class StoreOptionSet
{
    public Option<string?> ModelsDirectory { get; } = new("--models-dir")
    {
        HelpName = "dir",
        Description = $"Model store for pulled models (default: ${ModelStore.EnvironmentVariable}, else 'models' next to the executable).",
        Recursive = true,
    };

    /// <summary>The store directory the command line selects.</summary>
    public string ReadRoot(ParseResult parseResult) => ModelStore.ResolveRoot(parseResult.GetValue(ModelsDirectory));
}
