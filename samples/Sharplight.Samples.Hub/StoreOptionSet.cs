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
        Description = $"Model store for pulled models (default: ${ModelStore.EnvironmentVariable}, else the location chosen with 'store use', "
            + $"else '{ModelStore.ApplicationDirectoryName}' next to the executable, or {ModelStore.UserRoot} when that is not writable).",
        Recursive = true,
    };

    /// <summary>The store the command line selects, and why.</summary>
    public ModelStoreResolution Resolve(ParseResult parseResult) => ModelStore.Resolve(parseResult.GetValue(ModelsDirectory));

    /// <summary>
    /// The store directory the command line selects. When nothing chose it explicitly, it has no models and another store has
    /// some (the per-user store of earlier versions), prints a one-line hint to the error output.
    /// </summary>
    public string ReadRoot(ParseResult parseResult)
    {
        var resolution = Resolve(parseResult);
        if (resolution.Source is ModelStoreSource.Default or ModelStoreSource.Settings && ModelStore.FindPreviousStore(resolution) is { } previous)
        {
            var use = previous.Root == new ModelStore(ModelStore.UserRoot).Root ? "user" : $"\"{previous.Root}\"";
            Console.Error.WriteLine($"Hint: {previous.List().Count} model(s) are in {previous.Root}, not in the store {resolution.Root}; "
                + $"keep using them with 'store use {use}', or move them with 'store migrate'.");
        }

        return resolution.Root;
    }
}
