using System.CommandLine;
using Sharplight.Hub;
using Sharplight.Storage;

namespace Sharplight.Samples.Hub;

/// <summary><c>--models-dir</c>: the model store, given once on the root command and valid for every subcommand.</summary>
public sealed class StoreOptionSet
{
    private readonly Lazy<StorageLayout> _layout = new(() => StorageLayout.Resolve(StorageEnvironment.Process));

    /// <summary><c>--models-dir</c> (recursive: valid after any subcommand).</summary>
    public Option<string?> ModelsDirectory { get; } = new("--models-dir")
    {
        HelpName = "dir",
        Description = $"Model store for pulled models (default: ${ModelStore.EnvironmentVariable}, else the location chosen with 'store use', "
            + "else the 'models' folder of the storage location: portable, system or custom, see 'storage').",
        Recursive = true,
    };

    /// <summary>
    /// The storage layout of this tool (<see cref="StorageLayout.Resolve"/>: <c>SHARPLIGHT_HOME</c>, a portable marker next to the
    /// executable, the pointer file, else the system folders).
    /// </summary>
    public StorageLayout Layout => _layout.Value;

    /// <summary>The shared store settings file of the layout (<c>store use</c> writes it).</summary>
    public string SettingsFile => ModelStoreSettings.FileFor(Layout);

    /// <summary>The store the command line selects, and why.</summary>
    public ModelStoreResolution Resolve(ParseResult parseResult) =>
        ModelStore.Resolve(parseResult.GetValue(ModelsDirectory), ModelStoreSettings.Load(SettingsFile), Layout);

    /// <summary>
    /// The store directory the command line selects. When nothing chose it explicitly, it has no models and another store has
    /// some (the system store, or the folder next to the executable of earlier versions), prints a one-line hint to the error output.
    /// </summary>
    public string ReadRoot(ParseResult parseResult)
    {
        var resolution = Resolve(parseResult);
        if (resolution.Source is ModelStoreSource.Default or ModelStoreSource.Settings && ModelStore.FindPreviousStore(resolution, Layout) is { } previous)
        {
            var use = previous.Root == new ModelStore(ModelStore.SystemRoot).Root ? "user" : $"\"{previous.Root}\"";
            Console.Error.WriteLine($"Hint: {previous.List().Count} model(s) are in {previous.Root}, not in the store {resolution.Root}; "
                + $"keep using them with 'store use {use}', or move them with 'store migrate'.");
        }

        return resolution.Root;
    }
}
