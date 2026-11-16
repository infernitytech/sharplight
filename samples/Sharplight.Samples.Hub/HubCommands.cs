using System.CommandLine;

namespace Sharplight.Samples.Hub;

/// <summary>
/// Adds the model store commands (<c>pull</c>, <c>search</c>, <c>list</c>, <c>show</c>, <c>rm</c>, <c>store</c>, <c>storage</c>) and the recursive
/// <c>--models-dir</c> option to a sample's root command, so every Sharplight executable manages the same store the same way.
/// </summary>
public static class HubCommands
{
    /// <summary>Adds <c>--models-dir</c> and the store subcommands to <paramref name="root"/>.</summary>
    /// <param name="root">The root command of the executable.</param>
    /// <param name="store">The store option, also used by the executable's own commands that take a model.</param>
    /// <param name="runtime">The Sharplight services (conversion and precompilation of pulled models).</param>
    public static void AddTo(Command root, StoreOptionSet store, SampleRuntime runtime)
    {
        root.Options.Add(store.ModelsDirectory);
        root.Subcommands.Add(new PullCommand(store, runtime).Build());
        root.Subcommands.Add(new SearchCommand(store, runtime).Build());
        root.Subcommands.Add(new ListCommand(store).Build());
        root.Subcommands.Add(new ShowCommand(store).Build());
        root.Subcommands.Add(new RemoveCommand(store).Build());
        root.Subcommands.Add(new StoreCommand(store).Build());
        root.Subcommands.Add(new StorageCommand(store).Build());
    }
}
