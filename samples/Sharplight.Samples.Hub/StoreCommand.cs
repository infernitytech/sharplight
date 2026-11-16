using System.CommandLine;
using Sharplight.Hub;

namespace Sharplight.Samples.Hub;

/// <summary>
/// <c>store</c>: shows where the model store is and why; <c>store use default|user|DIR</c> chooses it in the shared settings
/// file of the tools; <c>store migrate [--copy]</c> moves the models of the previous store (the system store, or the folder next to
/// the executable of earlier versions) into it. Where everything else is kept is the <c>storage</c> command's business.
/// </summary>
public sealed class StoreCommand(StoreOptionSet store)
{
    private const string DefaultChoice = "default";
    private const string LegacyDefaultChoice = "application";
    private const string UserChoice = "user";

    private readonly Argument<string> _location = new("location")
    {
        Description = $"'{DefaultChoice}' (the 'models' folder of the storage location, see 'storage'), "
            + $"'{UserChoice}' (the system store, {ModelStore.SystemRoot}) or a directory.",
    };

    private readonly Option<string?> _from = new("--from")
    {
        HelpName = "dir",
        Description = "The store to take the models from (default: the system store, or the folder next to the executable of earlier versions).",
    };

    private readonly Option<bool> _copy = new("--copy") { Description = "Copy the models instead of moving them." };

    public Command Build()
    {
        var command = new Command("store", "Shows where pulled models are stored and why.");
        command.SetAction(HubGuard.Guard((parseResult, _) =>
        {
            var resolution = store.Resolve(parseResult);
            Console.WriteLine($"Model store: {resolution.Root} ({resolution.Description})");
            Console.WriteLine($"Storage:     {store.Layout.Description}");
            Console.WriteLine($"Settings:    {store.SettingsFile}");
            if (ModelStore.FindPreviousStore(resolution, store.Layout) is { } previous)
            {
                Console.WriteLine($"Previous:    {previous.Root} has {previous.List().Count} model(s); 'store migrate' moves them here.");
            }

            return Task.FromResult(ExitCodes.Success);
        }));
        command.Subcommands.Add(BuildUse());
        command.Subcommands.Add(BuildMigrate());
        return command;
    }

    private Command BuildUse()
    {
        var command = new Command("use", "Chooses the model store of the tools (overridden by --models-dir and $" + ModelStore.EnvironmentVariable + ").");
        command.Arguments.Add(_location);
        command.SetAction(HubGuard.Guard((parseResult, _) =>
        {
            var value = parseResult.GetValue(_location)!.Trim();
            var settings = value.ToLowerInvariant() switch
            {
                DefaultChoice or LegacyDefaultChoice => new ModelStoreSettings { Location = ModelStoreLocation.Application },
                UserChoice => new ModelStoreSettings { Location = ModelStoreLocation.User },
                _ => new ModelStoreSettings { Location = ModelStoreLocation.Custom, CustomPath = Path.GetFullPath(value) },
            };
            settings.Save(store.SettingsFile);
            var resolution = ModelStore.Resolve(null, settings, store.Layout);
            Console.WriteLine($"Saved to {store.SettingsFile}: the model store is {resolution.Root} ({resolution.Description}).");
            return Task.FromResult(ExitCodes.Success);
        }));
        return command;
    }

    private Command BuildMigrate()
    {
        var command = new Command("migrate", "Moves the models of another store (by default the system store, or the folder of earlier versions) into the model store.");
        command.Options.Add(_from);
        command.Options.Add(_copy);
        command.SetAction(HubGuard.Guard((parseResult, cancellationToken) =>
        {
            var destination = new ModelStore(store.Resolve(parseResult).Root);
            var from = parseResult.GetValue(_from);
            var source = from is not null ? new ModelStore(from) : ModelStore.FindPreviousStore(store.Resolve(parseResult), store.Layout);
            if (source is null || source.Root == destination.Root)
            {
                Console.WriteLine($"Nothing to migrate into {destination.Root}.");
                return Task.FromResult(ExitCodes.Success);
            }

            bool copy = parseResult.GetValue(_copy);
            int count = ModelStoreTransfer.Transfer(source, destination, move: !copy, Console.WriteLine, cancellationToken);
            Console.WriteLine($"{count} model(s) {(copy ? "copied" : "moved")} from {source.Root} to {destination.Root}.");
            return Task.FromResult(ExitCodes.Success);
        }));
        return command;
    }
}
