using System.CommandLine;
using Sharplight.Hub;

namespace Sharplight.Samples.Hub;

/// <summary>
/// <c>store</c>: shows where the model store is and why; <c>store use application|user|DIR</c> chooses it in the shared settings
/// file of the tools; <c>store migrate [--copy]</c> moves the models of the previous store (the per-user store) into it.
/// </summary>
public sealed class StoreCommand(StoreOptionSet store)
{
    private const string ApplicationChoice = "application";
    private const string UserChoice = "user";

    private readonly Argument<string> _location = new("location")
    {
        Description = $"'{ApplicationChoice}' (the '{ModelStore.ApplicationDirectoryName}' folder next to the executable, the default), "
            + $"'{UserChoice}' ({ModelStore.UserRoot}) or a directory.",
    };

    private readonly Option<string?> _from = new("--from")
    {
        HelpName = "dir",
        Description = "The store to take the models from (default: the per-user store, or 'models' next to the executable).",
    };

    private readonly Option<bool> _copy = new("--copy") { Description = "Copy the models instead of moving them." };

    public Command Build()
    {
        var command = new Command("store", "Shows where pulled models are stored and why.");
        command.SetAction(HubGuard.Guard((parseResult, _) =>
        {
            var resolution = store.Resolve(parseResult);
            Console.WriteLine($"Model store: {resolution.Root} ({resolution.Description})");
            Console.WriteLine($"Settings:    {ModelStoreSettings.DefaultFile}");
            if (ModelStore.FindPreviousStore(resolution) is { } previous)
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
                ApplicationChoice => new ModelStoreSettings { Location = ModelStoreLocation.Application },
                UserChoice => new ModelStoreSettings { Location = ModelStoreLocation.User },
                _ => new ModelStoreSettings { Location = ModelStoreLocation.Custom, CustomPath = Path.GetFullPath(value) },
            };
            settings.Save(ModelStoreSettings.DefaultFile);
            var resolution = ModelStore.Resolve(null, settings);
            Console.WriteLine($"Saved to {ModelStoreSettings.DefaultFile}: the model store is {resolution.Root} ({resolution.Description}).");
            return Task.FromResult(ExitCodes.Success);
        }));
        return command;
    }

    private Command BuildMigrate()
    {
        var command = new Command("migrate", "Moves the models of another store (by default the per-user store of earlier versions) into the model store.");
        command.Options.Add(_from);
        command.Options.Add(_copy);
        command.SetAction(HubGuard.Guard((parseResult, cancellationToken) =>
        {
            var destination = new ModelStore(store.Resolve(parseResult).Root);
            var from = parseResult.GetValue(_from);
            var source = from is not null ? new ModelStore(from) : ModelStore.FindPreviousStore(store.Resolve(parseResult));
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
