using System.CommandLine;
using Sharplight.Hub;
using Sharplight.Runtime;
using Sharplight.Storage;

namespace Sharplight.Samples.Hub;

/// <summary>
/// <c>storage</c>: shows where the tools keep their files (portable, system or custom; see <see cref="StorageLayout"/>);
/// <c>storage use portable|system|DIR [--copy|--fresh] [--keep-models]</c> changes it and moves the existing files along.
/// </summary>
public sealed class StorageCommand(StoreOptionSet store)
{
    private const string PortableChoice = "portable";
    private const string SystemChoice = "system";

    private readonly Argument<string> _location = new("location")
    {
        Description = $"'{PortableChoice}' (a '{StorageLayout.PortableDirectoryName}' folder next to the executable), "
            + $"'{SystemChoice}' (the platform's per-user folders) or a directory (custom).",
    };

    private readonly Option<bool> _copy = new("--copy") { Description = "Copy the files instead of moving them." };

    private readonly Option<bool> _fresh = new("--fresh") { Description = "Leave the files where they are and start with empty folders." };

    private readonly Option<bool> _keepModels = new("--keep-models")
    {
        Description = "Leave the model store where it is (it is recorded in the new location's store settings).",
    };

    public Command Build()
    {
        var command = new Command("storage", "Shows where configuration, models, caches and logs are kept.");
        command.SetAction(HubGuard.Guard((parseResult, _) =>
        {
            var layout = store.Layout;
            Console.WriteLine($"Storage:     {layout.Description}");
            Console.WriteLine($"Config:      {layout.ConfigDirectory}");
            Console.WriteLine($"Data:        {layout.DataDirectory}");
            Console.WriteLine($"Models:      {store.Resolve(parseResult).Root}");
            Console.WriteLine($"Cache:       {layout.CacheDirectory}");
            Console.WriteLine($"Logs:        {layout.LogsDirectory}");
            Console.WriteLine($"Kernels:     {ArtifactStore.DirectoryFor(layout) ?? "off"}");
            return Task.FromResult(ExitCodes.Success);
        }));
        command.Subcommands.Add(BuildUse());
        return command;
    }

    private Command BuildUse()
    {
        var command = new Command("use", $"Changes where the tools keep their files (overridden by ${StorageLayout.HomeVariable}).");
        command.Arguments.Add(_location);
        command.Options.Add(_copy);
        command.Options.Add(_fresh);
        command.Options.Add(_keepModels);
        command.SetAction(HubGuard.Guard(async (parseResult, cancellationToken) =>
        {
            var current = store.Layout;
            if (current.Source == StorageModeSource.EnvironmentVariable)
            {
                throw new SampleException($"${StorageLayout.HomeVariable} sets the storage folder; unset it to choose another one.");
            }

            var value = parseResult.GetValue(_location)!.Trim();
            var environment = StorageEnvironment.Process;
            var target = value.ToLowerInvariant() switch
            {
                PortableChoice when !StorageLayout.CanUsePortable(environment) =>
                    throw new SampleException($"{current.PortableRoot} cannot be written; the portable mode needs a writable folder next to the executable."),
                PortableChoice => StorageLayout.Create(StorageMode.Portable, null, environment),
                SystemChoice => StorageLayout.Create(StorageMode.System, null, environment),
                _ => StorageLayout.Create(StorageMode.Custom, Path.GetFullPath(value), environment),
            };

            var kind = parseResult.GetValue(_fresh) ? StorageTransferKind.StartFresh
                : parseResult.GetValue(_copy) ? StorageTransferKind.Copy
                : StorageTransferKind.Move;
            var models = store.Resolve(parseResult);
            bool keepModels = parseResult.GetValue(_keepModels) || models.Location != ModelStoreLocation.Application;
            var categories = Enum.GetValues<StorageCategory>().Where(category => !keepModels || category != StorageCategory.Models);
            var migration = new StorageMigration(current, target, categories);
            if (migration.Validate() is { } problem && kind != StorageTransferKind.StartFresh)
            {
                throw new SampleException(problem);
            }

            var result = await migration.RunAsync(kind, new StorageProgressPrinter(), cancellationToken);
            if (keepModels && models.Location == ModelStoreLocation.Application)
            {
                new ModelStoreSettings { Location = ModelStoreLocation.Custom, CustomPath = models.Root }.Save(ModelStoreSettings.FileFor(target));
            }

            // Nothing of this tool has the old files open: a move finishes right away.
            int deleted = StorageMigration.CompletePending(target);
            Console.Error.WriteLine();
            Console.WriteLine($"Storage is now {target.Description.ToLowerInvariant()}: {target.Root ?? target.DataDirectory}. "
                + $"{result.FilesCopied} file(s) {(kind == StorageTransferKind.Move ? "moved" : "copied")}"
                + (result.RenamedCategories.Count > 0 ? ", the model store moved in one step" : "")
                + (deleted > 0 ? $", {deleted} original(s) removed." : "."));
            return ExitCodes.Success;
        }));
        return command;
    }
}
