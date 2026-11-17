using Sharplight.Hub;
using Sharplight.Storage;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Settings;

/// <summary><see cref="IStorageLocationService"/> over <see cref="StorageLayout"/> and <see cref="StorageMigration"/>.</summary>
internal sealed class StorageLocationService : IStorageLocationService
{
    private static readonly string[] s_rootMarkers = ["config", "models", StudioPaths.ApplicationDirectoryName, "cache"];

    private readonly StorageEnvironment _environment;
    private readonly ISettingsService _settings;

    /// <summary>Creates the service for the layout <paramref name="current"/> resolved from <paramref name="environment"/>.</summary>
    public StorageLocationService(StorageEnvironment environment, StorageLayout current, ISettingsService settings)
    {
        _environment = environment;
        _settings = settings;
        Current = current;
        var home = environment.GetVariable(StudioPaths.HomeVariable);
        IsStudioHomeOverridden = !string.IsNullOrWhiteSpace(home);
        Paths = StudioPaths.From(current, home);
    }

    /// <inheritdoc/>
    public StorageLayout Current { get; }

    /// <inheritdoc/>
    public StudioPaths Paths { get; }

    /// <inheritdoc/>
    public bool IsStudioHomeOverridden { get; }

    /// <inheritdoc/>
    public StorageLayout? Pending { get; private set; }

    /// <inheritdoc/>
    public bool CanUsePortable => StorageLayout.CanUsePortable(_environment);

    /// <inheritdoc/>
    public StorageLayout? Preview(StorageMode mode, string? customRoot)
    {
        if (mode == StorageMode.Custom && string.IsNullOrWhiteSpace(customRoot))
        {
            return null;
        }

        return StorageLayout.Create(mode, customRoot is null ? null : Path.GetFullPath(customRoot.Trim()), _environment);
    }

    /// <inheritdoc/>
    public string? Validate(StorageChangeRequest request)
    {
        if (Current.Source == StorageModeSource.EnvironmentVariable)
        {
            return $"{StorageLayout.HomeVariable} sets the storage folder; unset it to choose another one.";
        }

        if (Preview(request.Mode, request.CustomRoot) is not { } target)
        {
            return "Choose a folder for the custom location.";
        }

        if (request.Mode == StorageMode.Portable && !CanUsePortable)
        {
            return $"{Current.PortableRoot} cannot be written (a read-only or system-wide installation), so the portable mode is not available.";
        }

        if (target.HasSameFolders(Pending ?? Current))
        {
            return "This is the location in use.";
        }

        if (request.Mode == StorageMode.Custom && !IsEmptyOrSharplightRoot(target.Root!))
        {
            return $"{target.Root} already holds other files. Choose an empty folder or an existing Sharplight storage folder.";
        }

        return request.Transfer == StorageTransferKind.StartFresh ? null : CreateMigration(target, request).Validate();
    }

    /// <inheritdoc/>
    public async Task<StorageMigrationResult> ChangeAsync(StorageChangeRequest request, IProgress<StorageMigrationProgress>? progress,
        CancellationToken cancellationToken)
    {
        if (Validate(request) is { } problem)
        {
            throw new InvalidOperationException(problem);
        }

        var target = Preview(request.Mode, request.CustomRoot)!;
        var store = _settings.Store;
        bool keepModels = request.KeepModels && FollowsLayout(store);
        if (keepModels)
        {
            // Before the copy, so the copied settings keep using the models where they are.
            _settings.Update(current => current with { ModelsLocation = ModelStoreLocation.Custom, ModelsDirectory = store.Root });
        }

        var migration = CreateMigration(target, request);
        var result = await Task.Run(() => migration.RunAsync(request.Transfer, progress, cancellationToken), cancellationToken);
        if (request.Transfer == StorageTransferKind.StartFresh && keepModels && !IsStudioHomeOverridden)
        {
            var settingsFile = StudioPaths.From(target, null).SettingsFile;
            if (!File.Exists(settingsFile))
            {
                JsonSettingsService.Save(settingsFile, new AppSettings { ModelsLocation = ModelStoreLocation.Custom, ModelsDirectory = store.Root });
            }
        }

        Pending = target;
        if (!IsStudioHomeOverridden)
        {
            _settings.RelocateTo(StudioPaths.From(target, null).SettingsFile);
        }

        return result;
    }

    /// <summary>True when the model store is the layout's models folder (not a folder chosen on its own, the command line or the environment).</summary>
    internal static bool FollowsLayout(ModelStoreResolution store) =>
        store.Location == ModelStoreLocation.Application && store.Source is ModelStoreSource.Default or ModelStoreSource.Settings;

    private StorageMigration CreateMigration(StorageLayout target, StorageChangeRequest request)
    {
        bool moveModels = !request.KeepModels && FollowsLayout(_settings.Store);
        var categories = Enum.GetValues<StorageCategory>().Where(category => moveModels || category != StorageCategory.Models);
        return new StorageMigration(Current, target, categories);
    }

    private static bool IsEmptyOrSharplightRoot(string root)
    {
        if (!Directory.Exists(root) || !Directory.EnumerateFileSystemEntries(root).Any())
        {
            return true;
        }

        return s_rootMarkers.Any(marker => Directory.Exists(Path.Combine(root, marker)));
    }
}
