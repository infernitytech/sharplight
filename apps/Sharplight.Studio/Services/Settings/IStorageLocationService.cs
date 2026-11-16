using Sharplight.Storage;

namespace Sharplight.Studio.Services.Settings;

/// <summary>
/// The storage location of everything Studio and the engine write (portable, system or custom; see <see cref="StorageLayout"/>):
/// the layout in effect, previews of the other modes, and changing it with the existing files.
/// </summary>
internal interface IStorageLocationService
{
    /// <summary>The layout in effect for this run.</summary>
    public StorageLayout Current { get; }

    /// <summary>Studio's own folders in effect for this run.</summary>
    public StudioPaths Paths { get; }

    /// <summary><c>SHARPLIGHT_STUDIO_HOME</c> keeps Studio's own files elsewhere, whatever the layout is.</summary>
    public bool IsStudioHomeOverridden { get; }

    /// <summary>The layout committed for the next start by <see cref="ChangeAsync"/>, or null when nothing changed.</summary>
    public StorageLayout? Pending { get; }

    /// <summary>The portable mode can be used: the folder next to the package is writable and not part of an installation.</summary>
    public bool CanUsePortable { get; }

    /// <summary>The layout <paramref name="mode"/> would give (custom: <paramref name="customRoot"/>); null for a custom mode without a folder.</summary>
    public StorageLayout? Preview(StorageMode mode, string? customRoot);

    /// <summary>Why <paramref name="request"/> cannot be carried out, or null.</summary>
    public string? Validate(StorageChangeRequest request);

    /// <summary>
    /// Transfers the files and records the new location for the next start (see <see cref="StorageMigration"/>); a move deletes the
    /// originals at that start. Studio must restart to use the new location.
    /// </summary>
    public Task<StorageMigrationResult> ChangeAsync(StorageChangeRequest request, IProgress<StorageMigrationProgress>? progress,
        CancellationToken cancellationToken);
}
