namespace Sharplight.Storage;

/// <summary>
/// Records the storage mode for the next start: the portable marker next to the package, or the pointer file in the system
/// configuration folder (see <see cref="StorageLayout.Resolve"/>). Moving existing files is <see cref="StorageMigration"/>'s job.
/// </summary>
public static class StorageSelection
{
    /// <summary>Makes <paramref name="mode"/> the storage of the package <paramref name="current"/> belongs to.</summary>
    /// <param name="current">The layout in effect (gives the package folder and the pointer file).</param>
    /// <param name="mode">The mode to use from now on.</param>
    /// <param name="customRoot">The root folder of <see cref="StorageMode.Custom"/>; ignored otherwise.</param>
    /// <exception cref="ArgumentException">A custom mode without a root.</exception>
    public static void Apply(StorageLayout current, StorageMode mode, string? customRoot)
    {
        ArgumentNullException.ThrowIfNull(current);
        switch (mode)
        {
            case StorageMode.Portable:
                Directory.CreateDirectory(current.PortableRoot);
                StorageFiles.WriteAtomically(current.PortableMarkerFile, "{}" + Environment.NewLine);
                break;
            case StorageMode.Custom:
                if (string.IsNullOrWhiteSpace(customRoot))
                {
                    throw new ArgumentException("A custom storage folder is required.", nameof(customRoot));
                }

                StorageFiles.DeleteIfExists(current.PortableMarkerFile);
                new StoragePointer { Mode = StorageMode.Custom, Root = Path.GetFullPath(customRoot.Trim()) }.Save(current.PointerFile);
                break;
            default:
                StorageFiles.DeleteIfExists(current.PortableMarkerFile);
                StorageFiles.DeleteIfExists(current.PointerFile);
                break;
        }
    }
}
