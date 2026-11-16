namespace Sharplight.Storage;

/// <summary>One folder a <see cref="StorageMigration"/> transfers.</summary>
/// <param name="Category">The kind of files.</param>
/// <param name="Source">The folder in the layout in effect.</param>
/// <param name="Destination">The folder in the new layout.</param>
public sealed record StorageMigrationItem(StorageCategory Category, string Source, string Destination);
