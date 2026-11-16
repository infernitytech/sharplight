namespace Sharplight.Storage;

/// <summary>What a <see cref="StorageMigration"/> does with the files of the old location.</summary>
public enum StorageTransferKind
{
    /// <summary>Copy them, then delete the originals once the copies are verified and the new location is in use.</summary>
    Move,

    /// <summary>Copy them and keep the originals.</summary>
    Copy,

    /// <summary>Leave them where they are and start with empty folders.</summary>
    StartFresh,
}
