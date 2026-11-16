namespace Sharplight.Storage;

/// <summary>What a <see cref="StorageMigration"/> transferred.</summary>
/// <param name="FilesCopied">Files copied and verified (including files the destination already had unchanged).</param>
/// <param name="BytesCopied">Bytes of those files.</param>
/// <param name="RenamedCategories">Folders moved in one rename (same volume) instead of being copied.</param>
public sealed record StorageMigrationResult(int FilesCopied, long BytesCopied, IReadOnlyList<StorageCategory> RenamedCategories);
