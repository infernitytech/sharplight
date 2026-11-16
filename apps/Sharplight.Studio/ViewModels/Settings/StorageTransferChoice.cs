using Sharplight.Storage;

namespace Sharplight.Studio.ViewModels.Settings;

/// <summary>What happens to the existing files when the storage location changes.</summary>
/// <param name="Value">The transfer.</param>
/// <param name="Name">Shown name.</param>
internal sealed record StorageTransferChoice(StorageTransferKind Value, string Name)
{
    /// <summary>The choices.</summary>
    public static IReadOnlyList<StorageTransferChoice> All { get; } =
    [
        new(StorageTransferKind.Move, "Move the existing files"),
        new(StorageTransferKind.Copy, "Copy the existing files (keep the originals)"),
        new(StorageTransferKind.StartFresh, "Start fresh (leave the existing files where they are)"),
    ];

    /// <summary>The choice for <paramref name="value"/>.</summary>
    public static StorageTransferChoice Get(StorageTransferKind value) => All.First(choice => choice.Value == value);
}
