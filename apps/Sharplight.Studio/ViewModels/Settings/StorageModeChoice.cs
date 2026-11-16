using Sharplight.Storage;

namespace Sharplight.Studio.ViewModels.Settings;

/// <summary>A storage mode in the picker.</summary>
/// <param name="Value">The mode.</param>
/// <param name="Name">Shown name.</param>
/// <param name="Description">One line on where that is.</param>
internal sealed record StorageModeChoice(StorageMode Value, string Name, string Description)
{
    /// <summary>The choices.</summary>
    public static IReadOnlyList<StorageModeChoice> All { get; } =
    [
        new(StorageMode.Portable, "Portable",
            $"Everything in a '{StorageLayout.PortableDirectoryName}' folder next to the app (the AppImage, the install folder or the app bundle), so it can be moved or carried along."),
        new(StorageMode.System, "System folders", "The platform's per-user folders (XDG folders on Linux, AppData on Windows, ~/Library on macOS)."),
        new(StorageMode.Custom, "Custom folder", "Everything in one folder of your choice, e.g. on a larger disk."),
    ];

    /// <summary>The choice for <paramref name="value"/>.</summary>
    public static StorageModeChoice Get(StorageMode value) => All.First(choice => choice.Value == value);
}
