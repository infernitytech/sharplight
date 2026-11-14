using Sharplight.Hub;

namespace Sharplight.Studio.ViewModels.Settings;

/// <summary>A model store location in the picker.</summary>
/// <param name="Value">The location.</param>
/// <param name="Name">Shown name.</param>
/// <param name="Description">One line on where that is.</param>
internal sealed record StoreLocationChoice(ModelStoreLocation Value, string Name, string Description)
{
    /// <summary>The choices.</summary>
    public static IReadOnlyList<StoreLocationChoice> All { get; } =
    [
        new(ModelStoreLocation.Application, "Application folder",
            $"The '{ModelStore.ApplicationDirectoryName}' folder next to the app; the user folder when the app is installed read-only."),
        new(ModelStoreLocation.User, "User folder", $"Shared by every Sharplight tool of this user: {ModelStore.UserRoot}"),
        new(ModelStoreLocation.Custom, "Custom folder", "A folder of your choice."),
    ];

    /// <summary>The choice for <paramref name="value"/>.</summary>
    public static StoreLocationChoice Get(ModelStoreLocation value) => All.First(choice => choice.Value == value);
}
