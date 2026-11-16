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
        new(ModelStoreLocation.Application, "With the other data",
            "The 'models' folder of the storage location below (portable, system or custom); moves along when the location changes."),
        new(ModelStoreLocation.User, "System folder", $"The system store, shared by every Sharplight tool of this user: {ModelStore.SystemRoot}"),
        new(ModelStoreLocation.Custom, "Custom folder", "A folder of your choice."),
    ];

    /// <summary>The choice for <paramref name="value"/>.</summary>
    public static StoreLocationChoice Get(ModelStoreLocation value) => All.First(choice => choice.Value == value);
}
