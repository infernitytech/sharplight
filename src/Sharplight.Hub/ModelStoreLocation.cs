namespace Sharplight.Hub;

/// <summary>Where a model store lives (see <see cref="ModelStoreSettings"/> and <see cref="ModelStoreResolution"/>).</summary>
public enum ModelStoreLocation
{
    /// <summary>
    /// With the other data: the <c>models</c> folder of the storage layout in effect (portable, system or custom; see
    /// <see cref="Sharplight.Storage.StorageLayout.ModelsDirectory"/>). The default. (The name is kept from earlier versions, where
    /// it meant the folder next to the executable, so existing settings files stay readable.)
    /// </summary>
    Application,

    /// <summary>The model store of the system folders (<see cref="ModelStore.SystemRoot"/>), whatever the storage layout is.</summary>
    User,

    /// <summary>A directory chosen by the user (settings, command line or <c>SHARPLIGHT_MODELS</c>).</summary>
    Custom,
}
