namespace Sharplight.Hub;

/// <summary>Where a model store lives (see <see cref="ModelStoreSettings"/> and <see cref="ModelStoreResolution"/>).</summary>
public enum ModelStoreLocation
{
    /// <summary>
    /// The <c>data</c> folder in the application directory (<see cref="ModelStore.ApplicationRoot"/>), created on first use. The
    /// default; when that folder is not writable the per-user store is used instead.
    /// </summary>
    Application,

    /// <summary>The per-user store shared by every Sharplight executable of the user (<see cref="ModelStore.UserRoot"/>).</summary>
    User,

    /// <summary>A directory chosen by the user (settings, command line or <c>SHARPLIGHT_MODELS</c>).</summary>
    Custom,
}
