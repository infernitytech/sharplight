namespace Sharplight.Hub;

/// <summary>What decided the model store directory (see <see cref="ModelStoreResolution"/>).</summary>
public enum ModelStoreSource
{
    /// <summary>The <c>--models-dir</c> option.</summary>
    CommandLine,

    /// <summary>The <c>SHARPLIGHT_MODELS</c> environment variable (<see cref="ModelStore.EnvironmentVariable"/>).</summary>
    EnvironmentVariable,

    /// <summary>The application's settings (<see cref="ModelStoreSettings"/>).</summary>
    Settings,

    /// <summary>Nothing was configured: the application folder.</summary>
    Default,
}
