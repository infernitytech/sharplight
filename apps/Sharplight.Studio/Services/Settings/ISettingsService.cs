using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Settings;

/// <summary>The current settings, changes to them, and where the model store is.</summary>
internal interface ISettingsService
{
    /// <summary>The settings now.</summary>
    public AppSettings Current { get; }

    /// <summary>The model store directory in effect.</summary>
    public string StoreRoot { get; }

    /// <summary>Why <see cref="StoreRoot"/> is what it is (command line, environment, settings, default, not-writable fallback).</summary>
    public string StoreRootSource { get; }

    /// <summary>
    /// A store that still holds models while <see cref="StoreRoot"/> has none and was not given by <c>--models-dir</c> or
    /// <c>SHARPLIGHT_MODELS</c> (the per-user store of earlier versions); null otherwise.
    /// </summary>
    public string? FindPreviousStore();

    /// <summary>Raised after the settings changed.</summary>
    public event EventHandler? Changed;

    /// <summary>Applies <paramref name="change"/> and saves.</summary>
    public void Update(Func<AppSettings, AppSettings> change);
}
