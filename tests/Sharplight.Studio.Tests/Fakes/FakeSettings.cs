using Sharplight.Hub;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Settings;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>
/// In-memory settings. The store follows the settings like the real service: the application folder is
/// <c>&lt;storeRoot&gt;</c>, the system folder <see cref="UserRoot"/>, a custom folder its path.
/// </summary>
internal sealed class FakeSettings(AppSettings? initial = null, string storeRoot = "/models") : ISettingsService
{
    public AppSettings Current { get; private set; } = initial ?? new AppSettings();

    /// <summary>The application folder store.</summary>
    public string ApplicationRoot { get; set; } = storeRoot;

    /// <summary>The user folder store.</summary>
    public string UserRoot { get; set; } = "/user-models";

    /// <summary>What <see cref="FindPreviousStore"/> returns.</summary>
    public string? PreviousStore { get; set; }

    public ModelStoreResolution Store => Resolve();

    public string StoreRoot => Resolve().Root;

    public string StoreRootSource => Resolve().Description;

    public int Updates { get; private set; }

    public event EventHandler? Changed;

    public string? FindPreviousStore() => PreviousStore;

    public string? RelocatedTo { get; private set; }

    public void RelocateTo(string settingsFile) => RelocatedTo = settingsFile;

    public void Update(Func<AppSettings, AppSettings> change)
    {
        Current = change(Current);
        Updates++;
        Changed?.Invoke(this, EventArgs.Empty);
    }

    private ModelStoreResolution Resolve()
    {
        var settings = StoreLocation.ToStoreSettings(Current);
        return settings switch
        {
            null => new ModelStoreResolution(ApplicationRoot, ModelStoreSource.Default, ModelStoreLocation.Application),
            { Location: ModelStoreLocation.User } => new ModelStoreResolution(UserRoot, ModelStoreSource.Settings, ModelStoreLocation.User),
            { Location: ModelStoreLocation.Custom, CustomPath: { } path } => new ModelStoreResolution(path, ModelStoreSource.Settings, ModelStoreLocation.Custom),
            _ => new ModelStoreResolution(ApplicationRoot, ModelStoreSource.Settings, ModelStoreLocation.Application),
        };
    }
}
