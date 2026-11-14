using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Settings;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>In-memory settings.</summary>
internal sealed class FakeSettings(AppSettings? initial = null, string storeRoot = "/models") : ISettingsService
{
    public AppSettings Current { get; private set; } = initial ?? new AppSettings();

    public string StoreRoot { get; set; } = storeRoot;

    public string StoreRootSource => "Test";

    public int Updates { get; private set; }

    public event EventHandler? Changed;

    public void Update(Func<AppSettings, AppSettings> change)
    {
        Current = change(Current);
        Updates++;
        Changed?.Invoke(this, EventArgs.Empty);
    }
}
