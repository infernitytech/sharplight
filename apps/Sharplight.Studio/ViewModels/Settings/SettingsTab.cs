namespace Sharplight.Studio.ViewModels.Settings;

/// <summary>The tabs of the settings page, in display order (the value is the tab index).</summary>
internal enum SettingsTab
{
    /// <summary>Appearance, audio devices and new chat defaults.</summary>
    General = 0,

    /// <summary>Where everything Sharplight writes is kept.</summary>
    Storage = 1,

    /// <summary>Model store, Hugging Face token and download behaviour.</summary>
    Models = 2,

    /// <summary>Load defaults, residency and engine features.</summary>
    Engine = 3,

    /// <summary>Version, channel and update checks.</summary>
    Updates = 4,

    /// <summary>Statistics data and the about text.</summary>
    DataAndAbout = 5,
}
