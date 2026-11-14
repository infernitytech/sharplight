using CommunityToolkit.Mvvm.ComponentModel;

namespace Sharplight.Studio.ViewModels.Settings;

/// <summary>A device feature the user can switch off.</summary>
internal sealed partial class FeatureToggleViewModel : ObservableObject
{
    private readonly Action _changed;

    public FeatureToggleViewModel(string name, string description, string detectedOn, bool enabled, Action changed)
    {
        Name = name;
        Description = description;
        DetectedOn = detectedOn;
        _changed = changed;
        IsEnabled = enabled;
    }

    /// <summary>Provider-neutral feature name.</summary>
    public string Name { get; }

    /// <summary>What it is.</summary>
    public string Description { get; }

    /// <summary>The devices that have it, or "not detected".</summary>
    public string DetectedOn { get; }

    /// <summary>Allowed (not in the disable list).</summary>
    [ObservableProperty]
    public partial bool IsEnabled { get; set; }

    partial void OnIsEnabledChanged(bool value) => _changed();
}
