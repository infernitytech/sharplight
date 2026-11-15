using CommunityToolkit.Mvvm.ComponentModel;

namespace Sharplight.Studio.ViewModels.Inference.Parameters;

/// <summary>A free-text parameter.</summary>
internal sealed partial class TextParameterViewModel : ParameterViewModel
{
    private readonly string _default;

    /// <param name="key">The stable name presets store the value under.</param>
    /// <param name="label">Shown above the box.</param>
    /// <param name="hint">One line of explanation, or empty.</param>
    /// <param name="placeholder">Shown while the box is empty.</param>
    /// <param name="defaultValue">The initial text.</param>
    public TextParameterViewModel(string key, string label, string hint, string placeholder = "", string defaultValue = "")
        : base(key, label, hint)
    {
        Placeholder = placeholder;
        _default = defaultValue;
        Value = defaultValue;
    }

    /// <summary>Shown while the box is empty.</summary>
    public string Placeholder { get; }

    /// <summary>The text.</summary>
    [ObservableProperty]
    public partial string Value { get; set; }

    /// <inheritdoc/>
    public override string Serialize() => Value;

    /// <inheritdoc/>
    public override void Deserialize(string text) => Value = text;

    /// <inheritdoc/>
    public override void Reset() => Value = _default;
}
