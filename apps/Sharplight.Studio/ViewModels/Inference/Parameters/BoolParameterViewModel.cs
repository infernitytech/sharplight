using CommunityToolkit.Mvvm.ComponentModel;

namespace Sharplight.Studio.ViewModels.Inference.Parameters;

/// <summary>A yes/no parameter.</summary>
internal sealed partial class BoolParameterViewModel : ParameterViewModel
{
    private readonly bool _default;

    /// <param name="key">The stable name presets store the value under.</param>
    /// <param name="label">Shown next to the switch.</param>
    /// <param name="hint">One line of explanation, or empty.</param>
    /// <param name="defaultValue">The initial value.</param>
    public BoolParameterViewModel(string key, string label, string hint, bool defaultValue = false)
        : base(key, label, hint)
    {
        _default = defaultValue;
        Value = defaultValue;
    }

    /// <summary>The value.</summary>
    [ObservableProperty]
    public partial bool Value { get; set; }

    /// <inheritdoc/>
    public override string Serialize() => Value ? "true" : "false";

    /// <inheritdoc/>
    public override void Deserialize(string text)
    {
        if (bool.TryParse(text, out bool value))
        {
            Value = value;
        }
    }

    /// <inheritdoc/>
    public override void Reset() => Value = _default;
}
