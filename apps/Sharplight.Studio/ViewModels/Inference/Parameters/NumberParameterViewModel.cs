using System.Globalization;
using CommunityToolkit.Mvvm.ComponentModel;

namespace Sharplight.Studio.ViewModels.Inference.Parameters;

/// <summary>A numeric parameter chosen with a slider.</summary>
internal sealed partial class NumberParameterViewModel : ParameterViewModel
{
    private readonly double _default;

    /// <param name="key">The stable name presets store the value under.</param>
    /// <param name="label">Shown next to the slider.</param>
    /// <param name="hint">One line of explanation, or empty.</param>
    /// <param name="minimum">The lowest value.</param>
    /// <param name="maximum">The highest value.</param>
    /// <param name="step">The step of the slider; 1 makes the parameter an integer.</param>
    /// <param name="defaultValue">The initial value.</param>
    public NumberParameterViewModel(string key, string label, string hint, double minimum, double maximum, double step, double defaultValue)
        : base(key, label, hint)
    {
        Minimum = minimum;
        Maximum = maximum;
        Step = step;
        _default = defaultValue;
        Value = defaultValue;
    }

    /// <summary>The lowest value.</summary>
    public double Minimum { get; }

    /// <summary>The highest value.</summary>
    public double Maximum { get; }

    /// <summary>The step of the slider.</summary>
    public double Step { get; }

    /// <summary>The number format of the shown value.</summary>
    public string Format => Step >= 1 ? "0" : Step >= 0.1 ? "0.0" : "0.00";

    /// <summary>The value.</summary>
    [ObservableProperty]
    public partial double Value { get; set; }

    /// <summary>The value as an integer.</summary>
    public int IntValue => (int)Math.Round(Value);

    /// <inheritdoc/>
    public override string Serialize() => Value.ToString("R", CultureInfo.InvariantCulture);

    /// <inheritdoc/>
    public override void Deserialize(string text)
    {
        if (double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double value))
        {
            Value = Math.Clamp(value, Minimum, Maximum);
        }
    }

    /// <inheritdoc/>
    public override void Reset() => Value = _default;
}
