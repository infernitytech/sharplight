using CommunityToolkit.Mvvm.ComponentModel;

namespace Sharplight.Studio.ViewModels.Inference.Parameters;

/// <summary>A parameter chosen from a list.</summary>
internal sealed partial class ChoiceParameterViewModel : ParameterViewModel
{
    private readonly string _default;

    /// <param name="key">The stable name presets store the value under.</param>
    /// <param name="label">Shown next to the list.</param>
    /// <param name="hint">One line of explanation, or empty.</param>
    /// <param name="choices">The options; the first is the default unless <paramref name="defaultValue"/> names another.</param>
    /// <param name="defaultValue">The <see cref="ParameterChoice.Value"/> chosen initially, or null for the first option.</param>
    public ChoiceParameterViewModel(string key, string label, string hint, IReadOnlyList<ParameterChoice> choices, string? defaultValue = null)
        : base(key, label, hint)
    {
        Choices = choices;
        _default = defaultValue ?? choices[0].Value;
        Selected = Find(_default);
    }

    /// <summary>The options.</summary>
    public IReadOnlyList<ParameterChoice> Choices { get; private set; }

    /// <summary>Replaces the options (a page lists what the chosen model offers, such as its voices); keeps the choice when it is still offered.</summary>
    /// <param name="choices">The new options; at least one.</param>
    public void SetChoices(IReadOnlyList<ParameterChoice> choices)
    {
        string current = Selected.Value;
        Choices = choices;
        OnPropertyChanged(nameof(Choices));
        Selected = choices.FirstOrDefault(choice => choice.Value == current) ?? Find(_default);
    }

    /// <summary>The chosen option.</summary>
    [ObservableProperty]
    public partial ParameterChoice Selected { get; set; }

    /// <summary>The value of the chosen option.</summary>
    public string Value => Selected.Value;

    /// <inheritdoc/>
    public override string Serialize() => Selected.Value;

    /// <inheritdoc/>
    public override void Deserialize(string text)
    {
        if (Choices.FirstOrDefault(choice => choice.Value == text) is { } choice)
        {
            Selected = choice;
        }
    }

    /// <inheritdoc/>
    public override void Reset() => Selected = Find(_default);

    private ParameterChoice Find(string value) => Choices.FirstOrDefault(choice => choice.Value == value) ?? Choices[0];
}
