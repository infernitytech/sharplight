using CommunityToolkit.Mvvm.ComponentModel;

namespace Sharplight.Studio.ViewModels.Inference.Parameters;

/// <summary>
/// One parameter of an inference page: a field of the task's options type (for example the beam size of
/// <c>SpeechRecognitionOptions</c>) with its label, hint and a text form for presets. The parameter panel lists them and the page maps
/// their values to the options.
/// </summary>
internal abstract class ParameterViewModel : ObservableObject
{
    /// <param name="key">The stable name presets store the value under.</param>
    /// <param name="label">Shown next to the control.</param>
    /// <param name="hint">One line of explanation, or empty.</param>
    protected ParameterViewModel(string key, string label, string hint)
    {
        Key = key;
        Label = label;
        Hint = hint;
    }

    /// <summary>The stable name presets store the value under.</summary>
    public string Key { get; }

    /// <summary>Shown next to the control.</summary>
    public string Label { get; }

    /// <summary>One line of explanation, or empty.</summary>
    public string Hint { get; }

    /// <summary>Whether the parameter is shown (a page hides parameters the chosen model does not use).</summary>
    public bool IsVisibleNow
    {
        get;
        set => SetProperty(ref field, value);
    } = true;

    /// <summary>The value as invariant text (what a preset stores).</summary>
    public abstract string Serialize();

    /// <summary>Sets the value from the text <see cref="Serialize"/> produced; text that does not fit leaves the value as it is.</summary>
    /// <param name="text">The stored text.</param>
    public abstract void Deserialize(string text);

    /// <summary>Sets the default value.</summary>
    public abstract void Reset();
}
