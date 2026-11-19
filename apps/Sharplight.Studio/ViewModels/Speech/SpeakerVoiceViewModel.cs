using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Studio.ViewModels.Inference.Parameters;

namespace Sharplight.Studio.ViewModels.Speech;

/// <summary>The voice of one speaker of a script: the model's own voice, a built-in voice, a saved voice or a reference recording.</summary>
internal sealed partial class SpeakerVoiceViewModel : ObservableObject
{
    /// <summary>The key of the choice "the model's own voice".</summary>
    public const string Own = "";

    /// <summary>The key of the choice "a reference recording".</summary>
    public const string ReferenceFile = "file";

    internal SpeakerVoiceViewModel(string speaker)
    {
        Speaker = speaker;
        Choices = [new ParameterChoice(Own, "Model's own voice")];
        Selected = Choices[0];
    }

    /// <summary>The label as written in the script (<c>1</c> for <c>Speaker 1:</c>).</summary>
    public string Speaker { get; }

    /// <summary>The name shown.</summary>
    public string Name => $"Speaker {Speaker}";

    /// <summary>The voices to choose from.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<ParameterChoice> Choices { get; private set; }

    /// <summary>The chosen voice.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(UsesReference))]
    public partial ParameterChoice Selected { get; set; }

    /// <summary>The reference recording asks for a file.</summary>
    public bool UsesReference => Selected.Value == ReferenceFile;

    /// <summary>The path of the reference recording (a WAV file).</summary>
    [ObservableProperty]
    public partial string ReferencePath { get; set; } = "";

    /// <summary>The words spoken in the reference recording, for models that need them.</summary>
    [ObservableProperty]
    public partial string ReferenceText { get; set; } = "";

    /// <summary>Replaces the choices (built-in voices, saved voices, the reference recording) and keeps the selection when it is still offered.</summary>
    /// <param name="builtIn">The voices the model offers.</param>
    /// <param name="saved">The saved voices of the model's family.</param>
    /// <param name="canReference">The model can make a voice from a recording.</param>
    internal void SetChoices(IEnumerable<string> builtIn, IEnumerable<string> saved, bool canReference)
    {
        var choices = new List<ParameterChoice> { new(Own, "Model's own voice") };
        choices.AddRange(builtIn.Select(voice => new ParameterChoice("voice:" + voice, "Voice " + voice)));
        choices.AddRange(saved.Select(name => new ParameterChoice("saved:" + name, "Saved: " + name)));
        if (canReference)
        {
            choices.Add(new ParameterChoice(ReferenceFile, "Reference recording"));
        }

        string current = Selected.Value;
        Choices = choices;
        Selected = choices.FirstOrDefault(choice => choice.Value == current) ?? choices[0];
    }
}
