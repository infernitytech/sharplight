using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Tasks;

/// <summary>One model kind on the Tasks page: what it is, whether the engine runs it, and how many models of it there are.</summary>
internal sealed partial class KindCardViewModel : ObservableObject
{
    private readonly INavigator _navigator;

    /// <param name="info">The kind.</param>
    /// <param name="navigator">Moves to the library, Discover or the chat.</param>
    public KindCardViewModel(ModelKindInfo info, INavigator navigator)
    {
        Info = info;
        _navigator = navigator;
    }

    /// <summary>The kind.</summary>
    public ModelKindInfo Info { get; }

    /// <summary>The kind's name.</summary>
    public string Title => Info.Title;

    /// <summary>The kind's icon resource key.</summary>
    public string Icon => Info.Icon;

    /// <summary>The engine runs models of this kind.</summary>
    public bool IsRunnable => Info.Runnable;

    /// <summary>The engine cannot run this kind yet; its page comes with the engine's support.</summary>
    public bool IsPlanned => !Info.Runnable;

    /// <summary>Models of this kind in the library.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(Summary))]
    public partial int LibraryCount { get; set; }

    /// <summary>Models of this kind that are resident.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(Summary))]
    public partial int ResidentCount { get; set; }

    /// <summary>The state of the kind in one line: <c>2 in library, 1 resident</c>, or why nothing can be run.</summary>
    public string Summary
    {
        get
        {
            string models = LibraryCount == 0 ? "None in the library" : $"{LibraryCount} in library";
            if (ResidentCount > 0)
            {
                models += $", {ResidentCount} resident";
            }

            return Info.Runnable ? models : $"{models}. Not yet supported";
        }
    }

    /// <summary>The text of the button that opens the kind's page.</summary>
    public string OpenText => Info.Id switch
    {
        Sharplight.Models.ModelKinds.TextGeneration => "Open chat",
        Sharplight.Models.ModelKinds.SpeechRecognition => "Open transcribe",
        Sharplight.Models.ModelKinds.SpeechSynthesis => "Open speak",
        Sharplight.Models.ModelKinds.ImageGeneration => "Open image",
        ModelKindCatalog.Embeddings => "Open search",
        Sharplight.Models.ModelKinds.ImageClassification or Sharplight.Models.ModelKinds.ImageEmbedding or Sharplight.Models.ModelKinds.ObjectDetection
            or Sharplight.Models.ModelKinds.DepthEstimation or Sharplight.Models.ModelKinds.ImageSegmentation or Sharplight.Models.ModelKinds.SemanticSegmentation or Sharplight.Models.ModelKinds.Ocr
            or Sharplight.Models.ModelKinds.ImageUpscaling => "Open vision",
        _ => "Open library",
    };

    /// <summary>Opens the page of the kind (chat, transcribe, speak, image or vision), or the filtered library for a kind without a page.</summary>
    [CommunityToolkit.Mvvm.Input.RelayCommand]
    private void Open()
    {
        if (!_navigator.OpenKind(Info.Id, null))
        {
            _navigator.OpenLibrary(Info.Id);
        }
    }

    /// <summary>Shows the library models of this kind.</summary>
    [CommunityToolkit.Mvvm.Input.RelayCommand]
    private void ShowLibrary() => _navigator.OpenLibrary(Info.Id);

    /// <summary>Browses this kind's models on Hugging Face.</summary>
    [CommunityToolkit.Mvvm.Input.RelayCommand]
    private void Browse()
    {
        if (Info.Tasks.Count > 0)
        {
            _navigator.OpenDiscover(Info.Tasks[0]);
        }
    }
}
