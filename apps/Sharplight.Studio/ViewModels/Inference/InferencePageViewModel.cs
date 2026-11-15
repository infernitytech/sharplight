using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Pages;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.ViewModels.Inference;

/// <summary>
/// The page of one model kind, built from the shared blocks: model picker, parameter panel (with presets), run queue, history and
/// outputs gallery. Every kind's page (transcribe, speak, and later image, vision, audio and video pages) derives from it, so they
/// look and behave the same; a page adds its input, its output view and the mapping of its run to the task client.
/// </summary>
internal abstract partial class InferencePageViewModel : PageViewModel
{
    private readonly INavigator _navigator;
    private readonly string _kind;

    /// <param name="services">The services.</param>
    /// <param name="navigator">Moves to the library and Discover.</param>
    /// <param name="kind">The model kind the page runs (<see cref="Sharplight.Models.ModelKinds"/>).</param>
    /// <param name="pageId">The page's stable id (the name of its storage directory).</param>
    /// <param name="parameters">The page's parameter panel.</param>
    protected InferencePageViewModel(StudioServices services, INavigator navigator, string kind, string pageId, ParameterPanelViewModel parameters)
    {
        Services = services;
        _navigator = navigator;
        _kind = kind;
        PageId = pageId;
        Parameters = parameters;
        Picker = new Inference.ModelPickerViewModel(services, kind);
        Queue = new RunQueueViewModel(services.Dispatcher);
        Archive = new PageArchive(services.Pages, pageId);
        History = new RunHistoryViewModel(Archive, Restore);
        Gallery = new OutputGalleryViewModel(Archive, services.Platform);
        Picker.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(ModelPickerViewModel.SelectedModel))
            {
                OnPropertyChanged(nameof(SelectedModel));
                OnPropertyChanged(nameof(HasModel));
                OnModelChanged(SelectedModel);
            }
            else if (args.PropertyName == nameof(ModelPickerViewModel.HasModels))
            {
                OnPropertyChanged(nameof(HasModels));
            }
        };
        Queue.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(RunQueueViewModel.IsBusy))
            {
                OnBusyChanged();
            }
        };
    }

    /// <summary>The services.</summary>
    protected StudioServices Services { get; }

    /// <summary>The page's stable id.</summary>
    public string PageId { get; }

    /// <summary>The model picker (resident models of the page's kind).</summary>
    public ModelPickerViewModel Picker { get; }

    /// <summary>The parameters of the page's task options, with presets.</summary>
    public ParameterPanelViewModel Parameters { get; }

    /// <summary>The queued and running runs.</summary>
    public RunQueueViewModel Queue { get; }

    /// <summary>The runs the page kept.</summary>
    public RunHistoryViewModel History { get; }

    /// <summary>The outputs the page kept.</summary>
    public OutputGalleryViewModel Gallery { get; }

    /// <summary>The persisted runs behind the history and the gallery.</summary>
    protected PageArchive Archive { get; }

    /// <summary>The model chosen in the picker, or null.</summary>
    public ILoadedModel? SelectedModel => Picker.SelectedModel?.Model;

    /// <summary>A model is chosen.</summary>
    public bool HasModel => SelectedModel is not null;

    /// <summary>At least one model of the kind is resident.</summary>
    public bool HasModels => Picker.HasModels;

    /// <summary>What the page runs, for the hint shown while no model is loaded (for example <c>speech to text</c>).</summary>
    public abstract string KindNoun { get; }

    /// <summary>The page's main panel is bound to this: text for the user (a failed run, a missing file), empty when all is well.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasNotice))]
    public partial string Notice { get; set; } = "";

    /// <summary>There is a notice.</summary>
    public bool HasNotice => Notice.Length > 0;

    /// <summary>Which side tab is shown: 0 parameters, 1 history, 2 outputs.</summary>
    [ObservableProperty]
    public partial int SideTab { get; set; }

    /// <inheritdoc/>
    public override Task ActivateAsync()
    {
        Picker.Refresh();
        return Task.CompletedTask;
    }

    /// <summary>Chooses <paramref name="model"/> in the picker.</summary>
    /// <param name="model">The model, or null to keep the choice.</param>
    public void SelectModel(ILoadedModel? model) => Picker.Select(model);

    /// <summary>Shows a run of the history again: its parameter values and its result.</summary>
    /// <param name="record">The run.</param>
    protected abstract void Restore(RunRecord record);

    /// <summary>Called when the chosen model changed (a page shows what the model offers).</summary>
    /// <param name="model">The new model, or null.</param>
    protected virtual void OnModelChanged(ILoadedModel? model)
    {
    }

    /// <summary>Called when the queue became busy or idle.</summary>
    protected virtual void OnBusyChanged()
    {
    }

    /// <summary>Records a finished run in the history (and the gallery when it produced files).</summary>
    /// <param name="record">The run.</param>
    /// <param name="files">Its output files by name.</param>
    protected void Archived(RunRecord record, IReadOnlyDictionary<string, byte[]>? files = null)
    {
        Services.Dispatcher.Post(() => Archive.Add(record, files));
    }

    [RelayCommand]
    private void OpenLibrary() => _navigator.OpenLibrary(_kind);

    [RelayCommand]
    private void OpenDiscover()
    {
        var info = ModelKindCatalog.For(_kind);
        if (info.Tasks.Count > 0)
        {
            _navigator.OpenDiscover(info.Tasks[0]);
        }
        else
        {
            _navigator.OpenDiscover();
        }
    }
}
