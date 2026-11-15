using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.ViewModels.Inference;

/// <summary>
/// The model picker every inference page shares: the resident models of one kind (see <see cref="ILoadedModel.Kind"/>) and the
/// chosen one. It follows loads and unloads, and keeps the choice while that model stays resident. The chat page and the per-kind
/// pages use this one component, so models of every kind are chosen, loaded and unloaded through the same residency mechanism.
/// </summary>
internal sealed partial class ModelPickerViewModel : ObservableObject
{
    private readonly StudioServices _services;

    /// <param name="services">The runtime that lists the resident models.</param>
    /// <param name="kind">The model kind the picker offers (<see cref="Sharplight.Models.ModelKinds"/>).</param>
    public ModelPickerViewModel(StudioServices services, string kind)
        : this(services, [kind])
    {
    }

    /// <param name="services">The runtime that lists the resident models.</param>
    /// <param name="kinds">The model kinds the picker offers; the first is the primary one (a page that runs several kinds, such as vision).</param>
    public ModelPickerViewModel(StudioServices services, IReadOnlyList<string> kinds)
    {
        _services = services;
        Kinds = kinds;
        services.Runtime.ModelsChanged += (_, _) => services.Dispatcher.Post(Refresh);
        Refresh();
    }

    /// <summary>The model kinds the picker offers.</summary>
    public IReadOnlyList<string> Kinds { get; }

    /// <summary>The primary model kind.</summary>
    public string Kind => Kinds[0];

    /// <summary>The resident models of the kind.</summary>
    public ObservableCollection<ModelOptionViewModel> Models { get; } = [];

    /// <summary>The chosen model.</summary>
    [ObservableProperty]
    public partial ModelOptionViewModel? SelectedModel { get; set; }

    /// <summary>At least one model of the kind is resident.</summary>
    public bool HasModels => Models.Count > 0;

    /// <summary>Lists the resident models of the kind again, keeping the chosen one while it is resident.</summary>
    public void Refresh()
    {
        var selectedId = SelectedModel?.Id;
        Models.Clear();
        foreach (var model in _services.Runtime.LoadedModels.Where(model => Kinds.Contains(model.Kind, StringComparer.OrdinalIgnoreCase)))
        {
            Models.Add(new ModelOptionViewModel(model));
        }

        SelectedModel = Models.FirstOrDefault(option => option.Id == selectedId) ?? Models.FirstOrDefault();
        OnPropertyChanged(nameof(HasModels));
    }

    /// <summary>Chooses <paramref name="model"/> when it is resident and of the kind; otherwise keeps the choice.</summary>
    /// <param name="model">The model to choose, or null to only refresh the list.</param>
    public void Select(ILoadedModel? model)
    {
        Refresh();
        if (model is not null && Models.FirstOrDefault(option => option.Id == model.Id) is { } option)
        {
            SelectedModel = option;
        }
    }
}
