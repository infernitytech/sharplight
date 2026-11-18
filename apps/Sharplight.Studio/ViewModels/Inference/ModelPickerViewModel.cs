using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Studio.Models;
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
    private readonly object _refreshLock = new();
    private ResidentState? _selectedState;
    private bool _refreshing;
    private bool _refreshAgain;

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

    /// <summary>Raised on the UI thread when the chosen model's state changed (it finished loading, went to sleep, ...) without a new choice.</summary>
    public event EventHandler? SelectedModelStateChanged;

    /// <summary>
    /// The model to choose when none is chosen yet or the chosen one is gone (a remembered choice that is not resident yet); the first
    /// resident model when it is not resident either.
    /// </summary>
    public string? PreferredId { get; set; }

    /// <summary>True while the list is being re-read: a change of <see cref="SelectedModel"/> then is the picker's own, not the user's choice.</summary>
    public bool IsRefreshing => _refreshing;

    /// <summary>The primary model kind.</summary>
    public string Kind => Kinds[0];

    /// <summary>The resident models of the kind.</summary>
    public ObservableCollection<ModelOptionViewModel> Models { get; } = [];

    /// <summary>The chosen model.</summary>
    [ObservableProperty]
    public partial ModelOptionViewModel? SelectedModel { get; set; }

    partial void OnSelectedModelChanged(ModelOptionViewModel? value) => _selectedState = value?.Model.State;

    /// <summary>At least one model of the kind is resident.</summary>
    public bool HasModels => Models.Count > 0;

    /// <summary>
    /// Re-reads the resident models. The options of models that stay resident are kept (the same instances), so the choice does not
    /// change when only a state changes (a run starts or ends, a model loads): a new option for the chosen model would make the page
    /// treat it as a new choice on every lease, and any page that reads the model on a new choice would trigger the next refresh.
    /// </summary>
    public void Refresh()
    {
        // Serialized: the runtime raises ModelsChanged from engine threads (after a load, unload or lease), and a dispatcher that
        // runs inline off the UI thread would then edit the collection while another refresh enumerates it. A refresh requested from
        // inside a refresh (a handler of the collection or of the choice reaching the runtime) runs after the current one, because
        // editing the collection while its change notifications are being delivered is what throws "Collection was modified".
        lock (_refreshLock)
        {
            if (_refreshing)
            {
                _refreshAgain = true;
                return;
            }

            _refreshing = true;
            try
            {
                do
                {
                    _refreshAgain = false;
                    RefreshCore();
                }
                while (_refreshAgain);
            }
            finally
            {
                _refreshing = false;
            }
        }
    }

    private void RefreshCore()
    {
        var current = _services.Runtime.LoadedModels.Where(model => Kinds.Contains(model.Kind, StringComparer.OrdinalIgnoreCase)).ToList();
        if (!Models.Select(option => option.Model).SequenceEqual(current))
        {
            var selectedId = SelectedModel?.Id;
            var existing = Models.ToDictionary(option => option.Model);
            Models.Clear();
            foreach (var model in current)
            {
                Models.Add(existing.TryGetValue(model, out var option) ? option : new ModelOptionViewModel(model));
            }

            SelectedModel = Models.FirstOrDefault(option => option.Id == selectedId) ?? Models.FirstOrDefault(option => option.Id == PreferredId) ?? Models.FirstOrDefault();
            OnPropertyChanged(nameof(HasModels));
        }

        var state = SelectedModel?.Model.State;
        if (state != _selectedState)
        {
            _selectedState = state;
            SelectedModelStateChanged?.Invoke(this, EventArgs.Empty);
        }
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
