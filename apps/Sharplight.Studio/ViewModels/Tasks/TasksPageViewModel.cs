using System.Collections.ObjectModel;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;

namespace Sharplight.Studio.ViewModels.Tasks;

/// <summary>
/// The Tasks page: one card per model kind with its library and resident counts. It is the placeholder for the per-kind inference
/// pages. The chat page serves text generation; a kind the engine gains support for gets its page from the shared building blocks
/// (<see cref="Inference.ModelPickerViewModel"/> first) and its card opens it instead of the library.
/// </summary>
internal sealed class TasksPageViewModel : PageViewModel
{
    private readonly StudioServices _services;

    public TasksPageViewModel(StudioServices services, INavigator navigator)
    {
        _services = services;
        foreach (var info in ModelKindCatalog.All.Where(info => info.Id != ModelKindCatalog.Other))
        {
            Kinds.Add(new KindCardViewModel(info, navigator));
        }

        services.Runtime.ModelsChanged += (_, _) => services.Dispatcher.Post(UpdateResident);
    }

    /// <inheritdoc/>
    public override string Title => "Tasks";

    /// <inheritdoc/>
    public override string Icon => "IconSparkle";

    /// <summary>The kinds, runnable ones first.</summary>
    public ObservableCollection<KindCardViewModel> Kinds { get; } = [];

    /// <inheritdoc/>
    public override async Task ActivateAsync()
    {
        await RefreshAsync();
    }

    /// <summary>Counts the library and resident models of every kind.</summary>
    public async Task RefreshAsync()
    {
        try
        {
            var entries = await _services.Library.ListAsync(CancellationToken.None);
            foreach (var card in Kinds)
            {
                card.LibraryCount = entries.Count(entry => string.Equals(ModelKindCatalog.For(entry.Kind).Id, card.Info.Id, StringComparison.OrdinalIgnoreCase));
            }
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // The counts stay as they were.
        }

        UpdateResident();
        var ordered = Kinds.OrderByDescending(card => card.IsRunnable).ToList();
        if (!ordered.SequenceEqual(Kinds))
        {
            Kinds.Clear();
            foreach (var card in ordered)
            {
                Kinds.Add(card);
            }
        }
    }

    private void UpdateResident()
    {
        var loaded = _services.Runtime.LoadedModels;
        foreach (var card in Kinds)
        {
            card.ResidentCount = loaded.Count(model => string.Equals(model.Kind, card.Info.Id, StringComparison.OrdinalIgnoreCase));
        }
    }
}
