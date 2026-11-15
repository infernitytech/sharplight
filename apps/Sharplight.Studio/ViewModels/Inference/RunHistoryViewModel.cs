using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services.Pages;

namespace Sharplight.Studio.ViewModels.Inference;

/// <summary>
/// The history every inference page shares: its runs, newest first, persisted per page (<c>history.json</c>). A run can be restored
/// (the page shows its result and options again) or deleted.
/// </summary>
internal sealed partial class RunHistoryViewModel : ObservableObject
{
    private readonly PageArchive _archive;
    private readonly Action<RunRecord> _restore;

    /// <param name="archive">The page's runs.</param>
    /// <param name="restore">Shows a run's result and options on the page.</param>
    public RunHistoryViewModel(PageArchive archive, Action<RunRecord> restore)
    {
        _archive = archive;
        _restore = restore;
        archive.Changed += (_, _) => Refresh();
        Refresh();
    }

    /// <summary>The runs, newest first.</summary>
    public ObservableCollection<HistoryItemViewModel> Items { get; } = [];

    /// <summary>There is no run yet.</summary>
    public bool IsEmpty => Items.Count == 0;

    /// <summary>Lists the archive's runs again.</summary>
    public void Refresh()
    {
        Items.Clear();
        foreach (var record in _archive.Runs)
        {
            Items.Add(new HistoryItemViewModel(record));
        }

        OnPropertyChanged(nameof(IsEmpty));
    }

    [RelayCommand]
    private void Restore(HistoryItemViewModel item) => _restore(item.Record);

    [RelayCommand]
    private void Delete(HistoryItemViewModel item)
    {
        if (!item.ConfirmDelete)
        {
            item.ConfirmDelete = true;
            return;
        }

        _archive.Delete(item.Record);
    }

    /// <summary>Removes every run and its output files.</summary>
    [RelayCommand]
    private void Clear() => _archive.Clear();
}
