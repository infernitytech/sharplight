using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Studio.Services.Pages;
using Sharplight.Studio.Services.Platform;

namespace Sharplight.Studio.ViewModels.Inference;

/// <summary>
/// The outputs gallery every inference page shares: the completed runs of the page with their output files and metadata, each with
/// open, export and delete. It follows the page's <see cref="PageArchive"/>.
/// </summary>
internal sealed partial class OutputGalleryViewModel : ObservableObject
{
    private readonly PageArchive _archive;
    private readonly IPlatformServices _platform;
    private readonly Action<RunRecord>? _reuse;

    /// <param name="archive">The page's runs.</param>
    /// <param name="platform">Opens and exports files.</param>
    /// <param name="reuse">Takes over a run's parameters and result (the "Reuse parameters" button); null for no button.</param>
    public OutputGalleryViewModel(PageArchive archive, IPlatformServices platform, Action<RunRecord>? reuse = null)
    {
        _archive = archive;
        _platform = platform;
        _reuse = reuse;
        archive.Changed += (_, _) => Refresh();
        Refresh();
    }

    /// <summary>The outputs, newest first.</summary>
    public ObservableCollection<OutputItemViewModel> Items { get; } = [];

    /// <summary>There is no output yet.</summary>
    public bool IsEmpty => Items.Count == 0;

    /// <summary>Lists the archive's completed runs that have files.</summary>
    public void Refresh()
    {
        Items.Clear();
        foreach (var record in _archive.Runs.Where(record => record.Outcome == RunOutcome.Completed && record.Files.Count > 0))
        {
            Items.Add(new OutputItemViewModel(record, _archive, _platform, Delete, _reuse));
        }

        OnPropertyChanged(nameof(IsEmpty));
    }

    private void Delete(OutputItemViewModel item) => _archive.Delete(item.Record);
}
