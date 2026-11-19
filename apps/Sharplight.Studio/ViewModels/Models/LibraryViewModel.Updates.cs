using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Models;

// Updating every feed package that has a newer version, one after the other, with the sizes, overall progress and the atomic swap of a single update.
internal sealed partial class LibraryViewModel
{
    private CancellationTokenSource? _updateAllCancellation;

    /// <summary>Raised on the UI thread when the number of models with a newer feed version may have changed (the Models badge follows it).</summary>
    public event EventHandler? UpdatesChanged;

    /// <summary>How many library models have a newer version in the feed.</summary>
    public int UpdateCount => Entries.Count(entry => entry.HasFeedUpdate);

    /// <summary>At least one model has a newer version.</summary>
    public bool HasUpdates => UpdateCount > 0;

    /// <summary>The download size of all newer versions together.</summary>
    public long UpdateBytes => Entries.Where(entry => entry.HasFeedUpdate).Sum(PackageBytes);

    /// <summary><c>Update all (3 · 4.2 GB)</c>.</summary>
    public string UpdateAllText => UpdateCount == 0 ? "Update all" : $"Update all ({UpdateCount} · {ByteFormat.Format(UpdateBytes)})";

    /// <summary>The update of all models is running.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(UpdateAllCommand))]
    public partial bool IsUpdatingAll { get; private set; }

    /// <summary>Overall progress of the running update of all models, 0..1 (weighted by download size).</summary>
    [ObservableProperty]
    public partial double UpdateAllProgress { get; private set; }

    /// <summary><c>Updating 2 of 3: Qwen3 0.6B · Downloading …</c>.</summary>
    [ObservableProperty]
    public partial string UpdateAllStatus { get; private set; } = "";

    /// <summary>Tells the page that the number of models with a newer version changed.</summary>
    public void NotifyUpdatesChanged()
    {
        OnPropertyChanged(nameof(UpdateCount));
        OnPropertyChanged(nameof(HasUpdates));
        OnPropertyChanged(nameof(UpdateBytes));
        OnPropertyChanged(nameof(UpdateAllText));
        UpdateAllCommand.NotifyCanExecuteChanged();
        UpdatesChanged?.Invoke(this, EventArgs.Empty);
    }

    private long PackageBytes(LibraryEntryViewModel entry)
        => entry.Entry.Manifest?.FeedPackage is { } id ? Feed?.AllPackages.FirstOrDefault(package => package.Id == id)?.Summary.Size ?? 0 : 0;

    private bool CanUpdateAll() => !IsUpdatingAll && Feed is not null && Entries.Any(entry => entry.CanUpdateFromFeed);

    /// <summary>
    /// Updates every model with a newer feed version, one after the other. Each update downloads and verifies the new version next to the
    /// old one and replaces the file in one step, so a failed or cancelled update leaves that model as it was; the others continue.
    /// </summary>
    [RelayCommand(CanExecute = nameof(CanUpdateAll))]
    public async Task UpdateAllAsync()
    {
        if (Feed is null)
        {
            return;
        }

        var targets = Entries.Where(entry => entry.CanUpdateFromFeed).ToList();
        if (targets.Count == 0)
        {
            return;
        }

        long total = Math.Max(1, targets.Sum(PackageBytes));
        long before = 0;
        int updated = 0;
        var failures = new List<string>();
        using var cancellation = new CancellationTokenSource();
        _updateAllCancellation = cancellation;
        IsUpdatingAll = true;
        UpdateAllProgress = 0;
        try
        {
            for (int i = 0; i < targets.Count && !cancellation.IsCancellationRequested; i++)
            {
                var entry = targets[i];
                long size = Math.Max(1, PackageBytes(entry));
                var install = entry.Entry.Manifest?.FeedPackage is { } id ? Feed.Update(id) : null;
                if (install is null)
                {
                    failures.Add($"{entry.Name} is no longer in the feed.");
                    before += size;
                    continue;
                }

                entry.IsUpdating = true;
                int position = i + 1;
                long start = before;
                void Report()
                {
                    UpdateAllProgress = Math.Min(1, (start + install.Progress * size) / total);
                    UpdateAllStatus = $"Updating {position} of {targets.Count}: {entry.Name} · {install.Status}";
                }

                install.PropertyChanged += (_, _) => Report();
                Report();
                await using (cancellation.Token.Register(() => _services.Dispatcher.Post(() => install.CancelCommand.Execute(null))))
                {
                    await install.Completion;
                }

                entry.IsUpdating = false;
                if (install.IsDone)
                {
                    updated++;
                }
                else if (install.Error is { } error)
                {
                    failures.Add($"{entry.Name}: {error}");
                }

                before += size;
            }
        }
        finally
        {
            foreach (var entry in targets)
            {
                entry.IsUpdating = false;
            }

            bool cancelled = cancellation.IsCancellationRequested;
            _updateAllCancellation = null;
            IsUpdatingAll = false;
            UpdateAllProgress = 0;
            UpdateAllStatus = "";
            Status = $"Updated {updated} of {ByteFormat.Count(targets.Count, "model", "models")}."
                + (cancelled ? " Cancelled; the models that were not updated keep their installed version." : "")
                + (failures.Count > 0 ? " " + string.Join(" ", failures) : "");
            NotifyUpdatesChanged();
        }
    }

    /// <summary>Stops updating after cancelling the download in progress (its partial file is kept to resume).</summary>
    [RelayCommand]
    private void CancelUpdateAll() => _updateAllCancellation?.Cancel();
}
