using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Maintenance;

namespace Sharplight.Studio.ViewModels.Models;

// Disk usage and cleanup: sorting by size or last used, the size breakdown of a model, bulk selection, deleting models, removing only their
// precompiled artifacts, and the suggestion to clean up models that were not used for 90 days.
internal sealed partial class LibraryViewModel
{
    /// <summary>A model not loaded for this long is suggested for cleanup.</summary>
    public static readonly TimeSpan UnusedAfter = TimeSpan.FromDays(90);

    /// <summary>The clock (tests set it).</summary>
    public TimeProvider Time { get; }

    /// <summary>The file operation running now (deleting, exporting, importing).</summary>
    public FileOperationViewModel Operation { get; }

    /// <summary>The ways the library can be ordered.</summary>
    public IReadOnlyList<FilterOption<LibrarySort>> SortChoices { get; }

    /// <summary>The chosen order.</summary>
    [ObservableProperty]
    public partial FilterOption<LibrarySort> SelectedSort { get; set; }

    partial void OnSelectedSortChanged(FilterOption<LibrarySort> value) => ApplyKindFilter();

    /// <summary>The question waiting for an answer (delete, remove artifacts, import), or null.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasConfirmation))]
    public partial ConfirmationViewModel? Confirmation { get; set; }

    /// <summary>A question is shown.</summary>
    public bool HasConfirmation => Confirmation is not null;

    /// <summary>Models can be ticked for a bulk action.</summary>
    [ObservableProperty]
    public partial bool IsSelecting { get; set; }

    partial void OnIsSelectingChanged(bool value)
    {
        if (!value)
        {
            SelectNone();
        }
    }

    /// <summary>How many models are ticked.</summary>
    public int SelectedCount => Entries.Count(entry => entry.IsSelected);

    /// <summary>The ticked models' size on disk.</summary>
    public string SelectedSize => ByteFormat.Format(Entries.Where(entry => entry.IsSelected).Sum(entry => entry.SizeBytes));

    /// <summary>At least one model is ticked.</summary>
    public bool HasSelection => SelectedCount > 0;

    /// <summary>The models of the library that were not loaded for 90 days (outside the store, a model is only a list entry and is not suggested).</summary>
    public IReadOnlyList<LibraryEntryViewModel> UnusedEntries { get; private set; } = [];

    /// <summary>Some models were not used for 90 days.</summary>
    public bool HasUnusedSuggestion => UnusedEntries.Count > 0;

    /// <summary><c>3 models were not loaded for 90 days (24.5 GB).</c></summary>
    public string UnusedSuggestion => UnusedEntries.Count == 0
        ? ""
        : $"{ByteFormat.Count(UnusedEntries.Count, "model was", "models were")} not loaded for {UnusedAfter.Days} days ({ByteFormat.Format(UnusedEntries.Sum(entry => entry.SizeBytes))}).";

    /// <summary>Recomputes <see cref="UnusedEntries"/>.</summary>
    public void RefreshUnused()
    {
        var now = Time.GetUtcNow();
        UnusedEntries = [.. Entries.Where(entry => entry.Entry.Manifest is not null && !entry.IsLoaded && entry.LastUsedOrAdded is { } used && now - used > UnusedAfter)];
        OnPropertyChanged(nameof(UnusedEntries));
        OnPropertyChanged(nameof(HasUnusedSuggestion));
        OnPropertyChanged(nameof(UnusedSuggestion));
    }

    /// <summary>Starts the selection mode with the unused models ticked.</summary>
    [RelayCommand]
    public void ReviewUnused()
    {
        IsSelecting = true;
        foreach (var entry in Entries)
        {
            entry.IsSelected = UnusedEntries.Contains(entry);
        }
    }

    [RelayCommand]
    private void ToggleSelecting() => IsSelecting = !IsSelecting;

    [RelayCommand]
    private void SelectAll()
    {
        foreach (var entry in VisibleEntries)
        {
            entry.IsSelected = true;
        }
    }

    [RelayCommand]
    private void SelectNone()
    {
        foreach (var entry in Entries)
        {
            entry.IsSelected = false;
        }
    }

    [RelayCommand]
    private Task DeleteSelectedAsync() => RequestDeleteAsync([.. Entries.Where(entry => entry.IsSelected)]);

    [RelayCommand]
    private Task StripSelectedAsync() => RequestStripAsync([.. Entries.Where(entry => entry.IsSelected)]);

    private void OnEntryChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs args)
    {
        if (args.PropertyName == nameof(LibraryEntryViewModel.IsSelected))
        {
            OnPropertyChanged(nameof(SelectedCount));
            OnPropertyChanged(nameof(SelectedSize));
            OnPropertyChanged(nameof(HasSelection));
        }
        else if (args.PropertyName == nameof(LibraryEntryViewModel.FeedUpdateVersion))
        {
            NotifyUpdatesChanged();
        }
    }

    private IEnumerable<LibraryEntryViewModel> Sort(IEnumerable<LibraryEntryViewModel> entries) => SelectedSort.Value switch
    {
        LibrarySort.Size => entries.OrderByDescending(entry => entry.SizeBytes).ThenBy(entry => entry.Name, StringComparer.OrdinalIgnoreCase),
        LibrarySort.LastUsed => entries.OrderByDescending(entry => entry.LastUsedOrAdded ?? DateTimeOffset.MinValue).ThenBy(entry => entry.Name, StringComparer.OrdinalIgnoreCase),
        LibrarySort.LeastRecentlyUsed => entries.OrderBy(entry => entry.LastUsedOrAdded ?? DateTimeOffset.MinValue).ThenBy(entry => entry.Name, StringComparer.OrdinalIgnoreCase),
        _ => entries,
    };

    private void Ask(string message, string confirmText, bool destructive, Func<Task> action)
    {
        Confirmation = new ConfirmationViewModel(message, confirmText, destructive, action, answered =>
        {
            if (ReferenceEquals(Confirmation, answered))
            {
                Confirmation = null;
            }
        });
    }

    /// <summary>The reason a model cannot be deleted or rewritten now (resident, being precompiled), or null.</summary>
    private string? WhyBlocked(LibraryEntryViewModel entry, string action)
    {
        if (entry.IsLoaded)
        {
            return $"Unload {entry.Id} before {action} it.";
        }

        if (entry.IsPrecompiling)
        {
            return $"{entry.Id} is being precompiled; cancel that first.";
        }

        return _services.Maintenance.WhyInUse(entry.Entry);
    }

    [RelayCommand]
    private Task DeleteAsync(LibraryEntryViewModel entry) => RequestDeleteAsync([entry]);

    /// <summary>
    /// Asks before deleting <paramref name="entries"/>, saying how much space it frees; resident models are never part of it (the message says
    /// they were skipped, or refuses when nothing else is left).
    /// </summary>
    public async Task RequestDeleteAsync(IReadOnlyList<LibraryEntryViewModel> entries)
    {
        if (entries.Count == 0 || Operation.IsRunning)
        {
            return;
        }

        var usable = entries.Where(entry => WhyBlocked(entry, "deleting") is null).ToList();
        var blocked = entries.Except(usable).ToList();
        if (usable.Count == 0)
        {
            Status = WhyBlocked(blocked[0], "deleting") ?? "";
            return;
        }

        long freed = await Task.Run(() => usable.Sum(entry => _services.Maintenance.FreedByDelete(entry.Entry)));
        var stored = usable.Where(entry => !entry.IsLocalFile).ToList();
        var files = usable.Count - stored.Count;
        string message;
        if (stored.Count == 0)
        {
            message = usable.Count == 1
                ? $"Remove {usable[0].Name} from the list? The file stays where it is: no space is freed."
                : $"Remove {usable.Count} files from the list? The files stay where they are: no space is freed.";
        }
        else if (usable.Count == 1)
        {
            message = $"Delete {usable[0].Name}? This frees {ByteFormat.Format(freed)}.";
        }
        else
        {
            message = $"Delete {ByteFormat.Count(stored.Count, "model", "models")}? This frees {ByteFormat.Format(freed)}."
                + (files > 0 ? $" {ByteFormat.Count(files, "file outside the store is", "files outside the store are")} only removed from the list." : "");
        }

        if (blocked.Count > 0)
        {
            message += $" {ByteFormat.Count(blocked.Count, "model is", "models are")} skipped because {(blocked.Count == 1 ? "it is" : "they are")} in use: {string.Join(", ", blocked.Select(entry => entry.Id))}. Unload first.";
        }

        Ask(message, stored.Count == 0 ? "Remove" : "Delete", destructive: true, () => DeleteConfirmedAsync(usable));
    }

    private async Task DeleteConfirmedAsync(IReadOnlyList<LibraryEntryViewModel> entries)
    {
        var deleted = new List<LibraryEntryViewModel>();
        var skipped = new List<string>();
        var failed = new List<string>();
        long freed = 0;
        bool completed;
        try
        {
            completed = await Operation.RunAsync("Deleting", async (progress, cancellationToken) =>
            {
                for (int i = 0; i < entries.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    progress.Report(new MaintenanceProgress($"Deleting {entries[i].Name} ({i + 1} of {entries.Count})", i, entries.Count));
                    try
                    {
                        freed += await _services.Maintenance.DeleteAsync(entries[i].Entry, cancellationToken).ConfigureAwait(false);
                        deleted.Add(entries[i]);
                    }
                    catch (ModelInUseException error)
                    {
                        skipped.Add(error.Message);
                    }
                    catch (Exception error) when (error is IOException or UnauthorizedAccessException or Sharplight.Hub.HubException)
                    {
                        failed.Add($"{entries[i].Name}: {error.Message}");
                    }
                }
            });
        }
        finally
        {
            foreach (var entry in deleted)
            {
                Entries.Remove(entry);
            }

            RebuildKindFilters();
            RefreshUnused();
            OnPropertyChanged(nameof(IsEmpty));
            NotifyUpdatesChanged();
        }

        var text = deleted.Count == 0 ? "Nothing was deleted." : $"Deleted {ByteFormat.Count(deleted.Count, "model", "models")}, freed {ByteFormat.Format(freed)}.";
        if (!completed)
        {
            text += " Cancelled.";
        }

        if (skipped.Count > 0)
        {
            text += " " + string.Join(" ", skipped);
        }

        if (failed.Count > 0)
        {
            text += " Failed: " + string.Join("; ", failed);
        }

        Status = text;
    }

    [RelayCommand]
    private Task StripPrecompiledAsync(LibraryEntryViewModel entry) => RequestStripAsync([entry]);

    /// <summary>Asks before removing only the precompiled artifacts of <paramref name="entries"/> (the models stay; the code is rebuilt by precompiling).</summary>
    public async Task RequestStripAsync(IReadOnlyList<LibraryEntryViewModel> entries)
    {
        if (entries.Count == 0 || Operation.IsRunning)
        {
            return;
        }

        var candidates = entries.Where(entry => entry.IsPackage && entry.IsPrecompiled).ToList();
        var usable = candidates.Where(entry => WhyBlocked(entry, "changing") is null).ToList();
        var blocked = candidates.Except(usable).ToList();
        if (usable.Count == 0)
        {
            Status = blocked.Count > 0 ? WhyBlocked(blocked[0], "changing") ?? "" : entries.Count == 1 ? $"{entries[0].Name} has no precompiled artifacts." : "None of the models has precompiled artifacts.";
            return;
        }

        long freed = 0;
        foreach (var entry in usable)
        {
            freed += await _services.Maintenance.FreedByStripAsync(entry.Entry, CancellationToken.None);
        }

        var message = usable.Count == 1
            ? $"Remove the precompiled artifacts of {usable[0].Name}? This frees {ByteFormat.Format(freed)}. The model stays; the code is built again when you precompile it or load it."
            : $"Remove the precompiled artifacts of {usable.Count} models? This frees {ByteFormat.Format(freed)}. The models stay; the code is built again when you precompile or load them.";
        if (blocked.Count > 0)
        {
            message += $" {ByteFormat.Count(blocked.Count, "model is", "models are")} skipped because {(blocked.Count == 1 ? "it is" : "they are")} in use: {string.Join(", ", blocked.Select(entry => entry.Id))}. Unload first.";
        }

        Ask(message, "Remove precompiled", destructive: true, () => StripConfirmedAsync(usable));
    }

    private async Task StripConfirmedAsync(IReadOnlyList<LibraryEntryViewModel> entries)
    {
        long freed = 0;
        int done = 0;
        var problems = new List<string>();
        bool completed = await Operation.RunAsync("Removing precompiled artifacts", async (progress, cancellationToken) =>
        {
            for (int i = 0; i < entries.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                progress.Report(new MaintenanceProgress($"Compacting {entries[i].Name} ({i + 1} of {entries.Count})", i, entries.Count));
                try
                {
                    freed += await _services.Maintenance.StripPrecompiledAsync(entries[i].Entry, cancellationToken).ConfigureAwait(false);
                    done++;
                }
                catch (Exception error) when (error is ModelInUseException or IOException or UnauthorizedAccessException or InvalidDataException)
                {
                    problems.Add(error.Message);
                }
            }
        });
        foreach (var entry in entries)
        {
            entry.Breakdown = null;
            await ReloadAsync(entry);
            if (entry.IsBreakdownOpen)
            {
                await LoadBreakdownAsync(entry);
            }
        }

        Status = (done == 0 ? "No artifacts were removed." : $"Removed the precompiled artifacts of {ByteFormat.Count(done, "model", "models")}, freed {ByteFormat.Format(freed)}.")
            + (completed ? "" : " Cancelled.") + (problems.Count > 0 ? " " + string.Join(" ", problems) : "");
    }

    [RelayCommand]
    private async Task ToggleBreakdownAsync(LibraryEntryViewModel entry)
    {
        entry.IsBreakdownOpen = !entry.IsBreakdownOpen;
        if (entry.IsBreakdownOpen && entry.Breakdown is null)
        {
            await LoadBreakdownAsync(entry);
        }
    }

    private async Task LoadBreakdownAsync(LibraryEntryViewModel entry)
    {
        entry.IsLoadingBreakdown = true;
        try
        {
            entry.ShowBreakdown(await _services.Maintenance.GetBreakdownAsync(entry.Entry, CancellationToken.None));
        }
        finally
        {
            entry.IsLoadingBreakdown = false;
        }
    }
}
