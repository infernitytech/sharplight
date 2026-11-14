using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Hub;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>The download queue.</summary>
internal sealed partial class DownloadsViewModel : ObservableObject
{
    private readonly StudioServices _services;

    public DownloadsViewModel(StudioServices services)
    {
        _services = services;
    }

    /// <summary>Overall progress of the running downloads, 0..1.</summary>
    [ObservableProperty]
    public partial double OverallProgress { get; set; }

    /// <summary>Summary for the global indicator, e.g. <c>2 downloads · 41 %</c>.</summary>
    [ObservableProperty]
    public partial string Summary { get; set; } = "";

    /// <summary>Adds the downloads left over from the last session: running ones resume, paused ones stay paused.</summary>
    public void Restore()
    {
        foreach (var pending in _services.Settings.Current.Downloads)
        {
            if (!ModelReference.TryParse(pending.Reference, out var reference) || Items.Any(item => item.Reference == reference))
            {
                continue;
            }

            var item = Create(reference, pending.ExpectedBytes);
            Items.Add(item);
            if (pending.Paused)
            {
                item.RestorePaused();
            }
            else
            {
                _ = Run(item);
            }
        }

        OnPropertyChanged(nameof(IsEmpty));
        Update();
    }

    /// <summary>Raised on the UI thread when a model was stored.</summary>
    public event EventHandler<ModelManifest>? ModelStored;

    /// <summary>Raised when a download was added.</summary>
    public event EventHandler? Enqueued;

    /// <summary>Downloads, newest first.</summary>
    public ObservableCollection<DownloadItemViewModel> Items { get; } = [];

    /// <summary>Downloads still running.</summary>
    [ObservableProperty]
    public partial int ActiveCount { get; set; }

    /// <summary>True when the queue is empty.</summary>
    public bool IsEmpty => Items.Count == 0;

    /// <summary>Adds and starts a download (an existing entry for the same model is resumed instead).</summary>
    public DownloadItemViewModel Enqueue(ModelReference reference, long expectedBytes)
    {
        var existing = Items.FirstOrDefault(item => item.Reference == reference && !item.IsFinished);
        if (existing is not null)
        {
            if (existing.CanResume)
            {
                _ = Run(existing);
            }

            return existing;
        }

        var download = Create(reference, expectedBytes);
        Items.Insert(0, download);
        OnPropertyChanged(nameof(IsEmpty));
        Enqueued?.Invoke(this, EventArgs.Empty);
        _ = Run(download);
        return download;
    }

    [RelayCommand]
    private void Remove(DownloadItemViewModel item)
    {
        if (item.IsActive)
        {
            return;
        }

        Items.Remove(item);
        OnPropertyChanged(nameof(IsEmpty));
        Persist();
    }

    [RelayCommand]
    private void ClearFinished()
    {
        foreach (var item in Items.Where(item => item.IsFinished).ToList())
        {
            Items.Remove(item);
        }

        OnPropertyChanged(nameof(IsEmpty));
    }

    private static Task Run(DownloadItemViewModel item) => item.StartAsync();

    private DownloadItemViewModel Create(ModelReference reference, long expectedBytes)
    {
        var download = new DownloadItemViewModel(reference, expectedBytes, _services.Hub, _services.Dispatcher, OnCompleted);
        download.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(DownloadItemViewModel.Phase))
            {
                Persist();
            }

            if (args.PropertyName is nameof(DownloadItemViewModel.Phase) or nameof(DownloadItemViewModel.ReceivedBytes) or nameof(DownloadItemViewModel.StageCompleted))
            {
                Update();
            }
        };
        return download;
    }

    private void Update()
    {
        var active = Items.Where(item => item.IsActive).ToList();
        ActiveCount = active.Count;
        long total = active.Sum(item => item.TotalBytes);
        OverallProgress = total <= 0 ? 0 : (double)active.Sum(item => Math.Min(item.ReceivedBytes, item.TotalBytes)) / total;
        Summary = active.Count switch
        {
            0 => "",
            1 => $"{active[0].PhaseText} {active[0].Progress:P0}",
            _ => $"{active.Count} downloads · {OverallProgress:P0}",
        };
    }

    /// <summary>Saves the unfinished downloads so they survive a restart.</summary>
    private void Persist()
    {
        var pending = Items.Where(item => !item.IsFinished)
            .Select(item => new PendingDownload(item.Reference.ToString(), item.TotalBytes, item.IsPaused))
            .ToList();
        if (pending.SequenceEqual(_services.Settings.Current.Downloads))
        {
            return;
        }

        _services.Settings.Update(current => current with { Downloads = pending });
    }

    private void OnCompleted(DownloadItemViewModel item, ModelManifest manifest)
    {
        Update();
        Persist();
        ModelStored?.Invoke(this, manifest);
    }
}
