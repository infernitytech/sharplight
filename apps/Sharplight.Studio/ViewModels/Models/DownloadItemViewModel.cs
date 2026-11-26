using System.Diagnostics;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Hub;
using Sharplight.Hub.Catalog;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Compatibility;
using Sharplight.Studio.Services.Hub;
using Sharplight.Studio.Services.Platform;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A download: progress, speed, and pause / resume / cancel.</summary>
internal sealed partial class DownloadItemViewModel : ObservableObject
{
    private const double SpeedSmoothing = 0.3;

    private readonly IHubService _hub;
    private readonly IUiDispatcher _dispatcher;
    private readonly Action<DownloadItemViewModel, ModelManifest> _completed;
    private CancellationTokenSource? _cancellation;
    private bool _discard;
    private long _lastBytes;
    private long _lastTimestamp;

    public DownloadItemViewModel(ModelReference reference, long expectedBytes, IHubService hub, IUiDispatcher dispatcher, Action<DownloadItemViewModel, ModelManifest> completed)
    {
        Reference = reference;
        TotalBytes = expectedBytes;
        _hub = hub;
        _dispatcher = dispatcher;
        _completed = completed;
    }

    /// <summary>What is pulled.</summary>
    public ModelReference Reference { get; }

    /// <summary><c>owner/name:tag</c>.</summary>
    public string Name => DisplayName ?? Reference.ToString();

    /// <summary>The name shown instead of the store reference (a catalog install's model and variant), or null.</summary>
    public string? DisplayName { get; init; }

    /// <summary>The verified-catalog variant this download installs with its companions (<c>entry/variant</c>), or null for a plain pull.</summary>
    public string? CatalogKey { get; init; }

    /// <summary>The model kind, inferred from the Hub's pipeline tag (the model is not converted yet); null when not known.</summary>
    public string? Kind { get; set; }

    /// <summary>The kind's name, or empty.</summary>
    public string KindTitle => Kind is null ? "" : ModelKindCatalog.For(Kind).Title;

    /// <summary>The kind's icon resource key.</summary>
    public string KindIcon => ModelKindCatalog.For(Kind).Icon;

    /// <summary>The download's kind is known.</summary>
    public bool HasKind => Kind is not null;

    /// <summary>Current step.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(PhaseText), nameof(IsActive), nameof(CanPause), nameof(CanResume), nameof(IsIndeterminate), nameof(IsFinished),
        nameof(Progress), nameof(FileText), nameof(StepIndex))]
    [NotifyCanExecuteChangedFor(nameof(PauseCommand), nameof(ResumeCommand), nameof(CancelCommand))]
    public partial PullPhase Phase { get; set; } = PullPhase.Queued;

    /// <summary>Latest log line or error.</summary>
    [ObservableProperty]
    public partial string Message { get; set; } = "Queued";

    /// <summary>Downloaded bytes.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(Progress), nameof(BytesText), nameof(EtaText))]
    public partial long ReceivedBytes { get; set; }

    /// <summary>Bytes to download.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(Progress), nameof(BytesText), nameof(EtaText))]
    public partial long TotalBytes { get; set; }

    /// <summary>Smoothed download speed.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(SpeedText), nameof(EtaText))]
    public partial double BytesPerSecond { get; set; }

    /// <summary>The file being downloaded.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(FileText))]
    public partial string? CurrentFile { get; set; }

    /// <summary>Downloaded bytes of <see cref="CurrentFile"/>.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(FileText))]
    public partial long FileReceived { get; set; }

    /// <summary>Size of <see cref="CurrentFile"/>.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(FileText))]
    public partial long FileSize { get; set; }

    /// <summary>Work done in the converting / compiling step.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(Progress), nameof(IsIndeterminate))]
    public partial long StageCompleted { get; set; }

    /// <summary>Work of the converting / compiling step.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(Progress), nameof(IsIndeterminate))]
    public partial long StageTotal { get; set; }

    /// <summary>Progress of the current step, 0..1 (bytes while downloading, kernels while compiling).</summary>
    public double Progress => Phase is PullPhase.Converting or PullPhase.Precompiling
        ? StageTotal <= 0 ? 0 : Math.Clamp((double)StageCompleted / StageTotal, 0, 1)
        : TotalBytes <= 0 ? 0 : Math.Clamp((double)ReceivedBytes / TotalBytes, 0, 1);

    /// <summary><c>model-00001.safetensors · 1.2 GB / 4.0 GB</c> while downloading.</summary>
    public string FileText => Phase == PullPhase.Downloading && CurrentFile is not null
        ? $"{Path.GetFileName(CurrentFile)} · {ByteFormat.Format(FileReceived)} / {ByteFormat.Format(FileSize)}"
        : "";

    /// <summary>The steps of a pull with the current one marked, e.g. <c>Download › Convert › Compile</c>.</summary>
    public int StepIndex => Phase switch
    {
        PullPhase.Converting => 1,
        PullPhase.Precompiling => 2,
        PullPhase.Completed => 3,
        _ => 0,
    };

    /// <summary><c>1.2 GB / 4.5 GB</c>.</summary>
    public string BytesText => $"{ByteFormat.Format(ReceivedBytes)} / {ByteFormat.Format(TotalBytes)}";

    /// <summary>Speed while downloading.</summary>
    public string SpeedText => Phase == PullPhase.Downloading && BytesPerSecond > 0 ? ByteFormat.Rate(BytesPerSecond) : "";

    /// <summary>Remaining time while downloading.</summary>
    public string EtaText => Phase == PullPhase.Downloading && BytesPerSecond > 1 && TotalBytes > ReceivedBytes
        ? ByteFormat.Duration(TimeSpan.FromSeconds((TotalBytes - ReceivedBytes) / BytesPerSecond)) + " left"
        : "";

    /// <summary>Phase as text.</summary>
    public string PhaseText => Phase switch
    {
        PullPhase.Precompiling => "Compiling kernels",
        var phase => phase.ToString(),
    };

    /// <summary>Running (not paused, finished or failed).</summary>
    public bool IsActive => Phase is PullPhase.Queued or PullPhase.Resolving or PullPhase.Downloading or PullPhase.Converting or PullPhase.Precompiling;

    /// <summary>Done, failed or cancelled.</summary>
    public bool IsFinished => Phase is PullPhase.Completed or PullPhase.Failed or PullPhase.Cancelled;

    /// <summary>Progress cannot be measured (resolving, converting, compiling before its kernel count is known).</summary>
    public bool IsIndeterminate => Phase is PullPhase.Resolving || (Phase is PullPhase.Converting or PullPhase.Precompiling && StageTotal <= 0);

    /// <summary>Pausing is possible (a conversion or compilation in progress is stopped and redone on resume).</summary>
    public bool CanPause => Phase is PullPhase.Queued or PullPhase.Resolving or PullPhase.Downloading or PullPhase.Converting or PullPhase.Precompiling;

    /// <summary>Paused by the user before the app closed: restored paused instead of resuming.</summary>
    public bool IsPaused => Phase == PullPhase.Paused;

    /// <summary>Marks a restored download as paused without running it.</summary>
    public void RestorePaused()
    {
        Phase = PullPhase.Paused;
        Message = "Paused; resume continues where it stopped.";
    }

    /// <summary>Resuming is possible.</summary>
    public bool CanResume => Phase is PullPhase.Paused or PullPhase.Failed;

    /// <summary>Starts (or resumes) the pull; completes when it stopped for any reason.</summary>
    public async Task StartAsync()
    {
        if (IsActive && _cancellation is not null)
        {
            return;
        }

        _discard = false;
        using var cancellation = new CancellationTokenSource();
        _cancellation = cancellation;
        Phase = PullPhase.Resolving;
        Message = "Starting…";
        _lastTimestamp = Stopwatch.GetTimestamp();
        _lastBytes = ReceivedBytes;
        using var progress = new DispatchingProgress<PullUpdate>(_dispatcher, OnProgress, SameTransfer);
        try
        {
            var manifest = await Task.Run(() => CatalogKey is null ? _hub.PullAsync(Reference, progress, cancellation.Token) : InstallAsync(progress, cancellation.Token),
                CancellationToken.None);
            progress.Flush();
            Phase = PullPhase.Completed;
            Message = $"Stored as {manifest.ShortName}";
            ReceivedBytes = Math.Max(ReceivedBytes, TotalBytes);
            _completed(this, manifest);
        }
        catch (OperationCanceledException)
        {
            if (_discard)
            {
                _hub.DiscardPartial(Reference);
                Phase = PullPhase.Cancelled;
                Message = "Cancelled; downloaded parts deleted.";
            }
            else
            {
                Phase = PullPhase.Paused;
                Message = "Paused; resume continues where it stopped.";
            }
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            Phase = PullPhase.Failed;
            Message = ImportFailure.Describe(error, error.Message);
        }
        finally
        {
            _cancellation = null;
            BytesPerSecond = 0;
        }

    }

    /// <summary>Installs <see cref="CatalogKey"/> from the current catalog (a restored download resolves it after a restart).</summary>
    private async Task<ModelManifest> InstallAsync(IProgress<PullUpdate> progress, CancellationToken cancellationToken)
    {
        var catalog = await _hub.GetCatalogAsync(cancellationToken).ConfigureAwait(false);
        var parts = CatalogKey!.Split('/', 2);
        var entry = catalog.Find(parts[0]);
        var variant = parts.Length == 2 ? entry?.Variants.FirstOrDefault(candidate => candidate.Id == parts[1]) : null;
        if (entry is null || variant is null)
        {
            throw new HubException($"{CatalogKey} is no longer in the verified catalog.");
        }

        return await _hub.InstallAsync(CatalogInstallPlan.For(entry, variant), progress, cancellationToken).ConfigureAwait(false);
    }

    [RelayCommand(CanExecute = nameof(CanPause))]
    private void Pause() => _cancellation?.Cancel();

    [RelayCommand(CanExecute = nameof(CanResume))]
    private Task ResumeAsync() => StartAsync();

    private bool CanCancel() => !IsFinished;

    [RelayCommand(CanExecute = nameof(CanCancel))]
    private void Cancel()
    {
        if (_cancellation is { } running)
        {
            _discard = true;
            running.Cancel();
            return;
        }

        _hub.DiscardPartial(Reference);
        Phase = PullPhase.Cancelled;
        Message = "Cancelled; downloaded parts deleted.";
    }

    /// <summary>A byte count of the same phase, file and message supersedes the one before it (only the latest is shown).</summary>
    private static bool SameTransfer(PullUpdate previous, PullUpdate next)
        => previous.Phase == next.Phase && previous.File == next.File && previous.Message == next.Message;

    private void OnProgress(PullUpdate update)
    {
        if (!IsActive)
        {
            return;
        }

        Phase = update.Phase == PullPhase.Completed ? Phase : update.Phase;
        StageCompleted = update.StageCompleted;
        StageTotal = update.StageTotal;
        if (update.File is not null)
        {
            CurrentFile = update.File;
            FileReceived = update.FileReceived;
            FileSize = update.FileSize;
        }

        if (update.Message.Length > 0)
        {
            Message = update.Message;
        }

        if (update.Total > 0)
        {
            TotalBytes = update.Total;
            ReceivedBytes = update.Received;
            long now = Stopwatch.GetTimestamp();
            double seconds = Stopwatch.GetElapsedTime(_lastTimestamp, now).TotalSeconds;
            if (seconds >= 0.5)
            {
                double rate = Math.Max(0, update.Received - _lastBytes) / seconds;
                BytesPerSecond = BytesPerSecond <= 0 ? rate : (SpeedSmoothing * rate) + ((1 - SpeedSmoothing) * BytesPerSecond);
                _lastBytes = update.Received;
                _lastTimestamp = now;
            }
        }
    }
}
