using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.ModelFeed;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Feed;
using Sharplight.Studio.Services.Platform;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>One install (or update) from the model feed: progress, cancel (the partial download is kept) and retry (resumes it).</summary>
internal sealed partial class FeedInstallViewModel : ObservableObject
{
    private readonly IFeedService _feed;
    private readonly IUiDispatcher _dispatcher;
    private readonly Action<FeedInstallViewModel, FeedInstallResult> _completed;
    private readonly Func<CancellationToken, Task<bool>> _acceptLicense;
    private CancellationTokenSource? _cancellation;

    public FeedInstallViewModel(string id, string name, string version, bool isUpdate, IFeedService feed, IUiDispatcher dispatcher,
        Action<FeedInstallViewModel, FeedInstallResult> completed, Func<CancellationToken, Task<bool>>? acceptLicense = null)
    {
        _acceptLicense = acceptLicense ?? (_ => Task.FromResult(true));
        Id = id;
        Name = name;
        Version = version;
        IsUpdate = isUpdate;
        _feed = feed;
        _dispatcher = dispatcher;
        _completed = completed;
    }

    /// <summary>The package id.</summary>
    public string Id { get; }

    /// <summary>The name shown.</summary>
    public string Name { get; }

    /// <summary>The version being installed (the latest the feed lists; dependencies may bring others).</summary>
    public string Version { get; }

    /// <summary>An installed version is being replaced.</summary>
    public bool IsUpdate { get; }

    /// <summary>The title line: <c>Qwen3 0.6B 1.0.0</c> or <c>Updating …</c>.</summary>
    public string Title => $"{(IsUpdate ? "Update " : "")}{Name} {Version}";

    /// <summary>Completes when the install ended (successfully, failed or cancelled).</summary>
    public Task Completion { get; private set; } = Task.CompletedTask;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanRetry))]
    [NotifyCanExecuteChangedFor(nameof(CancelCommand), nameof(RetryCommand))]
    public partial bool IsActive { get; set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowCard))]
    public partial bool IsDone { get; set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanRetry), nameof(HasError))]
    [NotifyCanExecuteChangedFor(nameof(RetryCommand))]
    public partial string? Error { get; set; }

    /// <summary>The install failed because a signature, digest or hash did not match: nothing was installed.</summary>
    [ObservableProperty]
    public partial bool IsVerificationFailure { get; set; }

    /// <summary>The install was cancelled; the partial download is kept.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanRetry))]
    public partial bool IsCancelled { get; set; }

    [ObservableProperty]
    public partial double Progress { get; set; }

    [ObservableProperty]
    public partial string Status { get; set; } = "";

    public bool HasError => Error is not null;

    /// <summary>The install is still of interest as a card: running, failed or cancelled. A finished one only counts in the page's summary.</summary>
    public bool ShowCard => !IsDone;

    /// <summary>The install can be started again (it resumes where it stopped).</summary>
    public bool CanRetry => !IsActive && !IsDone && (IsCancelled || Error is not null && !IsVerificationFailure);

    /// <summary>Starts or resumes the install.</summary>
    public Task StartAsync()
    {
        if (IsActive)
        {
            return Completion;
        }

        Error = null;
        IsCancelled = false;
        IsVerificationFailure = false;
        IsActive = true;
        _cancellation = new CancellationTokenSource();
        Completion = RunAsync(_cancellation.Token);
        return Completion;
    }

    [RelayCommand(CanExecute = nameof(IsActive))]
    private void Cancel() => _cancellation?.Cancel();

    [RelayCommand(CanExecute = nameof(CanRetry))]
    private Task Retry() => StartAsync();

    private async Task RunAsync(CancellationToken cancellationToken)
    {
        using var progress = new DispatchingProgress<FeedInstallProgress>(_dispatcher, Apply, (previous, next) => previous.Phase == next.Phase && previous.PackageId == next.PackageId);
        Status = "Reading the feed…";
        try
        {
            Status = "Waiting for the licence terms to be accepted…";
            if (!await _acceptLicense(cancellationToken))
            {
                IsCancelled = true;
                Status = "Not installed: the licence terms were not accepted.";
                return;
            }

            Status = "Reading the feed…";
            var result = await Task.Run(() => _feed.InstallAsync(Id, progress, cancellationToken), CancellationToken.None);
            progress.Flush();
            Progress = 1;
            Status = result.Installed.Count == 0 ? "Already installed" : "Installed";
            IsDone = true;
            _completed(this, result);
        }
        catch (OperationCanceledException)
        {
            progress.Flush();
            IsCancelled = true;
            Status = "Cancelled. The partial download is kept and resumes when you install again.";
        }
        catch (FeedVerificationException error)
        {
            progress.Flush();
            IsVerificationFailure = true;
            Error = $"Not installed: verification failed. {error.Message}";
            Status = "Verification failed";
        }
        catch (Exception error) when (error is FeedFormatException or FeedInstallException or IOException or UnauthorizedAccessException or KeyNotFoundException)
        {
            progress.Flush();
            Error = error.Message;
            Status = "Failed";
        }
        finally
        {
            IsActive = false;
            _cancellation?.Dispose();
            _cancellation = null;
        }
    }

    private void Apply(FeedInstallProgress update)
    {
        Status = update.Phase switch
        {
            FeedInstallPhase.Resolving => "Reading the feed…",
            FeedInstallPhase.Downloading => $"Downloading {update.PackageId} · {ByteFormat.Format(update.Received)} of {ByteFormat.Format(update.Total)}",
            FeedInstallPhase.Installing => $"Installing {update.PackageId}…",
            _ => "Installed",
        };
        Progress = update.Total > 0 ? Math.Min(1, (double)update.Received / update.Total) : 0;
    }
}
