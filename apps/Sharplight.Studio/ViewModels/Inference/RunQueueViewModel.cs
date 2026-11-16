using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.Services.Threading;

namespace Sharplight.Studio.ViewModels.Inference;

/// <summary>
/// The run queue every inference page shares: runs are started one after the other per model (a second run on the same model waits
/// for the first), can be cancelled while queued or running, and stay listed with their outcome until cleared. Work runs on a pool
/// thread; progress and state changes reach the view models on the UI thread, progress throttled to about 15 updates a second.
/// </summary>
internal sealed partial class RunQueueViewModel : ObservableObject
{
    private readonly IUiDispatcher _dispatcher;
    private readonly Dictionary<string, SemaphoreSlim> _gates = new(StringComparer.Ordinal);
    private readonly Lock _lock = new();

    /// <param name="dispatcher">Runs UI updates on the UI thread.</param>
    public RunQueueViewModel(IUiDispatcher dispatcher)
    {
        _dispatcher = dispatcher;
    }

    /// <summary>The runs, newest last.</summary>
    public ObservableCollection<RunItemViewModel> Items { get; } = [];

    /// <summary>Runs that are queued or running.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsBusy), nameof(HasItems))]
    public partial int ActiveCount { get; set; }

    /// <summary>A run is queued or running.</summary>
    public bool IsBusy => ActiveCount > 0;

    /// <summary>The queue lists something.</summary>
    public bool HasItems => Items.Count > 0;

    /// <summary>Adds a run; it starts when the model is free.</summary>
    /// <param name="title">What the run is for.</param>
    /// <param name="modelId">The model that runs it; runs of the same model do not overlap.</param>
    /// <param name="work">The run. Exceptions other than cancellation end it as failed.</param>
    public RunItemViewModel Enqueue(string title, string modelId, Func<RunContext, Task> work)
    {
        var item = new RunItemViewModel(title, modelId) { Status = RunStatus.Queued };
        lock (_lock)
        {
            Items.Add(item);
            ActiveCount = CountActive();
        }

        OnPropertyChanged(nameof(HasItems));
        item.Completion = Task.Run(() => RunAsync(item, work));
        return item;
    }

    /// <summary>Cancels every queued and running run.</summary>
    [RelayCommand]
    public void CancelAll()
    {
        foreach (var item in Items.Where(item => item.IsActive))
        {
            item.Cancel();
        }
    }

    /// <summary>Removes the finished runs from the list.</summary>
    [RelayCommand]
    public void ClearFinished()
    {
        lock (_lock)
        {
            foreach (var item in Items.Where(item => item.IsFinished).ToList())
            {
                Items.Remove(item);
            }
        }

        OnPropertyChanged(nameof(HasItems));
    }

    private async Task RunAsync(RunItemViewModel item, Func<RunContext, Task> work)
    {
        SemaphoreSlim gate;
        lock (_lock)
        {
            if (!_gates.TryGetValue(item.ModelId, out gate!))
            {
                _gates[item.ModelId] = gate = new SemaphoreSlim(1, 1);
            }
        }

        RunStatus outcome;
        string detail = "";
        var progress = new RunProgress();
        using var throttle = new UpdateThrottle(() => Update(() => progress.ApplyTo(item)));
        try
        {
            await gate.WaitAsync(item.Token).ConfigureAwait(false);
            try
            {
                item.Token.ThrowIfCancellationRequested();
                Update(() => item.Status = RunStatus.Running);
                var context = new RunContext(item.Token, (fraction, text) =>
                {
                    // Runs report per step or token: the latest state reaches the UI at most about 15 times a second.
                    progress.Set(fraction, text);
                    throttle.Request();
                });
                await work(context).ConfigureAwait(false);
                outcome = RunStatus.Completed;
            }
            finally
            {
                throttle.Flush();
                gate.Release();
            }
        }
        catch (OperationCanceledException)
        {
            outcome = RunStatus.Cancelled;
        }
        catch (Exception error)
        {
            outcome = RunStatus.Failed;
            detail = error.GetBaseException().Message;
        }

        Update(() =>
        {
            // Under the lock: on the UI thread nothing contends, but a dispatcher that runs updates inline (tests) can finish two runs on
            // two threads at once, and the last count written must see both outcomes.
            lock (_lock)
            {
                item.Status = outcome;
                item.Detail = detail;
                item.IsIndeterminate = false;
                item.Progress = outcome == RunStatus.Completed ? 1 : item.Progress;
                ActiveCount = CountActive();
            }
        });
    }

    /// <summary>The runs still queued or running.</summary>
    private int CountActive()
    {
        lock (_lock)
        {
            return Items.Count(other => other.IsActive);
        }
    }

    private void Update(Action change) => _dispatcher.Post(change);
}
