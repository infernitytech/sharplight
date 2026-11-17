using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.Services.Threading;

namespace Sharplight.Studio.ViewModels.Inference;

/// <summary>
/// The run queue every inference page shares: runs are started one after the other per model, in the order they were queued (a second
/// run on the same model waits for the first), can be cancelled while queued or running, and stay listed with their outcome until cleared. Work runs on a pool
/// thread; progress and state changes reach the view models on the UI thread, progress throttled to about 15 updates a second.
/// </summary>
internal sealed partial class RunQueueViewModel : ObservableObject
{
    private readonly IUiDispatcher _dispatcher;
    // Per model, the end of the last queued run: a new run waits for it, so runs of one model start in the order they were queued.
    private readonly Dictionary<string, Task> _tails = new(StringComparer.Ordinal);
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
        var done = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        Task previous;
        lock (_lock)
        {
            Items.Add(item);
            ActiveCount = CountActive();
            previous = _tails.TryGetValue(modelId, out var tail) ? tail : Task.CompletedTask;
            _tails[modelId] = done.Task;
        }

        OnPropertyChanged(nameof(HasItems));
        item.Completion = Task.Run(() => RunAsync(item, work, previous, done));
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

    /// <param name="item">The run.</param>
    /// <param name="work">What it does.</param>
    /// <param name="previous">Completes when the run queued before it on the same model has ended.</param>
    /// <param name="done">Completed when this run has ended and the one before it too (the next run of the model waits for it).</param>
    private async Task RunAsync(RunItemViewModel item, Func<RunContext, Task> work, Task previous, TaskCompletionSource done)
    {
        RunStatus outcome;
        string detail = "";
        var progress = new RunProgress();
        using var throttle = new UpdateThrottle(() => Update(() => progress.ApplyTo(item)));
        try
        {
            try
            {
                await previous.WaitAsync(item.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                // A run cancelled while queued never starts; the runs queued after it still wait for the ones before it.
                _ = CompleteAfterAsync(previous, done);
                throw;
            }

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
                done.TrySetResult();
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

    /// <summary>Completes <paramref name="done"/> once <paramref name="previous"/> has completed.</summary>
    private static async Task CompleteAfterAsync(Task previous, TaskCompletionSource done)
    {
        await previous.ConfigureAwait(false);
        done.TrySetResult();
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
