using CommunityToolkit.Mvvm.ComponentModel;

namespace Sharplight.Studio.ViewModels.Inference;

/// <summary>One run in the <see cref="RunQueueViewModel"/>.</summary>
internal sealed partial class RunItemViewModel : ObservableObject
{
    private readonly CancellationTokenSource _cancellation = new();

    internal RunItemViewModel(string title, string modelId)
    {
        Title = title;
        ModelId = modelId;
    }

    /// <summary>What the run is for.</summary>
    public string Title { get; }

    /// <summary>The model that runs it.</summary>
    public string ModelId { get; }

    /// <summary>Completes when the run ended (also when it failed or was cancelled); never faults.</summary>
    public Task Completion { get; internal set; } = Task.CompletedTask;

    internal CancellationToken Token => _cancellation.Token;

    /// <summary>Where the run stands.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsActive), nameof(IsRunning), nameof(IsFinished), nameof(StatusText))]
    public partial RunStatus Status { get; set; }

    /// <summary>Progress from 0 to 1 while <see cref="IsIndeterminate"/> is false.</summary>
    [ObservableProperty]
    public partial double Progress { get; set; }

    /// <summary>The progress cannot be told.</summary>
    [ObservableProperty]
    public partial bool IsIndeterminate { get; set; } = true;

    /// <summary>What the run is doing, or the failure message.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(StatusText))]
    public partial string Detail { get; set; } = "";

    /// <summary>Queued or running.</summary>
    public bool IsActive => Status is RunStatus.Queued or RunStatus.Running;

    /// <summary>Running.</summary>
    public bool IsRunning => Status == RunStatus.Running;

    /// <summary>Completed, failed or cancelled.</summary>
    public bool IsFinished => !IsActive;

    /// <summary>The state in a few words, with the detail.</summary>
    public string StatusText
    {
        get
        {
            string state = Status switch
            {
                RunStatus.Queued => "Waiting for the model",
                RunStatus.Running => "Running",
                RunStatus.Completed => "Done",
                RunStatus.Failed => "Failed",
                _ => "Cancelled",
            };
            return Detail.Length == 0 ? state : $"{state}: {Detail}";
        }
    }

    /// <summary>Cancels the run (a queued run never starts).</summary>
    [CommunityToolkit.Mvvm.Input.RelayCommand]
    public void Cancel() => _cancellation.Cancel();
}
