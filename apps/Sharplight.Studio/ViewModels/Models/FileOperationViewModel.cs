using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services.Maintenance;
using Sharplight.Studio.Services.Platform;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>
/// The file operation running right now (measuring, deleting, exporting, importing): its text, progress and cancellation. The work runs on the
/// thread pool; its progress reports reach the UI thread throttled.
/// </summary>
internal sealed partial class FileOperationViewModel : ObservableObject
{
    private readonly IUiDispatcher _dispatcher;
    private CancellationTokenSource? _cancellation;

    public FileOperationViewModel(IUiDispatcher dispatcher)
    {
        _dispatcher = dispatcher;
    }

    /// <summary>An operation is running.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(CancelCommand))]
    public partial bool IsRunning { get; private set; }

    /// <summary>What is being done.</summary>
    [ObservableProperty]
    public partial string Text { get; private set; } = "";

    /// <summary>0..1.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsIndeterminate))]
    public partial double Progress { get; private set; }

    /// <summary>The operation reports no counts.</summary>
    public bool IsIndeterminate => Progress <= 0;

    /// <summary>Runs <paramref name="work"/> off the UI thread with progress and cancellation.</summary>
    /// <param name="text">What is being done, until the work reports something more specific.</param>
    /// <param name="work">The work; it gets the progress sink and the cancellation token.</param>
    /// <returns>Whether it ran to the end (false when it was cancelled).</returns>
    public async Task<bool> RunAsync(string text, Func<IProgress<MaintenanceProgress>, CancellationToken, Task> work)
    {
        using var cancellation = new CancellationTokenSource();
        _cancellation = cancellation;
        Text = text;
        Progress = 0;
        IsRunning = true;
        using var progress = new DispatchingProgress<MaintenanceProgress>(_dispatcher, Apply, (_, _) => true);
        try
        {
            await Task.Run(() => work(progress, cancellation.Token), CancellationToken.None);
            return true;
        }
        catch (OperationCanceledException)
        {
            return false;
        }
        finally
        {
            progress.Flush();
            IsRunning = false;
            Progress = 0;
            _cancellation = null;
        }
    }

    /// <summary>Asks the running operation to stop; it leaves nothing half written.</summary>
    [RelayCommand(CanExecute = nameof(IsRunning))]
    private void Cancel() => _cancellation?.Cancel();

    private void Apply(MaintenanceProgress update)
    {
        Text = update.Message;
        Progress = update.Fraction;
    }
}
