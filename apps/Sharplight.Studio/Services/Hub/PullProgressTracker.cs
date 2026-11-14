using Sharplight.Hub;
using Sharplight.Runtime;

namespace Sharplight.Studio.Services.Hub;

/// <summary>Turns the puller's log lines and download progress into <see cref="PullUpdate"/>s with the current phase.</summary>
internal sealed class PullProgressTracker : IProgress<DownloadProgress>
{
    private readonly IProgress<PullUpdate> _target;
    private PullPhase _phase = PullPhase.Resolving;
    private string _message = "";
    private DownloadProgress _download;
    private long _stageCompleted;
    private long _stageTotal;

    public PullProgressTracker(IProgress<PullUpdate> target)
    {
        _target = target;
    }

    /// <summary>A log line of the puller; its wording tells the phase.</summary>
    public void Log(string message)
    {
        var phase = PhaseOf(message, _phase);
        if (phase != _phase)
        {
            _stageCompleted = _stageTotal = 0;
        }

        _phase = phase;
        _message = message;
        Publish();
    }

    /// <summary>Reports a phase change.</summary>
    public void Report(PullPhase phase, string message)
    {
        _phase = phase;
        _message = message;
        Publish();
    }

    /// <summary>Kernel compilation progress of the precompile step.</summary>
    public void Report(LoadProgress value)
    {
        if (value.Stage != LoadStage.Compiling)
        {
            return;
        }

        _phase = PullPhase.Precompiling;
        _stageCompleted = value.Completed;
        _stageTotal = value.Total;
        _message = LoadProgressTrackerText(value);
        Publish();
    }

    private static string LoadProgressTrackerText(LoadProgress value) => Runtime.LoadProgressTracker.Describe(value);

    /// <inheritdoc/>
    public void Report(DownloadProgress value)
    {
        _download = value;
        _phase = PullPhase.Downloading;
        Publish();
    }

    /// <summary>The phase a puller log line starts (or <paramref name="current"/>).</summary>
    internal static PullPhase PhaseOf(string message, PullPhase current)
    {
        if (message.StartsWith("Pulling ", StringComparison.Ordinal))
        {
            return PullPhase.Downloading;
        }

        if (message.StartsWith("Converting", StringComparison.Ordinal))
        {
            return PullPhase.Converting;
        }

        if (message.StartsWith("Precompil", StringComparison.Ordinal))
        {
            return PullPhase.Precompiling;
        }

        return current;
    }

    private void Publish()
    {
        _target.Report(new PullUpdate(_phase, _message, _download.Received, _download.Total, _download.File, _download.FileReceived, _download.FileSize,
            _stageCompleted, _stageTotal));
    }
}
