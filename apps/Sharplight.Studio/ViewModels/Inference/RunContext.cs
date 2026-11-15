namespace Sharplight.Studio.ViewModels.Inference;

/// <summary>What a queued run receives: its cancellation token and a way to report progress (callable from any thread).</summary>
internal sealed class RunContext
{
    private readonly Action<double?, string?> _report;

    internal RunContext(CancellationToken cancellationToken, Action<double?, string?> report)
    {
        CancellationToken = cancellationToken;
        _report = report;
    }

    /// <summary>Cancelled when the user cancels the run.</summary>
    public CancellationToken CancellationToken { get; }

    /// <summary>Reports progress.</summary>
    /// <param name="fraction">Progress from 0 to 1, or null when it cannot be told.</param>
    /// <param name="text">What the run is doing, or null to keep the text.</param>
    public void Report(double? fraction, string? text = null) => _report(fraction, text);
}
