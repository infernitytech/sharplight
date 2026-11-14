using Sharplight.Runtime;

namespace Sharplight.Cpu.Tests;

/// <summary>Cancels a token on the first report of a stage, to cancel a compilation while it runs.</summary>
internal sealed class CancelOnReport(LoadStage stage, CancellationTokenSource source) : IProgress<LoadProgress>
{
    public void Report(LoadProgress value)
    {
        if (value.Stage == stage)
        {
            source.Cancel();
        }
    }
}
