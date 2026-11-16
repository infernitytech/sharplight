namespace Sharplight.Studio.Tests.Fakes;

/// <summary>Blocking work for the fakes: holds the calling thread like a graph run does (a UI thread running it would freeze).</summary>
internal static class SlowWork
{
    /// <summary>Blocks the calling thread for <paramref name="duration"/> (returns early when cancelled).</summary>
    /// <param name="duration">How long; zero does nothing.</param>
    /// <param name="cancellationToken">Ends the wait early.</param>
    public static void Block(TimeSpan duration, CancellationToken cancellationToken = default)
    {
        if (duration > TimeSpan.Zero)
        {
            cancellationToken.WaitHandle.WaitOne(duration);
        }
    }
}
