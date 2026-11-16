using System.Diagnostics;
using Avalonia.Threading;

namespace Sharplight.Studio.Tests;

/// <summary>
/// Measures how responsive the UI thread stays: a beat is scheduled on the UI thread every <see cref="Interval"/>, and the largest delay
/// of a beat beyond its interval is recorded (a UI thread busy with inference, or flooded with updates, delays the beats).
/// </summary>
internal sealed class UiHeartbeat
{
    /// <summary>The time between two beats.</summary>
    public static readonly TimeSpan Interval = TimeSpan.FromMilliseconds(50);

    /// <summary>
    /// The largest delay a beat may have. The spec asks for about 200 ms; a little more leaves room for a loaded test machine while a
    /// blocked UI thread (the fakes block for 600 ms per step) still fails clearly.
    /// </summary>
    public static readonly TimeSpan Tolerance = TimeSpan.FromMilliseconds(300);

    private readonly DispatcherTimer _timer;
    private readonly Stopwatch _watch = Stopwatch.StartNew();
    private TimeSpan _last;

    private UiHeartbeat()
    {
        _timer = new DispatcherTimer(Interval, DispatcherPriority.Input, (_, _) => Beat());
    }

    /// <summary>The largest delay of a beat so far.</summary>
    public TimeSpan WorstDelay { get; private set; }

    /// <summary>The beats so far.</summary>
    public int Beats { get; private set; }

    /// <summary>Starts beating (call on the UI thread).</summary>
    /// <returns>The heartbeat.</returns>
    public static UiHeartbeat Start()
    {
        Dispatcher.UIThread.VerifyAccess();
        var heartbeat = new UiHeartbeat();
        heartbeat._last = heartbeat._watch.Elapsed;
        heartbeat._timer.Start();
        return heartbeat;
    }

    /// <summary>Stops beating and checks that no beat was late by more than <see cref="Tolerance"/> and that beats happened at all.</summary>
    /// <param name="what">What ran meanwhile (for the message).</param>
    public void StopAndAssertResponsive(string what)
    {
        _timer.Stop();
        Beat();
        Assert.True(Beats >= 2, $"{what}: the UI thread beat only {Beats} times");
        Assert.True(WorstDelay <= Tolerance, $"{what}: a UI heartbeat was {WorstDelay.TotalMilliseconds:F0} ms late (at most {Tolerance.TotalMilliseconds:F0} ms)");
    }

    private void Beat()
    {
        var now = _watch.Elapsed;
        var late = now - _last - Interval;
        if (late > WorstDelay)
        {
            WorstDelay = late;
        }

        _last = now;
        Beats++;
    }
}
