using Sharplight.Studio.Services.Threading;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>The throttle that keeps progress, previews and streamed text from flooding the UI thread.</summary>
public sealed class UpdateThrottleTests
{
    private static readonly TimeSpan Interval = TimeSpan.FromMilliseconds(100);

    [Fact]
    public void FiresAtOnceAfterAQuietInterval_AndOnceAtTheEndOfABusyOne()
    {
        var time = new ManualTime();
        int fired = 0;
        using var throttle = new UpdateThrottle(() => fired++, Interval, time);

        throttle.Request();
        Assert.Equal(1, fired);
        for (int i = 0; i < 50; i++)
        {
            throttle.Request();
        }

        Assert.Equal(1, fired);
        time.Advance(Interval);
        Assert.Equal(2, fired);
        time.Advance(Interval * 3);
        Assert.Equal(2, fired);
        throttle.Request();
        Assert.Equal(3, fired);
    }

    [Fact]
    public void Flush_FiresAPendingRequestNow_AndDisposeDropsIt()
    {
        var time = new ManualTime();
        int fired = 0;
        var throttle = new UpdateThrottle(() => fired++, Interval, time);
        throttle.Request();
        throttle.Request();
        throttle.Flush();
        Assert.Equal(2, fired);
        throttle.Flush();
        Assert.Equal(2, fired);
        time.Advance(Interval);
        Assert.Equal(2, fired);

        time.Advance(Interval);
        throttle.Request();
        throttle.Request();
        throttle.Dispose();
        time.Advance(Interval);
        Assert.Equal(3, fired);
    }

    [Fact]
    public void DispatchingProgress_BatchesReports_ReplacesRefinements_AndKeepsTheOrder()
    {
        using var services = new TestServices();
        var applied = new List<string>();
        using var progress = new DispatchingProgress<string>(services.Platform, applied.Add, (previous, next) => previous[0] == next[0], TimeSpan.FromHours(1));
        progress.Report("a1");
        Assert.Equal(["a1"], applied);
        progress.Report("a2");
        progress.Report("a3");
        progress.Report("b1");
        progress.Report("a4");
        Assert.Equal(["a1"], applied);
        progress.Flush();
        Assert.Equal(["a1", "a3", "b1", "a4"], applied);
    }
}
