using Sharplight.Runtime;
using Sharplight.Studio.Services.Hub;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.Tests;

public sealed class ProgressTrackerTests
{
    [Fact]
    public void LoadProgress_IsWeightedOverTheStages_AndOnlyReadyCompletes()
    {
        var tracker = new LoadProgressTracker();
        var reading = tracker.Report(new LoadProgress(LoadStage.Reading, 1, 2, null));
        var compiling = tracker.Report(new LoadProgress(LoadStage.Compiling, 5, 10, null) { Cached = 2 });
        var uploading = tracker.Report(new LoadProgress(LoadStage.Uploading, 1_500_000_000, 3_000_000_000, null));
        var ready = tracker.Report(new LoadProgress(LoadStage.Ready, 1, 1, null));

        Assert.True(reading.Fraction < compiling.Fraction && compiling.Fraction < uploading.Fraction && uploading.Fraction < 1);
        Assert.Contains("Compiling kernels 5 / 10 (2 cached)", compiling.Text);
        Assert.Contains("Uploading weights 1.50 / 3.00 GB", uploading.Text);
        Assert.False(uploading.IsDone);
        Assert.True(ready.IsDone);
        Assert.Equal(1, ready.Fraction);
    }

    [Fact]
    public void LoadProgress_CountsSkippedStagesAsDone()
    {
        var status = new LoadProgressTracker().Report(new LoadProgress(LoadStage.Uploading, 0, 100, null));
        Assert.Equal(0.55, status.Fraction, 3);
    }

    [Fact]
    public void Pull_PhasesFollowTheLogAndTheStructuredProgress()
    {
        var updates = new List<PullUpdate>();
        var tracker = new PullProgressTracker(new SynchronousProgress<PullUpdate>(updates.Add));
        tracker.Log("Pulling Qwen/Qwen3-0.6B-GGUF:Q8_0");
        Assert.Equal(PullPhase.Downloading, updates[^1].Phase);
        tracker.Log("Converting model.gguf");
        Assert.Equal(PullPhase.Converting, updates[^1].Phase);
        tracker.Report(new LoadProgress(LoadStage.Compiling, 3, 40, null));
        Assert.Equal(PullPhase.Precompiling, updates[^1].Phase);
        Assert.Equal((3, 40), (updates[^1].StageCompleted, updates[^1].StageTotal));
        tracker.Report(new LoadProgress(LoadStage.Uploading, 3, 40, null));   // not a pull stage
        Assert.Equal(3, updates.Count);
        Assert.Equal(PullPhase.Converting, PullProgressTracker.PhaseOf("something else", PullPhase.Converting));
    }
}
