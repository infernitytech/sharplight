using Sharplight.Hub;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Hub;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

public sealed class DownloadsViewModelTests
{
    private static readonly ModelReference s_reference = ModelReference.Parse("Qwen/Qwen3-0.6B-GGUF:Q8_0");

    private static async Task WaitFor(Func<bool> condition)
    {
        for (int i = 0; i < 500 && !condition(); i++)
        {
            await Task.Delay(10, TestContext.Current.CancellationToken);
        }

        Assert.True(condition());
    }

    [Fact]
    public async Task Download_ReportsProgress_PersistsTheQueue_AndAnnouncesTheModel()
    {
        using var services = new TestServices();
        var downloads = new DownloadsViewModel(services.Services);
        ModelManifest? stored = null;
        downloads.ModelStored += (_, manifest) => stored = manifest;

        var item = downloads.Enqueue(s_reference, 1000);
        await services.Hub.Started.Task.WaitAsync(TestContext.Current.CancellationToken);
        await WaitFor(() => item.Phase == PullPhase.Downloading);

        Assert.Equal(1, downloads.ActiveCount);
        Assert.Equal(0.4, downloads.OverallProgress, 3);
        Assert.Equal((400, 1000), (item.FileReceived, item.FileSize));
        Assert.Contains("model.gguf", item.FileText);
        Assert.Equal(new PendingDownload(s_reference.ToString(), 1000, false), Assert.Single(services.Settings.Current.Downloads));

        services.Hub.Release();
        await WaitFor(() => item.Phase == PullPhase.Completed);
        Assert.NotNull(stored);
        Assert.Equal(0, downloads.ActiveCount);
        Assert.Empty(services.Settings.Current.Downloads);
    }

    [Fact]
    public async Task Pause_KeepsThePartsAndSurvivesARestart_Cancel_DeletesThem()
    {
        using var services = new TestServices();
        var downloads = new DownloadsViewModel(services.Services);
        var item = downloads.Enqueue(s_reference, 1000);
        await services.Hub.Started.Task.WaitAsync(TestContext.Current.CancellationToken);

        item.PauseCommand.Execute(null);
        await WaitFor(() => item.Phase == PullPhase.Paused);
        Assert.Empty(services.Hub.Discarded);
        Assert.True(Assert.Single(services.Settings.Current.Downloads).Paused);

        // A new session restores the paused download without starting it.
        var restored = new DownloadsViewModel(services.Services);
        restored.Restore();
        var again = Assert.Single(restored.Items);
        Assert.True(again.IsPaused);
        Assert.Equal(1, services.Hub.Pulls);

        // Resume continues; cancelling a running pull discards its parts.
        services.Hub.Reset();
        _ = again.ResumeCommand.ExecuteAsync(null);
        await services.Hub.Started.Task.WaitAsync(TestContext.Current.CancellationToken);
        again.CancelCommand.Execute(null);
        await WaitFor(() => again.Phase == PullPhase.Cancelled);
        Assert.Equal(s_reference, Assert.Single(services.Hub.Discarded));
        Assert.Empty(services.Settings.Current.Downloads);
    }

    [Fact]
    public async Task Restore_ResumesInterruptedDownloads_AndSkipsInvalidEntries()
    {
        using var services = new TestServices(new AppSettings
        {
            Downloads = [new PendingDownload(s_reference.ToString(), 1000, false), new PendingDownload("not a reference", 0, false)],
        });
        var downloads = new DownloadsViewModel(services.Services);
        downloads.Restore();
        await services.Hub.Started.Task.WaitAsync(TestContext.Current.CancellationToken);
        Assert.Single(downloads.Items);
        Assert.Equal(1, services.Hub.Pulls);
        services.Hub.Release();
        await WaitFor(() => downloads.Items[0].Phase == PullPhase.Completed);
    }

    [Fact]
    public async Task Enqueue_TwiceForTheSameModel_ReusesTheEntry()
    {
        using var services = new TestServices();
        var downloads = new DownloadsViewModel(services.Services);
        var first = downloads.Enqueue(s_reference, 1000);
        Assert.Same(first, downloads.Enqueue(s_reference, 1000));
        Assert.Single(downloads.Items);
        services.Hub.Release();
        await WaitFor(() => first.Phase == PullPhase.Completed);
    }
}
