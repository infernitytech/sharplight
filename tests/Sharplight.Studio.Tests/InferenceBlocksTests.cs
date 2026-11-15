using Sharplight.Studio.Services.Pages;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;

namespace Sharplight.Studio.Tests;

/// <summary>The blocks every inference page shares: parameter panel with presets, run queue, history, outputs gallery and export.</summary>
public sealed class InferenceBlocksTests
{
    private static ParameterPanelViewModel<(int Beams, string? Language, bool Fast)> Panel(IPageStorage storage) => new(
        storage, "blocks",
        [
            new NumberParameterViewModel("beam", "Beams", "", 1, 8, 1, 1),
            new ChoiceParameterViewModel("language", "Language", "", [new("", "Auto"), new("en", "English"), new("de", "German")]),
            new BoolParameterViewModel("fast", "Fast", ""),
            new TextParameterViewModel("prompt", "Prompt", ""),
        ],
        panel => (panel.GetInt("beam"), panel.GetChoice("language") is { Length: > 0 } language ? language : null, panel.GetBool("fast")),
        [new ParameterPreset("Accurate", new() { ["beam"] = "5", ["language"] = "en" }, BuiltIn: true)]);

    [Fact]
    public void ParameterPanel_MapsToTheOptions_AndAppliesBuiltInPresets()
    {
        using var services = new TestServices();
        var panel = Panel(services.Services.Pages);
        Assert.Equal((1, null, false), panel.Options);

        panel.SelectedPreset = panel.Presets.Single(preset => preset.Name == "Accurate");
        Assert.Equal((5, "en", false), panel.Options);
        Assert.False(panel.IsModified);

        panel.Parameters.OfType<BoolParameterViewModel>().Single().Value = true;
        Assert.True(panel.IsModified);
        Assert.Equal((5, "en", true), panel.Options);

        panel.ResetToDefaultsCommand.Execute(null);
        Assert.Equal((1, null, false), panel.Options);
        Assert.Null(panel.SelectedPreset);
    }

    [Fact]
    public void ParameterPanel_SavesPresetsAsJson_AndRestoresTheLastValues()
    {
        using var services = new TestServices();
        var storage = services.Services.Pages;
        var panel = Panel(storage);
        Assert.False(panel.SavePresetCommand.CanExecute(null));

        panel.Parameters.OfType<NumberParameterViewModel>().Single().Value = 3;
        panel.Parameters.OfType<ChoiceParameterViewModel>().Single().Deserialize("de");
        panel.PresetName = "German, 3 beams";
        Assert.True(panel.SavePresetCommand.CanExecute(null));
        panel.SavePresetCommand.Execute(null);
        panel.Parameters.OfType<TextParameterViewModel>().Single().Value = "names";
        panel.Persist();

        string json = File.ReadAllText(Path.Combine(storage.DirectoryFor("blocks"), "presets.json"));
        Assert.Contains("German, 3 beams", json, StringComparison.Ordinal);

        var reopened = Panel(storage);
        Assert.Contains(reopened.Presets, preset => preset.Name == "German, 3 beams" && !preset.BuiltIn);
        Assert.Equal((3, "de", false), reopened.Options);
        Assert.Equal("names", reopened.GetText("prompt"));

        reopened.SelectedPreset = reopened.Presets.Single(preset => preset.Name == "German, 3 beams");
        reopened.DeletePresetCommand.Execute(null);
        Assert.DoesNotContain(Panel(storage).Presets, preset => preset.Name == "German, 3 beams");
    }

    [Fact]
    public void ParameterPanel_IgnoresValuesThatDoNotFit()
    {
        using var services = new TestServices();
        var panel = Panel(services.Services.Pages);
        panel.Apply(new Dictionary<string, string> { ["beam"] = "99", ["language"] = "xx", ["fast"] = "maybe", ["unknown"] = "1" });
        Assert.Equal((8, null, false), panel.Options);
    }

    [Fact]
    public async Task RunQueue_RunsOneAtATimePerModel_AndModelsInParallel()
    {
        using var services = new TestServices();
        var queue = new RunQueueViewModel(services.Platform);
        var first = new TaskCompletionSource();
        var order = new List<string>();
        var a1 = queue.Enqueue("a1", "model-a", async context =>
        {
            order.Add("a1 start");
            context.Report(0.5, "halfway");
            await first.Task;
            order.Add("a1 end");
        });
        var a2 = queue.Enqueue("a2", "model-a", _ =>
        {
            order.Add("a2");
            return Task.CompletedTask;
        });
        var b1 = queue.Enqueue("b1", "model-b", _ =>
        {
            order.Add("b1");
            return Task.CompletedTask;
        });

        await b1.Completion.WaitAsync(TestContext.Current.CancellationToken);
        Assert.Equal(RunStatus.Completed, b1.Status);
        Assert.Equal(RunStatus.Queued, a2.Status);
        Assert.True(queue.IsBusy);
        for (int i = 0; i < 200 && a1.Progress < 0.5; i++)
        {
            await Task.Delay(10, TestContext.Current.CancellationToken);
        }

        Assert.Equal(0.5, a1.Progress);
        Assert.Contains("halfway", a1.StatusText, StringComparison.Ordinal);

        first.SetResult();
        await Task.WhenAll(a1.Completion, a2.Completion).WaitAsync(TestContext.Current.CancellationToken);
        Assert.Equal(["a1 start", "a1 end"], order.Where(entry => entry.StartsWith("a1", StringComparison.Ordinal)));
        Assert.True(order.IndexOf("a2") > order.IndexOf("a1 end"));
        Assert.False(queue.IsBusy);
        queue.ClearFinishedCommand.Execute(null);
        Assert.Empty(queue.Items);
    }

    [Fact]
    public async Task RunQueue_CancelsQueuedAndRunningRuns_AndReportsFailures()
    {
        using var services = new TestServices();
        var queue = new RunQueueViewModel(services.Platform);
        var running = queue.Enqueue("running", "m", async context => await Task.Delay(Timeout.Infinite, context.CancellationToken));
        var queued = queue.Enqueue("queued", "m", _ => Task.CompletedTask);
        for (int i = 0; i < 200 && running.Status != RunStatus.Running; i++)
        {
            await Task.Delay(10, TestContext.Current.CancellationToken);
        }

        queued.CancelCommand.Execute(null);
        running.CancelCommand.Execute(null);
        await Task.WhenAll(running.Completion, queued.Completion).WaitAsync(TestContext.Current.CancellationToken);
        Assert.Equal(RunStatus.Cancelled, running.Status);
        Assert.Equal(RunStatus.Cancelled, queued.Status);

        var failing = queue.Enqueue("failing", "m", _ => throw new InvalidOperationException("boom"));
        await failing.Completion.WaitAsync(TestContext.Current.CancellationToken);
        Assert.Equal(RunStatus.Failed, failing.Status);
        Assert.Contains("boom", failing.StatusText, StringComparison.Ordinal);
        Assert.False(queue.IsBusy);
    }

    [Fact]
    public async Task HistoryAndGallery_PersistPerPage_AndOfferOpenExportDelete()
    {
        using var services = new TestServices();
        var storage = services.Services.Pages;
        var restored = new List<string>();
        var archive = new PageArchive(storage, "blocks");
        var history = new RunHistoryViewModel(archive, record => restored.Add(record.Id));
        var gallery = new OutputGalleryViewModel(archive, services.Platform);
        Assert.True(history.IsEmpty);
        Assert.True(gallery.IsEmpty);

        archive.Add(new RunRecord
        {
            Id = "one", Title = "talk.wav", Model = "whisper", Seconds = 1.5, Preview = "Hello there", Files = ["out.txt"],
            Metadata = new() { ["language"] = "en" },
        }, new Dictionary<string, byte[]> { ["out.txt"] = "Hello there"u8.ToArray() });
        archive.Add(new RunRecord { Id = "two", Title = "bad.wav", Model = "whisper", Outcome = RunOutcome.Failed, Error = "no audio" });

        Assert.Equal(["bad.wav", "talk.wav"], history.Items.Select(item => item.Title));
        Assert.True(history.Items[0].IsFailed);
        Assert.False(history.Items[0].CanRestore);
        Assert.Single(gallery.Items);
        var item = gallery.Items[0];
        Assert.Equal(["language: en"], item.Facts);

        history.RestoreCommand.Execute(history.Items[1]);
        Assert.Equal(["one"], restored);

        item.OpenCommand.Execute(null);
        Assert.Equal(Path.Combine(archive.DirectoryOf(item.Record), "out.txt"), services.Platform.Opened.Single());

        string exported = Path.Combine(services.Directory, "exported.txt");
        services.Platform.SaveTo = exported;
        await item.Files[0].ExportCommand.ExecuteAsync(null);
        Assert.Equal("Hello there", await File.ReadAllTextAsync(exported, TestContext.Current.CancellationToken));

        // A second archive over the same storage sees the same runs.
        Assert.Equal(2, new PageArchive(storage, "blocks").Runs.Count);
        Assert.Empty(new PageArchive(storage, "other-page").Runs);

        item.DeleteCommand.Execute(null);
        Assert.Equal(2, history.Items.Count);   // the first click only asks
        item.DeleteCommand.Execute(null);
        Assert.Single(history.Items);
        Assert.True(gallery.IsEmpty);
        Assert.False(Directory.Exists(archive.DirectoryOf(item.Record)));
    }

    [Fact]
    public void Archive_KeepsAtMostItsCapacity()
    {
        using var services = new TestServices();
        var archive = new PageArchive(services.Services.Pages, "capacity");
        for (int i = 0; i < PageArchive.Capacity + 5; i++)
        {
            archive.Add(new RunRecord { Id = $"run-{i}", Title = $"run {i}" });
        }

        Assert.Equal(PageArchive.Capacity, archive.Runs.Count);
        Assert.Equal($"run-{PageArchive.Capacity + 4}", archive.Runs[0].Id);
    }
}
