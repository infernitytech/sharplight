using Sharplight.Studio.Services.Maintenance;
using Sharplight.Studio.Tests.Fakes;

namespace Sharplight.Studio.Tests;

/// <summary>Last-used tracking: the JSON store and the recorder that follows the runtime.</summary>
public sealed class ModelUsageTests
{
    [Fact]
    public void TheStoreKeepsTheLastLoadTimeAcrossRestarts_AndForgets()
    {
        using var dir = new TempDir();
        var file = Path.Combine(dir.Path, "data", "model-usage.json");
        var when = new DateTimeOffset(2026, 11, 1, 12, 0, 0, TimeSpan.Zero);
        var store = new JsonModelUsageStore(file);
        Assert.Null(store.LastUsed("/models/a.slm"));

        store.Record("/models/a.slm", when);
        store.Record("/models/b.slm", when.AddDays(1));

        var reopened = new JsonModelUsageStore(file);
        Assert.Equal(when, reopened.LastUsed("/models/a.slm"));
        reopened.Forget("/models/a.slm");
        Assert.Null(new JsonModelUsageStore(file).LastUsed("/models/a.slm"));
        Assert.Equal(when.AddDays(1), new JsonModelUsageStore(file).LastUsed("/models/b.slm"));
        Assert.Empty(Directory.EnumerateFiles(Path.GetDirectoryName(file)!, "*.tmp"));
    }

    [Fact]
    public void ACorruptFileIsIgnored()
    {
        using var dir = new TempDir();
        var file = dir.File("model-usage.json");
        File.WriteAllText(file, "{ not json");

        var store = new JsonModelUsageStore(file);

        Assert.Null(store.LastUsed("/models/a.slm"));
        store.Record("/models/a.slm", DateTimeOffset.UtcNow);
        Assert.NotNull(new JsonModelUsageStore(file).LastUsed("/models/a.slm"));
    }

    [Fact]
    public void TheRecorderStampsEveryModelThatBecomesResident_OnlyOnce()
    {
        using var dir = new TempDir();
        var store = new JsonModelUsageStore(dir.File("usage.json"));
        var runtime = new FakeRuntime();
        var time = new FakeTime(new DateTimeOffset(2026, 11, 1, 0, 0, 0, TimeSpan.Zero));
        using var recorder = new ModelUsageRecorder(runtime, store, time);

        runtime.Add(new FakeModel("a") { Path = "/models/a.slm" });
        Assert.Equal(time.GetUtcNow(), store.LastUsed("/models/a.slm"));

        time.Now = time.Now.AddDays(3);
        runtime.Add(new FakeModel("b") { Path = "/models/b.slm" });
        Assert.Equal(time.GetUtcNow(), store.LastUsed("/models/b.slm"));
        Assert.Equal(time.GetUtcNow().AddDays(-3), store.LastUsed("/models/a.slm"));
    }
}
