using Microsoft.Extensions.DependencyInjection;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Runtime;

namespace Sharplight.Hosting.Tests;

/// <summary><see cref="ModelHostManager"/> over fake models: registration from options, one memory budget for every kind, status.</summary>
public sealed class ModelHostManagerTests
{
    private const long GiB = 1L << 30;

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Fact]
    public async Task ChatEmbeddingAndRerankerShareOneBudgetWithLeastRecentlyUsedEviction()
    {
        var pool = new FakePool("gpu:0", 10 * GiB);
        var factory = new FakeLoaderFactory(pool);
        using var services = TestServices.Create(factory, options =>
        {
            options.Reserve = GiB;
            options.Lazy = true;
            options.Models.Add(new HostedModelOptions { Name = "chat", Path = "fake://5G" });
            options.Models.Add(new HostedModelOptions { Name = "embed", Path = "fake://2G", Kind = ModelKind.Embedding });
            options.Models.Add(new HostedModelOptions { Name = "rerank", Path = "fake://3G", Kind = ModelKind.Reranker });
        });
        var manager = services.GetRequiredService<IModelHostManager>();

        foreach (var name in new[] { "chat", "embed", "rerank" })
        {
            using var lease = await manager.Host.AcquireAsync(name, ct: Ct);
        }

        // 5 + 2 + 3 = 10 GiB does not fit 10 GiB minus the 1 GiB reserve: the least recently used one (chat) made room.
        var status = manager.GetStatus();
        Assert.Equal(["chat", "embed", "rerank"], factory.Loaded);
        Assert.Equal(ModelState.Unloaded, status.Models.Single(model => model.Name == "chat").State);
        Assert.Equal(2, status.LoadedModels);

        using (await manager.Host.AcquireAsync("embed", ct: Ct))
        {
            // embed is now the most recently used, so loading chat again evicts rerank.
        }

        using (await manager.Host.AcquireAsync("chat", ct: Ct))
        {
            status = manager.GetStatus();
        }

        Assert.Equal(ModelState.Unloaded, status.Models.Single(model => model.Name == "rerank").State);
        Assert.Equal([ModelKind.Chat, ModelKind.Embedding], status.Models.Where(model => model.State == ModelState.Loaded).Select(model => model.Kind));
    }

    [Fact]
    public async Task StatusReportsDevicesModelsAndTheirMemory()
    {
        var pool = new FakePool("gpu:0", 16 * GiB);
        using var services = TestServices.Create(new FakeLoaderFactory(pool), options =>
        {
            options.Reserve = 2 * GiB;
            options.MaxLoadedModels = 4;
            options.Models.Add(new HostedModelOptions { Name = "chat", Path = "fake://6G", Pinned = true });
            options.Models.Add(new HostedModelOptions { Name = "embed", Path = "fake://1G", Kind = ModelKind.Embedding, Lazy = true });
        });
        var manager = services.GetRequiredService<IModelHostManager>();
        await manager.Host.LoadStartupModelsAsync(Ct);

        var status = manager.GetStatus();
        var device = Assert.Single(status.Devices);
        Assert.Equal("gpu:0", device.Id);
        Assert.Equal(16 * GiB, device.TotalBytes);
        Assert.Equal(10 * GiB, device.FreeBytes);
        Assert.Equal(8 * GiB, device.AvailableBytes);
        Assert.Equal(6 * GiB, device.ModelBytes);
        Assert.Equal(["chat"], device.Models);
        Assert.Equal(4, status.MaxLoadedModels);
        Assert.Equal(2 * GiB, status.ReserveBytes);
        var chat = status.Models[0];
        Assert.Equal(ModelState.Loaded, chat.State);
        Assert.True(chat.Pinned);
        Assert.Equal(["gpu:0"], chat.Devices);
        Assert.Equal(ModelState.Unloaded, status.Models[1].State);

        var again = HostStatus.FromJson(status.ToJson());
        Assert.Equal(status.Models.Select(model => (model.Name, model.State)), again.Models.Select(model => (model.Name, model.State)));
        Assert.Contains("\"availableBytes\"", status.ToJson(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task ModelsCanBeAddedUnloadedAndRemovedAtRunTime()
    {
        var pool = new FakePool("gpu:0", 16 * GiB);
        using var services = TestServices.Create(new FakeLoaderFactory(pool), _ => { });
        var manager = services.GetRequiredService<IModelHostManager>();

        var settings = manager.Register(new HostedModelOptions { Name = "late", Path = "fake://1G", Kind = ModelKind.Reranker, ContextLength = 512 });
        Assert.Equal(512, settings.ContextLength);
        Assert.Same(settings, manager.Find("late"));
        Assert.Throws<InvalidOperationException>(() => manager.Register(new HostedModelOptions { Name = "late", Path = "fake://1G" }));

        await manager.LoadAsync("late", Ct);
        Assert.Equal(GiB, pool.GetMemoryInfo().UsedByUs);
        Assert.True(await manager.UnloadAsync("late", cancellationToken: Ct));
        Assert.Equal(0, pool.GetMemoryInfo().UsedByUs);
        Assert.True(await manager.RemoveAsync("late", cancellationToken: Ct));
        Assert.Null(manager.Find("late"));
        Assert.False(manager.Host.Contains("late"));
    }

    [Fact]
    public void ModelsWithoutALoaderAreRejected()
    {
        using var services = TestServices.Create(new FakeLoaderFactory(new FakePool("gpu:0", GiB)), _ => { });
        var manager = services.GetRequiredService<IModelHostManager>();
        // The built-in task factory needs a task kind.
        Assert.Throws<ArgumentException>(() => manager.Register(new HostedModelOptions { Name = "t", Path = "/m.slm", Kind = ModelKind.Task }));
        Assert.Throws<ArgumentException>(() => manager.Register(new HostedModelOptions { Name = "", Path = "/m.slm" }));
    }

    [Fact]
    public async Task OnDemandModelsComeFromTheResolver()
    {
        var pool = new FakePool("gpu:0", 16 * GiB);
        var resolver = new FakeResolver();
        var services = new ServiceCollection();
        services.AddSingleton(new ExecutionProviderCatalog([]));
        services.AddSingleton<IHostedModelResolver>(resolver);
        services.AddSharplightHosting(options => options.OnDemand = true);
        services.AddSingleton<Loaders.IHostedModelLoaderFactory>(new FakeLoaderFactory(pool));
        using var provider = services.BuildServiceProvider();
        var manager = provider.GetRequiredService<IModelHostManager>();

        Assert.True(manager.Host.Contains("listed"));
        Assert.False(manager.Host.Contains("store/other"));
        using (var lease = await manager.Host.AcquireAsync("store/other", ct: Ct))
        {
            Assert.Equal("store/other", lease.Name);
        }

        Assert.True(manager.Find("store/other")!.Lazy);
        await Assert.ThrowsAsync<ModelNotFoundException>(() => manager.Host.AcquireAsync("missing", ct: Ct));
    }

    [Fact]
    public async Task PlanFitCountsEvictableModels()
    {
        var pool = new FakePool("gpu:0", 10 * GiB);
        using var services = TestServices.Create(new FakeLoaderFactory(pool), options =>
        {
            options.Reserve = GiB;
            options.Models.Add(new HostedModelOptions { Name = "a", Path = "fake://6G" });
        });
        var manager = services.GetRequiredService<IModelHostManager>();
        await manager.Host.LoadStartupModelsAsync(Ct);

        // 7 GiB do not fit the 3 GiB free beyond the reserve, but do once the idle model is evicted.
        var report = manager.PlanFit(new ModelFootprint(5 * GiB, GiB, GiB), "gpu:0");
        Assert.Equal(Fit.FitVerdict.FitsAfterUnloading, report.Verdict);
        Assert.Equal(["a"], report.Unload);
        Assert.True(report.UnloadIsAutomatic);
        Assert.Equal(Fit.FitVerdict.Fits, manager.PlanFit(new ModelFootprint(GiB, GiB, GiB)).Verdict);
    }
}
