using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Serving.Tests;

public class ModelHostTests
{
    private const long GiB = 1L << 30;

    private static ModelRegistration Reg(string name, FakeLoader l, bool eager = false, bool pinned = false, TimeSpan? idle = null)
        => new() { Name = name, Loader = l, LoadAtStartup = eager, Pinned = pinned, IdleTimeout = idle };

    private static ModelHost Host(long reserve = 0, int max = 0, TimeProvider? time = null, TimeSpan? idle = null)
        => new(new ModelHostOptions { ReserveBytes = reserve, MaxLoadedModels = max, Time = time ?? TimeProvider.System, IdleTimeout = idle, IdleCheckInterval = TimeSpan.FromHours(1) });

    [Fact]
    public async Task Lazy_ModelLoadsOnFirstAcquire_AndOnlyOnce()
    {
        var pool = new FakePool("gpu", 10 * GiB);
        var l = new FakeLoader(pool, 2 * GiB);
        using var host = Host();
        host.Register(Reg("a", l));
        Assert.Equal(ModelState.Unloaded, host.Get("a")!.State);
        Assert.Equal(0, l.Prepares);

        // Many concurrent first requests cause exactly one load.
        var leases = await Task.WhenAll(Enumerable.Range(0, 12).Select(_ => Task.Run(() => host.AcquireAsync("a"))));
        Assert.Equal(1, l.Prepares);
        Assert.Equal(1, l.Loads);
        Assert.Equal(12, host.Get("a")!.ActiveLeases);
        Assert.Equal(ModelState.Loaded, host.Get("a")!.State);
        Assert.Equal(2 * GiB, pool.Ours);
        foreach (var x in leases)
        {
            x.Dispose();
        }

        Assert.Equal(0, host.Get("a")!.ActiveLeases);
        using (await host.AcquireAsync("a")) { }
        Assert.Equal(1, l.Loads);
    }

    [Fact]
    public async Task Eager_ModelsLoadAtStartup()
    {
        var pool = new FakePool("gpu", 10 * GiB);
        var a = new FakeLoader(pool, GiB);
        var b = new FakeLoader(pool, GiB);
        using var host = Host();
        host.Register(Reg("a", a, eager: true));
        host.Register(Reg("b", b));
        await host.LoadStartupModelsAsync();
        Assert.Equal(ModelState.Loaded, host.Get("a")!.State);
        Assert.Equal(ModelState.Unloaded, host.Get("b")!.State);
    }

    [Fact]
    public async Task UnknownModel_AndWrongKind_Throw()
    {
        var pool = new FakePool("gpu", 10 * GiB);
        using var host = Host();
        host.Register(Reg("emb", new FakeLoader(pool, GiB) { Kind = ModelKind.Embedding }));
        await Assert.ThrowsAsync<ModelNotFoundException>(() => host.AcquireAsync("nope"));
        await Assert.ThrowsAsync<ModelKindMismatchException>(() => host.AcquireAsync("emb", ModelKind.Chat));
        Assert.Equal("emb", host.DefaultName(ModelKind.Embedding));
        Assert.Null(host.DefaultName(ModelKind.Chat));
        using var lease = await host.AcquireAsync("emb", ModelKind.Embedding);   // non-chat kinds register and load the same way
        Assert.Equal(ModelKind.Embedding, lease.Model.Kind);
    }

    [Fact]
    public async Task OverBudget_IsRefused_WithClearError_AndNothingIsLoaded()
    {
        var pool = new FakePool("gpu", 10 * GiB);
        var big = new FakeLoader(pool, 9 * GiB);
        using var host = Host(reserve: 2 * GiB);   // 10 - 2 reserve = 8 usable < 9
        host.Register(Reg("big", big));
        var ex = await Assert.ThrowsAsync<ModelDoesNotFitException>(() => host.AcquireAsync("big"));
        Assert.Equal("insufficient_memory", ex.Code);
        Assert.Contains("9216 MiB", ex.Message);
        Assert.Contains("8192 MiB", ex.Message);
        Assert.Contains("2048 MiB in reserve", ex.Message);
        Assert.Equal(0, big.Loads);
        var info = host.Get("big")!;
        Assert.Equal(ModelState.Failed, info.State);
        Assert.NotNull(info.Error);
        Assert.Equal(0, pool.Ours);
    }

    [Fact]
    public async Task ExternalUsage_ReducesWhatFits()
    {
        var pool = new FakePool("gpu", 10 * GiB) { External = 7 * GiB };   // another application holds 7 GiB
        using var host = Host(reserve: 0);
        host.Register(Reg("m", new FakeLoader(pool, 4 * GiB)));
        await Assert.ThrowsAsync<ModelDoesNotFitException>(() => host.AcquireAsync("m"));
        pool.External = 2 * GiB;   // it frees memory; the retry succeeds
        using var lease = await host.AcquireAsync("m");
        Assert.Equal(ModelState.Loaded, host.Get("m")!.State);
    }

    [Fact]
    public async Task OverBudget_Offloads_WhenLoaderCan()
    {
        var pool = new FakePool("gpu", 10 * GiB) { External = 8 * GiB };
        var l = new FakeLoader(pool, 6 * GiB, offload: true);
        using var host = Host(reserve: 0);
        host.Register(Reg("m", l));
        using var lease = await host.AcquireAsync("m");
        Assert.Equal(2 * GiB, l.LastLimit);          // got only what is free
        Assert.Equal(2 * GiB, lease.Model.ResidentBytes);
    }

    [Fact]
    public async Task Eviction_UnloadsLeastRecentlyUsedIdleModel()
    {
        var pool = new FakePool("gpu", 10 * GiB);
        var t = new ManualTime();
        using var host = Host(time: t);
        var ls = new[] { "a", "b", "c" }.ToDictionary(n => n, _ => new FakeLoader(pool, 4 * GiB));
        foreach (var (n, l) in ls)
        {
            host.Register(Reg(n, l));
        }

        (await host.AcquireAsync("a")).Dispose();
        t.Advance(TimeSpan.FromSeconds(1));
        (await host.AcquireAsync("b")).Dispose();
        t.Advance(TimeSpan.FromSeconds(1));
        (await host.AcquireAsync("a")).Dispose();   // a is now more recent than b
        t.Advance(TimeSpan.FromSeconds(1));
        Assert.Equal(8 * GiB, pool.Ours);

        (await host.AcquireAsync("c")).Dispose();   // needs 4 GiB, 2 free -> evicts b (LRU)
        Assert.Equal(ModelState.Unloaded, host.Get("b")!.State);
        Assert.Equal(ModelState.Loaded, host.Get("a")!.State);
        Assert.Equal(ModelState.Loaded, host.Get("c")!.State);
        Assert.True(ls["b"].Instances[0].Disposed);
        Assert.Equal(8 * GiB, pool.Ours);

        (await host.AcquireAsync("b")).Dispose();   // comes back on demand, evicting the now-LRU a
        Assert.Equal(2, ls["b"].Loads);
        Assert.Equal(ModelState.Unloaded, host.Get("a")!.State);
    }

    [Fact]
    public async Task Eviction_SkipsModelsInUse_AndPinned_ThenRefuses()
    {
        var pool = new FakePool("gpu", 10 * GiB);
        using var host = Host();
        host.Register(Reg("a", new FakeLoader(pool, 4 * GiB)));
        host.Register(Reg("b", new FakeLoader(pool, 4 * GiB), pinned: true));
        host.Register(Reg("c", new FakeLoader(pool, 4 * GiB)));
        var la = await host.AcquireAsync("a");       // in use
        (await host.AcquireAsync("b")).Dispose();    // pinned
        var ex = await Assert.ThrowsAsync<ModelDoesNotFitException>(() => host.AcquireAsync("c"));
        Assert.Contains("a 4096 MiB [in use]", ex.Message);
        Assert.Equal(ModelState.Loaded, host.Get("a")!.State);
        Assert.Equal(ModelState.Loaded, host.Get("b")!.State);
        la.Dispose();
        (await host.AcquireAsync("c")).Dispose();    // a is idle now and gets evicted
        Assert.Equal(ModelState.Unloaded, host.Get("a")!.State);
        Assert.Equal(ModelState.Loaded, host.Get("b")!.State);
    }

    [Fact]
    public async Task MaxLoadedModels_EvictsEvenWhenMemoryIsPlentiful()
    {
        var pool = new FakePool("gpu", 100 * GiB);
        var t = new ManualTime();
        using var host = Host(max: 2, time: t);
        foreach (var n in new[] { "a", "b", "c" })
        {
            host.Register(Reg(n, new FakeLoader(pool, GiB)));
        }

        foreach (var n in new[] { "a", "b", "c" })
        {
            (await host.AcquireAsync(n)).Dispose();
            t.Advance(TimeSpan.FromSeconds(1));
        }
        Assert.Equal(ModelState.Unloaded, host.Get("a")!.State);
        Assert.Equal(2, host.List().Count(m => m.State == ModelState.Loaded));

        using var l1 = await host.AcquireAsync("b");
        using var l2 = await host.AcquireAsync("c");
        var ex = await Assert.ThrowsAsync<ModelBusyException>(() => host.AcquireAsync("a"));
        Assert.Equal("model_busy", ex.Code);
    }

    [Fact]
    public async Task IdleModels_AreUnloaded_AfterTheTimeout()
    {
        var pool = new FakePool("gpu", 10 * GiB);
        var t = new ManualTime();
        using var host = Host(time: t, idle: TimeSpan.FromMinutes(5));
        host.Register(Reg("a", new FakeLoader(pool, GiB)));
        host.Register(Reg("b", new FakeLoader(pool, GiB)));
        host.Register(Reg("keep", new FakeLoader(pool, GiB), pinned: true));
        host.Register(Reg("short", new FakeLoader(pool, GiB), idle: TimeSpan.FromSeconds(10)));
        var busy = await host.AcquireAsync("b");
        foreach (var n in new[] { "a", "keep", "short" })
        {
            (await host.AcquireAsync(n)).Dispose();
        }

        t.Advance(TimeSpan.FromSeconds(30));
        Assert.Equal(1, host.SweepIdle());
        Assert.Equal(ModelState.Unloaded, host.Get("short")!.State);
        t.Advance(TimeSpan.FromMinutes(10));
        Assert.Equal(1, host.SweepIdle());   // a: idle 10 min; b is leased, keep is pinned
        Assert.Equal(ModelState.Unloaded, host.Get("a")!.State);
        Assert.Equal(ModelState.Loaded, host.Get("b")!.State);
        Assert.Equal(ModelState.Loaded, host.Get("keep")!.State);
        busy.Dispose();
        t.Advance(TimeSpan.FromMinutes(6));
        Assert.Equal(1, host.SweepIdle());
        Assert.Equal(GiB, pool.Ours);   // only the pinned model remains
    }

    [Fact]
    public async Task Unload_FreesMemory_WaitsForRunningRequests_AndCanBeForced()
    {
        var pool = new FakePool("gpu", 10 * GiB);
        var l = new FakeLoader(pool, 3 * GiB);
        using var host = Host();
        host.Register(Reg("a", l));
        var lease = await host.AcquireAsync("a");
        Assert.Equal(3 * GiB, pool.Ours);

        var ex = await Assert.ThrowsAsync<ModelBusyException>(() => host.UnloadAsync("a", TimeSpan.FromMilliseconds(50)));
        Assert.Contains("1 request", ex.Message);
        Assert.Equal(ModelState.Loaded, host.Get("a")!.State);   // stays loaded and usable
        using (await host.AcquireAsync("a")) { }

        var unload = host.UnloadAsync("a", TimeSpan.FromSeconds(5));
        await Task.Delay(30);
        Assert.False(unload.IsCompleted);
        lease.Dispose();                                          // the last request ends -> the unload completes
        Assert.True(await unload);
        Assert.Equal(0, pool.Ours);
        Assert.Equal(ModelState.Unloaded, host.Get("a")!.State);
        Assert.False(await host.UnloadAsync("a"));

        var again = await host.AcquireAsync("a");                 // loads again on demand
        Assert.Equal(2, l.Loads);
        Assert.True(await host.UnloadAsync("a", TimeSpan.FromMilliseconds(20), force: true));
        Assert.Equal(0, pool.Ours);
        again.Dispose();
    }

    [Fact]
    public async Task FailedLoad_CanBeRetried()
    {
        var pool = new FakePool("gpu", 10 * GiB);
        var l = new FakeLoader(pool, GiB) { MakeDecoder = () => throw new InvalidOperationException("boom") };
        using var host = Host();
        host.Register(Reg("a", l));
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => host.AcquireAsync("a"));
        Assert.Equal("boom", ex.Message);
        Assert.Equal(ModelState.Failed, host.Get("a")!.State);
        l.MakeDecoder = null;
        using var lease = await host.AcquireAsync("a");
        Assert.Equal(ModelState.Loaded, host.Get("a")!.State);
    }

    [Fact]
    public async Task ConcurrentRequests_ToTwoModels_RunInParallel_AndStayIsolated()
    {
        var pool = new FakePool("gpu", 10 * GiB);
        var step = TimeSpan.FromMilliseconds(15);
        var a = new FakeLoader(pool, GiB) { Reply = "alpha alpha alpha alpha", MakeDecoder = () => new FakeBatchDecoder(2) { StepDelay = step } };
        var b = new FakeLoader(pool, GiB) { Reply = "bravo bravo bravo bravo", MakeDecoder = () => new FakeBatchDecoder(2) { StepDelay = step } };
        using var host = Host();
        host.Register(Reg("a", a));
        host.Register(Reg("b", b));

        async Task<string> Ask(string model)
        {
            using var lease = await host.AcquireAsync(model);
            var svc = ((FakeLoaded)lease.Model).Service;
            var res = await svc.StartChat(new ChatRequest { Messages = [new ChatTurn("user", "hi")], Sampling = SamplingParams.GreedyDefault, MaxTokens = 60 }).CompleteAsync();
            return res.Choices[0].Content;
        }
        await Ask("a"); await Ask("b");   // load both (and warm up)

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var solo = await Ask("a");
        var soloMs = sw.Elapsed.TotalMilliseconds;
        sw.Restart();
        var both = await Task.WhenAll(Enumerable.Range(0, 8).Select(i => Task.Run(() => Ask(i % 2 == 0 ? "a" : "b"))));
        var bothMs = sw.Elapsed.TotalMilliseconds;

        Assert.Equal("alpha alpha alpha alpha", solo);
        for (int i = 0; i < both.Length; i++)
        {
            Assert.Equal(i % 2 == 0 ? "alpha alpha alpha alpha" : "bravo bravo bravo bravo", both[i]);
        }
        // 8 requests over two 2-slot engines = 2 waves each; run serially on one thread they would take ~8x a single request.
        Assert.True(bothMs < soloMs * 5, $"models did not run concurrently: single {soloMs:F0} ms, 8 mixed {bothMs:F0} ms");
        Assert.True(a.Instances[0].Decoder.Steps.Count > 0 && b.Instances[0].Decoder.Steps.Count > 0);
    }

    [Fact]
    public async Task Dispose_UnloadsEverything()
    {
        var pool = new FakePool("gpu", 10 * GiB);
        var host = Host();
        host.Register(Reg("a", new FakeLoader(pool, GiB)));
        host.Register(Reg("b", new FakeLoader(pool, GiB)));
        (await host.AcquireAsync("a")).Dispose();
        (await host.AcquireAsync("b")).Dispose();
        Assert.Equal(2 * GiB, pool.Ours);
        host.Dispose();
        Assert.Equal(0, pool.Ours);
    }
}
