using System.Net;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Runtime;
using Sharplight.Server;

namespace Sharplight.Serving.Tests;

/// <summary>A memory pool with a fixed capacity; models charge it on load and give it back on dispose. Other applications' use is simulated with <see cref="External"/>.</summary>
public sealed class FakePool(string name, long capacity) : IMemoryDomain
{
    private long _ours, _external;
    public string Id => name;
    public string Name => name;
    public long External { get => Interlocked.Read(ref _external); set => Interlocked.Exchange(ref _external, value); }
    public long Ours => Interlocked.Read(ref _ours);
    public void Charge(long b) => Interlocked.Add(ref _ours, b);
    public DeviceMemoryInfo GetMemoryInfo() => new(capacity, capacity, Ours + External, Ours, true);
}

public sealed class FakeLoader : IModelLoader
{
    private readonly FakePool _pool;
    private readonly long _bytes;
    private readonly bool _offload;
    public int Prepares, Loads;
    public long? LastLimit;
    public Func<FakeBatchDecoder>? MakeDecoder;
    public TimeSpan LoadDelay;
    public string Reply = "ok";
    public List<FakeLoaded> Instances { get; } = [];
    public FakeLoader(FakePool pool, long bytes, bool offload = false) { _pool = pool; _bytes = bytes; _offload = offload; }
    public ModelKind Kind { get; init; } = ModelKind.Chat;

    public IPreparedModel Prepare(CancellationToken ct)
    {
        Interlocked.Increment(ref Prepares);
        return new Prepared(this);
    }

    private sealed class Prepared(FakeLoader o) : IPreparedModel
    {
        public IReadOnlyList<MemoryRequirement> Requirements => [new MemoryRequirement(o._pool, o._bytes)];
        public bool CanOffload => o._offload;
        public LoadedModel Load(long? limit)
        {
            Interlocked.Increment(ref o.Loads);
            o.LastLimit = limit;
            if (o.LoadDelay > TimeSpan.Zero)
            {
                Thread.Sleep(o.LoadDelay);
            }

            long bytes = limit is { } l ? Math.Min(l, o._bytes) : o._bytes;
            var dec = o.MakeDecoder?.Invoke() ?? new FakeBatchDecoder(2);
            dec.Script(o.Reply);
            var inst = new FakeLoaded(o._pool, bytes, o.Kind, dec);
            lock (o.Instances)
            {
                o.Instances.Add(inst);
            }

            return inst;
        }
        public void Dispose() { }
    }
}

public sealed class FakeLoaded : LoadedModel, IChatModel
{
    private readonly FakePool _pool;
    private readonly long _bytes;
    public FakeBatchDecoder Decoder { get; }
    public InferenceEngine Engine { get; }
    public ModelService Service { get; }
    public bool Disposed { get; private set; }
    public FakeLoaded(FakePool pool, long bytes, ModelKind kind, FakeBatchDecoder dec)
    {
        _pool = pool; _bytes = bytes; Kind = kind; Decoder = dec;
        pool.Charge(bytes);
        Engine = new InferenceEngine(dec, new FakeTokenizer());
        Service = new ModelService(Engine, ApiHost.Template, new ModelServiceOptions { ModelName = "x", DefaultMaxTokens = 64 });
    }
    public override ModelKind Kind { get; }
    public override long ResidentBytes => _bytes;
    public override bool IsIdle { get { var m = Engine.GetMetrics(); return m.ActiveSequences == 0 && m.QueueDepth == 0; } }
    public override Task<bool> DrainAsync(TimeSpan timeout) => Engine.DrainAsync(timeout);
    public override void Dispose()
    {
        if (Disposed)
        {
            return;
        }

        Disposed = true;
        Engine.Dispose();
        _pool.Charge(-_bytes);
    }
}

/// <summary>Clock the tests advance by hand.</summary>
internal sealed class ManualTime : TimeProvider
{
    private DateTimeOffset _now = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);
    public override DateTimeOffset GetUtcNow() => _now;
    public void Advance(TimeSpan t) => _now += t;
}

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
        var a = new FakeLoader(pool, GiB); var b = new FakeLoader(pool, GiB);
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

        foreach (var n in new[] { "a", "b", "c" }) { (await host.AcquireAsync(n)).Dispose(); t.Advance(TimeSpan.FromSeconds(1)); }
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
        host.Register(Reg("a", a)); host.Register(Reg("b", b));

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

/// <summary>Routing, listing and admin endpoints over a host with two (lazily loaded) models.</summary>
public sealed class MultiModelApiTests : IAsyncLifetime
{
    private FakePool _pool = null!;
    private FakeLoader _a = null!, _b = null!;
    private ModelHost _host = null!;
    private WebApplication _app = null!;
    private HttpClient _client = null!;

    public async ValueTask InitializeAsync() => await Start(o => { });

    private async Task Start(Action<SharplightServerOptions> configure)
    {
        _pool = new FakePool("gpu", 10L << 30);
        _a = new FakeLoader(_pool, 1L << 30) { Reply = "I am A" };
        _b = new FakeLoader(_pool, 1L << 30) { Reply = "I am B" };
        _host = new ModelHost(new ModelHostOptions { ReserveBytes = 0 });
        _host.Register(new ModelRegistration { Name = "model-a", Loader = _a });
        _host.Register(new ModelRegistration { Name = "model-b", Loader = _b, Description = "second" });
        var b = WebApplication.CreateBuilder();
        b.WebHost.UseTestServer();
        b.Logging.ClearProviders();
        b.Services.AddSharplight(_host, configure);
        _app = b.Build();
        _app.UseSharplight();
        _app.MapSharplight();
        await _app.StartAsync();
        _client = _app.GetTestClient();
    }

    public async ValueTask DisposeAsync()
    {
        _client.Dispose();
        await _app.StopAsync();
        await _app.DisposeAsync();
        _host.Dispose();
    }

    private async Task<(HttpStatusCode, JsonElement)> Post(string path, string json, params (string, string)[] headers)
    {
        var req = new HttpRequestMessage(HttpMethod.Post, path) { Content = new StringContent(json, Encoding.UTF8, "application/json") };
        foreach (var (k, v) in headers)
        {
            req.Headers.TryAddWithoutValidation(k, v);
        }

        using var r = await _client.SendAsync(req, TestContext.Current.CancellationToken);
        var text = await r.Content.ReadAsStringAsync(TestContext.Current.CancellationToken);
        return (r.StatusCode, text.Length == 0 ? default : JsonDocument.Parse(text).RootElement.Clone());
    }

    private string Chat(string model) => $$"""{"model":"{{model}}","messages":[{"role":"user","content":"hi"}],"temperature":0}""";

    [Fact]
    public async Task ModelField_SelectsTheModel_AndLoadsItLazily()
    {
        var (s1, b1) = await Post("/v1/chat/completions", Chat("model-b"));
        Assert.Equal(HttpStatusCode.OK, s1);
        Assert.Equal("I am B", b1.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString());
        Assert.Equal("model-b", b1.GetProperty("model").GetString());
        Assert.Equal(0, _a.Loads);   // A was never touched
        var (_, b2) = await Post("/v1/chat/completions", Chat("model-a"));
        Assert.Equal("I am A", b2.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString());

        var (s3, b3) = await Post("/v1/messages", """{"model":"model-a","max_tokens":20,"messages":[{"role":"user","content":"hi"}]}""", ("anthropic-version", "2023-06-01"));
        Assert.Equal(HttpStatusCode.OK, s3);
        Assert.Equal("I am A", b3.GetProperty("content")[0].GetProperty("text").GetString());
        var (s4, b4) = await Post("/v1/completions", """{"model":"model-b","prompt":[120,1],"max_tokens":10}""");
        Assert.Equal(HttpStatusCode.OK, s4);
        Assert.Equal("I am B", b4.GetProperty("choices")[0].GetProperty("text").GetString());
    }

    [Fact]
    public async Task UnknownModel_Is404_InEachApiShape()
    {
        var (s, b) = await Post("/v1/chat/completions", Chat("gpt-4"));
        Assert.Equal(HttpStatusCode.NotFound, s);
        Assert.Equal("model_not_found", b.GetProperty("error").GetProperty("code").GetString());
        Assert.Equal("invalid_request_error", b.GetProperty("error").GetProperty("type").GetString());
        Assert.Contains("gpt-4", b.GetProperty("error").GetProperty("message").GetString());

        var (s2, b2) = await Post("/v1/messages", """{"model":"claude-x","max_tokens":5,"messages":[{"role":"user","content":"hi"}]}""", ("anthropic-version", "2023-06-01"));
        Assert.Equal(HttpStatusCode.NotFound, s2);
        Assert.Equal("error", b2.GetProperty("type").GetString());
        Assert.Equal("not_found_error", b2.GetProperty("error").GetProperty("type").GetString());

        var (s3, _) = await Post("/v1/completions", """{"model":"nope","prompt":"x"}""");
        Assert.Equal(HttpStatusCode.NotFound, s3);
        using var g = await _client.GetAsync("/v1/models/nope", TestContext.Current.CancellationToken);
        Assert.Equal(HttpStatusCode.NotFound, g.StatusCode);
        Assert.Equal(0, _a.Prepares + _b.Prepares);
    }

    [Fact]
    public async Task NoModelField_UsesTheFirstRegisteredModel()
    {
        var (s, b) = await Post("/v1/chat/completions", """{"messages":[{"role":"user","content":"hi"}]}""");
        Assert.Equal(HttpStatusCode.OK, s);
        Assert.Equal("model-a", b.GetProperty("model").GetString());
    }

    [Fact]
    public async Task ModelList_ShowsAllModels_LoadedOrNot()
    {
        await Post("/v1/chat/completions", Chat("model-b"));
        var list = JsonDocument.Parse(await _client.GetStringAsync("/v1/models", TestContext.Current.CancellationToken)).RootElement.GetProperty("data");
        Assert.Equal(["model-a", "model-b"], list.EnumerateArray().Select(m => m.GetProperty("id").GetString()).ToArray());
        Assert.Equal("unloaded", list[0].GetProperty("sharplight").GetProperty("state").GetString());
        Assert.Equal("loaded", list[1].GetProperty("sharplight").GetProperty("state").GetString());
        Assert.Equal("chat", list[1].GetProperty("sharplight").GetProperty("kind").GetString());

        var req = new HttpRequestMessage(HttpMethod.Get, "/v1/models");
        req.Headers.Add("anthropic-version", "2023-06-01");
        using var r = await _client.SendAsync(req, TestContext.Current.CancellationToken);
        var an = JsonDocument.Parse(await r.Content.ReadAsStringAsync(TestContext.Current.CancellationToken)).RootElement;
        Assert.Equal(2, an.GetProperty("data").GetArrayLength());
        Assert.Equal("model-a", an.GetProperty("first_id").GetString());
        Assert.Equal("model-b", an.GetProperty("last_id").GetString());
    }

    [Fact]
    public async Task Admin_LoadUnload_AndMemoryReport()
    {
        var (s, b) = await Post("/admin/models/model-a/load", "");
        Assert.Equal(HttpStatusCode.OK, s);
        Assert.Equal("loaded", b.GetProperty("state").GetString());
        Assert.Equal(1L << 30, _pool.Ours);
        var report = JsonDocument.Parse(await _client.GetStringAsync("/admin/models", TestContext.Current.CancellationToken)).RootElement;
        Assert.Equal(2, report.GetProperty("models").GetArrayLength());
        Assert.Equal(1L << 30, report.GetProperty("devices")[0].GetProperty("used_by_sharplight_bytes").GetInt64());

        var (s2, b2) = await Post("/admin/models/model-a/unload", "");
        Assert.Equal(HttpStatusCode.OK, s2);
        Assert.Equal("unloaded", b2.GetProperty("state").GetString());
        Assert.Equal(0, _pool.Ours);
        var (s3, _) = await Post("/admin/models/missing/load", "");
        Assert.Equal(HttpStatusCode.NotFound, s3);
    }

    [Fact]
    public async Task Admin_NeedsTheApiKey()
    {
        await DisposeAsync();
        await Start(o => o.ApiKeys.Add("secret"));
        var (s, _) = await Post("/admin/models/model-a/load", "");
        Assert.Equal(HttpStatusCode.Unauthorized, s);
        var (s2, _) = await Post("/admin/models/model-a/load", "", ("Authorization", "Bearer secret"));
        Assert.Equal(HttpStatusCode.OK, s2);
        using var health = await _client.GetAsync("/health", TestContext.Current.CancellationToken);   // stays open
        Assert.Equal(HttpStatusCode.OK, health.StatusCode);
    }

    [Fact]
    public async Task OverBudget_Is503_WithTheReason()
    {
        _pool.External = 9_500L << 20;   // leaves < 1 GiB for a 1 GiB model
        var (s, b) = await Post("/v1/chat/completions", Chat("model-a"));
        Assert.Equal(HttpStatusCode.ServiceUnavailable, s);
        Assert.Equal("insufficient_memory", b.GetProperty("error").GetProperty("code").GetString());
        Assert.Contains("MiB", b.GetProperty("error").GetProperty("message").GetString());
    }

    [Fact]
    public async Task HealthAndMetrics_CoverEveryModel()
    {
        await Post("/v1/chat/completions", Chat("model-a"));
        await Post("/v1/chat/completions", Chat("model-b"));
        var h = JsonDocument.Parse(await _client.GetStringAsync("/health", TestContext.Current.CancellationToken)).RootElement;
        Assert.Equal("ok", h.GetProperty("status").GetString());
        Assert.Equal(2, h.GetProperty("models").GetArrayLength());
        var m = await _client.GetStringAsync("/metrics", TestContext.Current.CancellationToken);
        Assert.Contains("sharplight_requests_submitted_total{model=\"model-a\"} 1", m);
        Assert.Contains("sharplight_requests_submitted_total{model=\"model-b\"} 1", m);
        Assert.Equal(1, System.Text.RegularExpressions.Regex.Matches(m, "# TYPE sharplight_queue_depth ").Count);
        Assert.Contains("sharplight_model_loaded{model=\"model-a\",kind=\"chat\"} 1", m);
        Assert.Contains("sharplight_device_memory_bytes", m);
    }
}
