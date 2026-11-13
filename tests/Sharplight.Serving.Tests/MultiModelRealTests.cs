using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Serving.Tests;

/// <summary>
/// Two real small models on one shared Vulkan device (skipped when the files or a GPU are missing): generating from both at the
/// same time on two threads gives the same tokens as running each alone, memory is accounted, and unloading returns it.
/// </summary>
public class MultiModelRealTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    private static readonly string PathA = Path.Combine(Home, "models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf");
    private static readonly string PathB = Path.Combine(Home, "models/gguf/SmolLM2-360M-Instruct/smollm2-360m-instruct-q8_0.gguf");
    private const long MiB = 1L << 20;

    private static DevicePool? Setup(out IComputeDevice device, DeviceOptions? options = null)
    {
        device = null!;
        Assert.SkipUnless(File.Exists(PathA) && File.Exists(PathB), "models not found");
        ProviderRegistry.Register(() => new VulkanProvider());
        CpuExecutionProvider.Register();
        GgufImporter.Register();
        var vk = ProviderRegistry.Discover().FirstOrDefault(x => x.Provider.Name == "vulkan" && x.Availability.IsAvailable).Provider;
        Assert.SkipUnless(vk is not null && vk.GetDevices().Count > 0, "no Vulkan device");
        var pool = new DevicePool();
        device = pool.Open(vk!, vk!.GetDevices()[0], options);
        return pool;
    }

    private static LlmLoadOptions Options(IComputeDevice device) => new() { Devices = [new DeviceSlot { Device = device }], ContextLength = 1024, MaxBatchTokens = 128 };

    /// <summary>Greedy generation straight on a decoder: prefill the prompt, then <paramref name="n"/> argmax tokens.</summary>
    private static int[] Greedy(IBatchDecoder dec, LlmModel model, string prompt, int n, int slot = 0)
    {
        dec.Reset(slot);
        var toks = model.Tokenizer.Encode(prompt, addSpecial: true, parseSpecial: false);
        var outTokens = new List<int>();
        var input = toks;
        for (int i = 0; i < n; i++)
        {
            var lg = dec.Step([new SequenceInput(slot, input)])[0]!;
            int best = 0;
            for (int k = 1; k < lg.Length; k++)
            {
                if (lg[k] > lg[best])
                {
                    best = k;
                }
            }

            outTokens.Add(best);
            input = [best];
        }
        return [.. outTokens];
    }

    [Fact]
    public void DevicePool_OpensEachDeviceOnce()
    {
        using var pool = Setup(out var device)!;
        var vk = ProviderRegistry.Discover().First(x => x.Provider.Name == "vulkan" && x.Availability.IsAvailable).Provider;
        Assert.Same(device, pool.Open(vk, vk.GetDevices()[0]));
        Assert.Same(device, pool.Open("vulkan", 0));
        var mi = device.GetMemoryInfo();
        Console.Error.WriteLine($"[device] {device.Info.Name}: {mi}");
        Assert.True(mi.Total > 0 && mi.Budget > 0 && mi.Budget <= mi.Total);
        Assert.True(mi.Available > 0);
    }

    [Theory]
    [InlineData("Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf", 1, 1024, 128)]
    [InlineData("Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf", 4, 4096, 512)]
    [InlineData("SmolLM2-360M-Instruct/smollm2-360m-instruct-q8_0.gguf", 4, 2048, 512)]
    [InlineData("Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-Q8_0.gguf", 2, 8192, 512)]
    [InlineData("gemma-3-1b-it/gemma-3-1b-it-Q8_0.gguf", 4, 4096, 256)]
    [InlineData("Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf", 4, 2048, 256)]
    [InlineData("OLMoE-1B-7B-0924-Instruct/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf", 2, 2048, 256)]
    public void Estimate_CoversTheActualDeviceUsage(string rel, int slots, int ctx, int batch)
    {
        var path = Path.Combine(Home, "models/gguf", rel);
        Assert.SkipUnless(File.Exists(path), "model not found");
        using var pool = Setup(out var device)!;
        using var model = LlmModel.Load(path, new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = ctx, MaxBatchTokens = batch });
        var fp = model.EstimateFootprint(slots, ctx);
        using var dec = model.CreateBatchDecoder(slots, ctx);
        var mi = device.GetMemoryInfo();
        Console.Error.WriteLine($"[estimate] {rel} slots={slots} ctx={ctx} batch={batch}: est {fp}; compiled {dec.DeviceMemoryBytes / MiB} MiB; device used by us {mi.UsedByUs / MiB} MiB; driver {mi.Usage / MiB} MiB");
        Assert.True(fp.Total >= mi.UsedByUs, "estimate below real usage");
        Assert.True(fp.Total <= mi.UsedByUs * 2 + 256 * MiB, "estimate far above real usage");
    }

    [Fact]
    public void TwoModels_OneDevice_Concurrent_EqualsSeparate_AndMemoryReturns()
    {
        using var pool = Setup(out var device)!;
        var baseline = device.GetMemoryInfo();
        Console.Error.WriteLine($"[baseline] {baseline}");
        const string PromptA = "The capital of France is";
        const string PromptB = "Once upon a time, in a small village";
        const int N = 48;

        // ---- each model alone ----
        int[] soloA, soloB;
        long estA, estB, actA, actB;
        using (var ma = LlmModel.Load(PathA, Options(device)))
        {
            estA = ma.EstimateFootprint(2, 1024).Total;
            using var da = ma.CreateBatchDecoder(2, 1024);
            actA = da.DeviceMemoryBytes;
            var after = device.GetMemoryInfo();
            Console.Error.WriteLine($"[A alone] estimated {estA / MiB} MiB, compiled {actA / MiB} MiB, device used by us {after.UsedByUs / MiB} MiB, driver usage {after.Usage / MiB} MiB");
            soloA = Greedy(da, ma, PromptA, N);
        }
        var afterA = device.GetMemoryInfo();
        Assert.Equal(0, afterA.UsedByUs);   // everything went back to the device
        using (var mb = LlmModel.Load(PathB, Options(device)))
        {
            estB = mb.EstimateFootprint(2, 1024).Total;
            using var db = mb.CreateBatchDecoder(2, 1024);
            actB = db.DeviceMemoryBytes;
            Console.Error.WriteLine($"[B alone] estimated {estB / MiB} MiB, compiled {actB / MiB} MiB, device used by us {device.GetMemoryInfo().UsedByUs / MiB} MiB");
            soloB = Greedy(db, mb, PromptB, N);
        }
        Assert.Equal(0, device.GetMemoryInfo().UsedByUs);
        Assert.NotEqual(soloA, soloB);

        // ---- both on the same device, generating at the same time ----
        using var modelA = LlmModel.Load(PathA, Options(device));
        using var modelB = LlmModel.Load(PathB, Options(device));
        using var decA = modelA.CreateBatchDecoder(2, 1024);
        using var decB = modelB.CreateBatchDecoder(2, 1024);
        var both = device.GetMemoryInfo();
        Console.Error.WriteLine($"[A+B] device used by us {both.UsedByUs / MiB} MiB (A {actA / MiB} + B {actB / MiB}), driver usage {both.Usage / MiB} MiB, available {both.Available / MiB} MiB");
        Assert.InRange(both.UsedByUs, (actA + actB) * 95 / 100, (actA + actB) * 125 / 100 + 600 * MiB);

        // Sharing the device must not change results even without concurrency.
        Assert.Equal(soloA, Greedy(decA, modelA, PromptA, N));
        Assert.Equal(soloB, Greedy(decB, modelB, PromptB, N));
        Assert.Equal(soloA, Greedy(decA, modelA, PromptA, N));

        for (int round = 0; round < 3; round++)
        {
            int[]? a = null, b = null;
            Exception? ea = null;
            Exception? eb = null;
            using var go = new Barrier(2);
            var ta = new Thread(() => { try { go.SignalAndWait(); a = Greedy(decA, modelA, PromptA, N); } catch (Exception e) { ea = e; } });
            var tb = new Thread(() => { try { go.SignalAndWait(); b = Greedy(decB, modelB, PromptB, N); } catch (Exception e) { eb = e; } });
            ta.Start();
            tb.Start();
            ta.Join();
            tb.Join();
            Assert.Null(ea);
            Assert.Null(eb);
            Assert.Equal(soloA, a);
            Assert.Equal(soloB, b);
        }

        // The second slot of a decoder is as good as the first (a decoder itself is driven by one thread; batching several
        // requests of one model across slots is the engine's job and is covered by the host test below).
        Assert.Equal(soloA, Greedy(decA, modelA, PromptA, N, slot: 1));
    }

    /// <summary>Stress: four compiled models (two per file) on one device, each driven by its own thread at full speed.</summary>
    [Fact]
    public void FourModelInstances_OneDevice_FourThreads_Stress()
    {
        using var pool = Setup(out var device)!;
        const int N = 64;
        const string PromptA = "The capital of France is";
        const string PromptB = "Once upon a time, in a small village";
        using var ma = LlmModel.Load(PathA, Options(device));
        using var mb = LlmModel.Load(PathB, Options(device));
        var decs = new List<(IBatchDecoder Dec, LlmModel Model, string Prompt)>();
        try
        {
            decs.Add((ma.CreateBatchDecoder(1, 1024), ma, PromptA));
            decs.Add((mb.CreateBatchDecoder(2, 1024), mb, PromptB));
            decs.Add((ma.CreateBatchDecoder(2, 1024), ma, PromptA));
            decs.Add((mb.CreateBatchDecoder(1, 1024), mb, PromptB));
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var solo = decs.Select(d => Greedy(d.Dec, d.Model, d.Prompt, N)).ToArray();
            double seqMs = sw.Elapsed.TotalMilliseconds;
            for (int round = 0; round < 4; round++)
            {
                sw.Restart();
                var res = new int[decs.Count][];
                var errs = new Exception?[decs.Count];
                using var go = new Barrier(decs.Count);
                var threads = decs.Select((d, i) => new Thread(() =>
                {
                    try { go.SignalAndWait(); res[i] = Greedy(d.Dec, d.Model, d.Prompt, N); } catch (Exception e) { errs[i] = e; }
                })).ToList();
                threads.ForEach(t => t.Start());
                threads.ForEach(t => t.Join());
                Console.Error.WriteLine($"[stress] round {round}: 4 instances sequentially {seqMs:F0} ms, concurrently {sw.Elapsed.TotalMilliseconds:F0} ms");
                Assert.All(errs, Assert.Null);
                for (int i = 0; i < decs.Count; i++)
                {
                    Assert.Equal(solo[i], res[i]);
                }
            }
        }
        finally
        {
            foreach (var d in decs)
            {
                d.Dec.Dispose();
            }
        }
        Assert.Equal(0, device.GetMemoryInfo().UsedByUs);
    }

    [Fact]
    public async Task Host_TwoChatModels_Concurrent_Eviction_And_MemoryReturns()
    {
        using var pool = Setup(out var device)!;
        var baseline = device.GetMemoryInfo();
        var chat = new ChatModelOptions { Slots = 2, Service = new ModelServiceOptions { DefaultMaxTokens = 40 } };
        var host = new ModelHost(new ModelHostOptions { ReserveBytes = 1L << 30, Log = Console.Error.WriteLine });
        host.Register(new ModelRegistration { Name = "qwen", Loader = new LlmChatModelLoader("qwen", PathA, Options(device), chat) });
        host.Register(new ModelRegistration { Name = "smol", Loader = new LlmChatModelLoader("smol", PathB, Options(device), chat) });
        try
        {
            async Task<string> Ask(string model, string q)
            {
                using var lease = await host.AcquireAsync(model);
                var svc = ((IChatModel)lease.Model).Service;
                var r = await svc.StartChat(new ChatRequest
                {
                    Messages = [new ChatTurn("user", q)], Sampling = SamplingParams.GreedyDefault, MaxTokens = 40, EnableThinking = false,
                }).CompleteAsync();
                return r.Choices[0].Content;
            }

            var soloQ = await Ask("qwen", "Name a color.");
            await host.UnloadAsync("qwen");
            Assert.Equal(0, device.GetMemoryInfo().UsedByUs);
            var soloS = await Ask("smol", "Name an animal.");
            Assert.Equal(ModelState.Loaded, host.Get("smol")!.State);
            Assert.NotEmpty(soloQ);
            Assert.NotEmpty(soloS);

            // Both resident, several concurrent requests to each from different threads.
            _ = await Ask("qwen", "Name a color.");
            var infos = host.List();
            foreach (var i in infos)
            {
                Console.Error.WriteLine($"[host] {i.Name}: {i.State}, estimated {i.EstimatedBytes / MiB} MiB, resident {i.ResidentBytes / MiB} MiB");
            }

            Console.Error.WriteLine($"[host] device: {device.GetMemoryInfo()}");
            var results = await Task.WhenAll(Enumerable.Range(0, 8).Select(i => Task.Run(() => i % 2 == 0 ? Ask("qwen", "Name a color.") : Ask("smol", "Name an animal."))));
            for (int i = 0; i < results.Length; i++)
            {
                Assert.Equal(i % 2 == 0 ? soloQ : soloS, results[i]);
            }

            // Memory comes back and the estimate was in the right range (an over-estimate is fine, an under-estimate is not).
            long used = device.GetMemoryInfo().UsedByUs;
            Assert.True(await host.UnloadAsync("qwen"));
            Assert.True(await host.UnloadAsync("smol"));
            Assert.Equal(0, device.GetMemoryInfo().UsedByUs);
            Assert.True(used > 0);
            foreach (var i in infos)
            {
                Assert.True(i.EstimatedBytes >= i.ResidentBytes * 9 / 10, $"{i.Name}: estimate {i.EstimatedBytes / MiB} MiB < resident {i.ResidentBytes / MiB} MiB");
            }

            // LRU eviction with a model cap of one: asking for the other model replaces the first.
            host.Dispose();
            host = new ModelHost(new ModelHostOptions { ReserveBytes = 1L << 30, MaxLoadedModels = 1, Log = Console.Error.WriteLine });
            host.Register(new ModelRegistration { Name = "qwen", Loader = new LlmChatModelLoader("qwen", PathA, Options(device), chat) });
            host.Register(new ModelRegistration { Name = "smol", Loader = new LlmChatModelLoader("smol", PathB, Options(device), chat) });
            Assert.Equal(soloQ, await Ask("qwen", "Name a color."));
            Assert.Equal(soloS, await Ask("smol", "Name an animal."));
            Assert.Equal(ModelState.Unloaded, host.Get("qwen")!.State);
            Assert.Equal(ModelState.Loaded, host.Get("smol")!.State);
        }
        finally { host.Dispose(); }
        Assert.Equal(0, device.GetMemoryInfo().UsedByUs);
        Console.Error.WriteLine($"[host] after dispose: {device.GetMemoryInfo()} (baseline {baseline})");
    }

    [Fact]
    public async Task Host_RefusesAModel_ThatDoesNotFit()
    {
        // The device is opened with a 300 MiB budget (the same mechanism as --vram-budget): the ~1.1 GiB model cannot fit.
        using var pool = Setup(out var device, new DeviceOptions { MemoryBudget = 300 * MiB })!;
        using var host = new ModelHost(new ModelHostOptions { ReserveBytes = 0 });
        host.Register(new ModelRegistration { Name = "qwen", Loader = new LlmChatModelLoader("qwen", PathA, Options(device), new ChatModelOptions { Slots = 2 }) });
        var ex = await Assert.ThrowsAsync<ModelDoesNotFitException>(() => host.AcquireAsync("qwen"));
        Console.Error.WriteLine("[refusal] " + ex.Message);
        Assert.Contains("300 MiB", ex.Message);
        Assert.Equal(0, device.GetMemoryInfo().UsedByUs);
        Assert.Equal(ModelState.Failed, host.Get("qwen")!.State);
    }

    [Fact]
    public async Task Host_OffloadsLayersToTheCpu_WhenTheGpuBudgetIsTooSmall()
    {
        using var pool = Setup(out var device, new DeviceOptions { MemoryBudget = 900 * MiB })!;
        var cpuProv = new CpuExecutionProvider();
        using var cpu = cpuProv.OpenDevice(cpuProv.GetDevices()[0]);
        var opts = new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], Fallback = new DeviceSlot { Device = cpu }, ContextLength = 1024, MaxBatchTokens = 128, Log = Console.Error.WriteLine };
        using var host = new ModelHost(new ModelHostOptions { ReserveBytes = 0, Log = Console.Error.WriteLine });
        host.Register(new ModelRegistration { Name = "qwen", Loader = new LlmChatModelLoader("qwen", PathA, opts, new ChatModelOptions { Slots = 2 }) });
        using var lease = await host.AcquireAsync("qwen");
        var svc = ((IChatModel)lease.Model).Service;
        var r = await svc.StartChat(new ChatRequest { Messages = [new ChatTurn("user", "Name a color.")], Sampling = SamplingParams.GreedyDefault, MaxTokens = 20, EnableThinking = false }).CompleteAsync();
        Assert.NotEmpty(r.Choices[0].Content);
        long onGpu = device.GetMemoryInfo().UsedByUs;
        Console.Error.WriteLine($"[offload] on GPU {onGpu / MiB} MiB of a 900 MiB budget (resident reported {lease.Model.ResidentBytes / MiB} MiB); reply: {r.Choices[0].Content}");
        Assert.InRange(onGpu, 1, 900 * MiB);   // part of the model is on the GPU (within the budget), the rest on the CPU
    }
}
