using Sharplight.Cpu;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// Prefix caching on real model graphs (paged decoders, the default for several slots): prompts sharing a 40-token prefix decode
/// token-identically with the cache on and off, and later prompts reuse cached blocks: attention-only models attach them directly,
/// models with recurrent state (gated delta nets, Mamba, Mamba-2, short convolutions, RWKV-7) together with a snapshot of the
/// recurrent state taken at a block boundary.
/// </summary>
public class PrefixCacheParityTests
{
    private static readonly string Models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models");

    /// <summary>Model paths under ~/models and whether they have recurrent state, each on the CPU and on Vulkan.</summary>
    public static TheoryData<string, bool, bool> Cases()
    {
        var data = new TheoryData<string, bool, bool>();
        (string Path, bool Recurrent)[] models =
        [
            ("gguf/tiny-random-qwen1-f32.gguf", false),
            ("hf/tiny-random-granite", false),
            ("gguf/tiny-random-qwen3.8-flash-next-nomtp-f32.gguf", true),   // gated delta net + gated attention
            ("hf/tiny-random-MambaForCausalLM", true),
            ("hf/tiny-random-falcon-h1", true),                              // Mamba-2 + attention
            ("gguf/tiny-random-lfm3-f32.gguf", true),                        // short convolutions + attention
            ("gguf/tiny-random-arwkv7-f32.gguf", true),
        ];
        foreach (var (path, recurrent) in models)
        {
            data.Add(path, recurrent, false);
            data.Add(path, recurrent, true);
        }

        return data;
    }

    /// <summary>Real checkpoints (Vulkan only): Qwen3.5-0.8B (gated delta net hybrid) and Qwen3-0.6B.</summary>
    public static TheoryData<string, bool> RealCases() => new()
    {
        { "hf/Qwen3.5-0.8B", true },
        { "gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf", false },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public Task SharedPrefixes_DecodeIdentically_WithAndWithoutTheCache(string name, bool recurrent, bool vulkan) => Run(name, recurrent, vulkan, 16);

    [Theory]
    [MemberData(nameof(RealCases))]
    public Task RealModels_DecodeIdentically_WithAndWithoutTheCache(string name, bool recurrent) => Run(name, recurrent, true, 512);

    private static async Task Run(string name, bool recurrent, bool vulkan, int maxBatch)
    {
        string path = Path.Combine(Models, name);
        Assert.SkipUnless(Directory.Exists(path) || File.Exists(path), $"{path} not present");
        using var device = OpenDevice(vulkan);
        using var model = LlmModel.Load(TestHost.Loader, path, new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 256, MaxBatchTokens = maxBatch });
        var rng = new Random(5);
        int vocab = Math.Min(1000, model.Tokenizer.VocabSize);
        int[] prefix = [.. Enumerable.Range(0, 40).Select(_ => rng.Next(10, vocab))];
        int[][] prompts = [.. new[] { 5, 9, 12, 7, 11, 6 }.Select(n => prefix.Concat(Enumerable.Range(0, n).Select(_ => rng.Next(10, vocab))).ToArray())];
        var ct = TestContext.Current.CancellationToken;

        int[][] expected;
        using (var engine = new InferenceEngine(model.CreateBatchDecoder(slots: 2), model.Tokenizer, new EngineOptions { PrefixCache = false }))
        {
            expected = new int[prompts.Length][];
            for (int i = 0; i < prompts.Length; i++)
            {
                expected[i] = (await Generate(engine, prompts[i], ct)).Tokens;
            }
        }

        var decoder = model.CreateBatchDecoder(slots: 2);
        Assert.True(decoder.KvBlockSize > 0, "the decoder is not paged by default");
        Assert.True(decoder.CachesKvBlocks, "the decoder does not cache blocks");
        using var cached = new InferenceEngine(decoder, model.Tokenizer);
        var hits = new int[prompts.Length];
        // One after another: the first leaves blocks, the second (recurrent: a state snapshot at the shared prefix), the rest reuse them.
        for (int i = 0; i < 4; i++)
        {
            var (tokens, hit) = await Generate(cached, prompts[i], ct);
            Assert.Equal(expected[i], tokens);
            hits[i] = hit;
        }

        // Two at once, both finding the shared prefix cached.
        var both = await Task.WhenAll(Generate(cached, prompts[4], ct), Generate(cached, prompts[5], ct));
        Assert.Equal(expected[4], both[0].Tokens);
        Assert.Equal(expected[5], both[1].Tokens);
        Assert.Equal(0, hits[0]);
        Assert.True(hits[recurrent ? 2 : 1] >= 32, $"cached tokens per prompt: {string.Join(", ", hits)}");
        Assert.True(both[0].Cached >= 32 && both[1].Cached >= 32, $"cached tokens: {both[0].Cached}, {both[1].Cached}");
        if (recurrent)
        {
            Assert.True(cached.GetMetrics().StateSnapshots >= 1);
        }
    }

    private static async Task<(int[] Tokens, int Cached)> Generate(InferenceEngine engine, int[] prompt, CancellationToken ct)
    {
        var r = await engine.Submit(new GenerationRequest
        {
            PromptTokens = prompt, MaxTokens = 16, StopAtEog = false, Logprobs = true, Sampling = SamplingParams.GreedyDefault,
        }, ct).CompleteAsync(ct);
        Assert.Null(r.Error);
        return ([.. r.Logprobs.Select(l => l.TokenId)], r.Usage.CachedTokens);
    }

    private static IComputeDevice OpenDevice(bool vulkan)
    {
        if (!vulkan)
        {
            var cpu = new CpuExecutionProvider();
            return cpu.OpenDevice(cpu.GetDevices()[0]);
        }

        var p = new VulkanProvider();
        Assert.SkipUnless(p.CheckAvailability().IsAvailable, "Vulkan is not available");
        var devs = p.GetDevices();
        var dev = devs.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devs.FirstOrDefault();
        Assert.SkipWhen(dev is null, "no Vulkan device");
        return p.OpenDevice(dev!);
    }
}
