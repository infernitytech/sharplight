using Sharplight.Cpu;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// The families of bring-up round 11 (<see cref="HubSweepFamilyCases"/>) and of <see cref="LlamaCppAdditionsFamilyCases"/> behind the continuous-batching engine: greedy output of prompts of different
/// lengths decoded together, prefilled in small chunks, is token-identical to decoding each prompt alone.
/// </summary>
public class HubSweepFamilyServingTests
{
    private static readonly string Models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models");

    /// <summary>Model paths under ~/models, each on the CPU and on Vulkan, with whether the KV cache is kept in f32.</summary>
    public static TheoryData<string, bool, bool> Cases()
    {
        // The tiny PaliGemma models scale their final norm far above one, so f16 KV rounding decides near-ties; their caches are kept in f32
        // as in the generation cases.
        // tiny ZAYA1's router has near-ties that f16 KV rounding flips on the CPU (see LlamaCppAdditionsFamilyCases).
        string[] f32Kv = ["hf/tiny-random-paligemma", "hf/tiny-random-paligemma2", "hf/tiny-random-zaya1", "gguf/tiny-random-zaya1-skip-f32.gguf"];
        var data = new TheoryData<string, bool, bool>();
        string[] models =
        [
            "hf/tiny-random-lfm3",
            "hf/tiny-random-spark2_5",
            "hf/tiny-random-clover-moe",
            "hf/tiny-random-nemotron-h-omni-puzzle",
            "hf/tiny-random-paligemma",
            "hf/tiny-random-paligemma2",
            "hf/tiny-random-mimo-v2",
            "hf/tiny-random-llava-onevision2",
            // Architectures llama.cpp added after 8ca5a80 (LlamaCppAdditionsFamilyCases).
            "hf/tiny-random-domynedge",
            "hf/tiny-random-zaya1",
            // The ZAYA1 variant whose skip expert wins for part of the tokens (llama.cpp's converter keeps its class bias of 0).
            "gguf/tiny-random-zaya1-skip-f32.gguf",
        ];
        foreach (string model in models)
        {
            data.Add(model, false, f32Kv.Contains(model));
            data.Add(model, true, f32Kv.Contains(model));
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task Batched_GreedyOutput_Matches_SingleSequence(string name, bool vulkan, bool f32Kv)
    {
        string path = Path.Combine(Models, name);
        Assert.SkipUnless(Directory.Exists(path) || File.Exists(path), $"{path} not present");
        using var device = OpenDevice(vulkan);
        using var model = LlmModel.Load(TestHost.Loader, path, new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }], ContextLength = 128, MaxBatchTokens = 16,
            KvCacheType = f32Kv ? Sharplight.Tensors.DataType.F32 : null,
        });
        var rng = new Random(3);
        int[][] prompts = [.. new[] { 5, 13, 22, 37 }.Select(n => Enumerable.Range(0, n).Select(_ => rng.Next(10, 1000)).ToArray())];
        var ct = TestContext.Current.CancellationToken;

        var expected = new int[prompts.Length][];
        for (int i = 0; i < prompts.Length; i++)
        {
            expected[i] = (await Generate(model, model.CreateBatchDecoder(slots: 1), [prompts[i]], ct))[0];
        }

        var actual = await Generate(model, model.CreateBatchDecoder(slots: prompts.Length), prompts, ct);
        for (int i = 0; i < prompts.Length; i++)
        {
            Assert.Equal(expected[i], actual[i]);
        }
    }

    private static async Task<int[][]> Generate(LlmModel model, IBatchDecoder decoder, int[][] prompts, CancellationToken ct)
    {
        using var engine = new InferenceEngine(decoder, model.Tokenizer, new EngineOptions());
        var handles = prompts.Select(p => engine.Submit(new GenerationRequest
        {
            PromptTokens = p, MaxTokens = 24, StopAtEog = false, Logprobs = true, Sampling = SamplingParams.GreedyDefault,
        }, ct)).ToList();
        var results = await Task.WhenAll(handles.Select(h => h.CompleteAsync(ct)));
        foreach (var r in results)
        {
            Assert.Null(r.Error);
        }

        return [.. results.Select(r => r.Logprobs.Select(l => l.TokenId).ToArray())];
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
