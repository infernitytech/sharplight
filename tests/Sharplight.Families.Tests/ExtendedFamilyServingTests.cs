using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// The families of bring-up rounds 7 and 8 (<see cref="ExtendedFamilyCases"/>) behind the continuous-batching engine: greedy output of prompts of
/// different lengths decoded together, prefilled in small chunks, is token-identical to decoding each prompt alone.
/// </summary>
public class ExtendedFamilyServingTests
{
    private static readonly string Models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models");

    static ExtendedFamilyServingTests()
    {
        HuggingFaceImporter.Register();
        GgufImporter.Register();
    }

    /// <summary>Model paths under ~/models, each on the CPU and on Vulkan.</summary>
    public static TheoryData<string, bool> Cases()
    {
        var data = new TheoryData<string, bool>();
        string[] models =
        [
            // hf/tiny-random-chameleon is left out: one of these prompts has a top-2 margin of 0.001 at its first token, which the
            // batched kernels' accumulation order flips on Vulkan. The variants exercise the same per-head layer norms.
            "hf/tiny-random-chameleon-mha",
            "hf/tiny-random-chameleon-swin",
            "gguf/tiny-random-chameleon-swin-f32.gguf",
            "hf/tiny-random-minimax-m3",
            "gguf/tiny-random-minimax-m3-f32.gguf",
            "hf/tiny-random-glm5-next",
            "gguf/tiny-random-glm5-next-f32.gguf",
            // Round 8: the indexers in their sparse regime (budgets of 12 and 8 positions), per-head block selection and pooled selection.
            "hf/tiny-random-minimax-m3-sparse",
            "gguf/tiny-random-minimax-m3-sparse-f32.gguf",
            "hf/tiny-random-glm5-next-sparse",
            "gguf/tiny-random-glm5-next-sparse-f32.gguf",
            "gguf/tiny-random-mistral-large4-f32.gguf",
            "gguf/tiny-random-dots3note-f32.gguf",
        ];
        foreach (string model in models)
        {
            data.Add(model, false);
            data.Add(model, true);
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task Batched_GreedyOutput_Matches_SingleSequence(string name, bool vulkan)
    {
        string path = Path.Combine(Models, name);
        Assert.SkipUnless(Directory.Exists(path) || File.Exists(path), $"{path} not present");
        using var device = OpenDevice(vulkan);
        using var model = LlmModel.Load(path, new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 128, MaxBatchTokens = 16 });
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
