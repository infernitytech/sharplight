using Sharplight.Cpu;
using Sharplight.Import.Gguf;
using Sharplight.Import.HuggingFace;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// Inkling (<see cref="InklingFamilyCases"/>) behind the continuous-batching engine: greedy output of prompts
/// of different lengths decoded together, prefilled in small chunks, is token-identical to decoding each prompt alone (the prompts run past
/// the sliding window, the relative-bias extent and the temperature floor, and the short-convolution states follow each sequence).
/// </summary>
public class InklingFamilyServingTests
{
    private static readonly string Models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models");

    /// <summary>Model paths under ~/models, each on the CPU and on Vulkan.</summary>
    public static TheoryData<string, bool> Cases()
    {
        var data = new TheoryData<string, bool>();
        string[] models =
        [
            "hf/tiny-random-inkling",
            "gguf/tiny-random-inkling-f32.gguf",
        ];
        foreach (string model in models)
        {
            data.Add(model, false);
            data.Add(model, true);
        }

        return data;
    }

    /// <summary>The header-only checks (config / GGUF header) accept the tiny checkpoints that the importers load, and report the full vocabulary as their shape.</summary>
    [Fact]
    public void HeaderOnlyCheck_AgreesWithTheImport()
    {
        string hf = Path.Combine(Models, "hf/tiny-random-inkling");
        string gguf = Path.Combine(Models, "gguf/tiny-random-inkling-f32.gguf");
        Assert.SkipUnless(Directory.Exists(hf) && File.Exists(gguf), "tiny Inkling checkpoints not present");

        string config = File.ReadAllText(Path.Combine(hf, "config.json"));
        Assert.Equal(SupportLevel.Supported, HfSupport.GetArchitectureSupport(HfSupport.GetModelType(config)));
        Assert.Empty(HfSupport.Validate(config));

        var header = File.ReadAllBytes(gguf);
        var check = GgufSupport.Validate(GgufHeaderProbe.ReadMetadata(header), GgufHeaderProbe.ReadTensorTable(header)!);
        Assert.True(check.IsAccepted, string.Join(" ", check.Problems));

        foreach (string path in new[] { hf, gguf })
        {
            using var device = OpenDevice(vulkan: false);
            using var model = LlmModel.Load(TestHost.Loader, path, new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 64 });
            Assert.Equal(1024, model.Tokenizer.VocabSize);
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task Batched_GreedyOutput_Matches_SingleSequence(string name, bool vulkan)
    {
        string path = Path.Combine(Models, name);
        Assert.SkipUnless(Directory.Exists(path) || File.Exists(path), $"{path} not present");
        using var device = OpenDevice(vulkan);
        using var model = LlmModel.Load(TestHost.Loader, path, new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 128, MaxBatchTokens = 16 });
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
