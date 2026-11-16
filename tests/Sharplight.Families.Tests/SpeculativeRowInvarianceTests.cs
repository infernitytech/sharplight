using Sharplight.Cpu;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// A speculative verification step (pending token plus drafts, one input of several rows) must give every row the logits of
/// plain decoding, bit for bit, so greedy output with drafts equals greedy output without (the decoder is compiled row-invariant up
/// to the verification rows). Each model runs a prompt, one 4-row speculative step, a rollback, and the same 4 tokens as 4
/// single-row steps; the logits are compared bitwise. The model list goes from plain attention to the hybrid features one by one
/// (gated delta rule, MoE, hyper-connections with n-gram embedding and QSA indexer). Tiny Qwen3.8-Flash-Next on the CPU is left out:
/// it still differs in the last bits (open item in docs/model-support.md).
/// </summary>
public sealed class SpeculativeRowInvarianceTests
{
    private const int Rows = 4;

    [Theory]
    [InlineData("tiny-random-llama-eagle3-target", false)]
    [InlineData("tiny-random-llama-eagle3-target", true)]
    [InlineData("tiny-random-qwen3.5", false)]
    [InlineData("tiny-random-qwen3.5", true)]
    [InlineData("tiny-random-qwen3.8-moe", false)]
    [InlineData("tiny-random-qwen3.8-moe", true)]
    [InlineData("tiny-random-qwen3-next-moe", true)]
    [InlineData("tiny-random-qwen3.8-flash-next", true)]
    public void VerificationRows_EqualSingleRowSteps(string name, bool vulkan)
    {
        string path = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "hf", name);
        Assert.SkipUnless(Directory.Exists(path), $"{path} not present");
        var cpu = new CpuExecutionProvider();
        using var cpuDevice = cpu.OpenDevice(cpu.GetDevices()[0]);
        using var gpu = vulkan ? OpenVulkan() : null;
        using var model = LlmModel.Load(TestHost.Loader, path, new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = gpu ?? cpuDevice }],
            Fallback = gpu is null ? null : new DeviceSlot { Device = cpuDevice },
            ContextLength = 256,
            MaxBatchTokens = 64,
            DeviceSampling = false,
        });
        var decoder = model.CreateBatchDecoder(1, maxLogitsRows: Rows, draftTokens: Rows - 1);
        int[] prompt = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333];
        int[] tokens = [1234, 2345, 3456, 4567];
        decoder.Step([new SequenceInput(0, prompt)]);

        var batched = decoder.Step([new SequenceInput(0, tokens, LogitsRows: Rows, Speculative: true)])[0].ToArray();
        decoder.Truncate(0, prompt.Length);
        int vocab = decoder.VocabSize;
        for (int r = 0; r < Rows; r++)
        {
            var single = decoder.Step([new SequenceInput(0, tokens.AsMemory(r, 1))])[0].Span;
            for (int i = 0; i < vocab; i++)
            {
                if (BitConverter.SingleToInt32Bits(single[i]) != BitConverter.SingleToInt32Bits(batched[r * vocab + i]))
                {
                    Assert.Fail($"row {r}, logit {i}: {batched[r * vocab + i]:R} in the {Rows}-row step vs {single[i]:R} in a single-row step");
                }
            }
        }
    }

    private static IComputeDevice OpenVulkan()
    {
        var provider = new VulkanProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "Vulkan is not available");
        var device = provider.GetDevices().FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? provider.GetDevices().FirstOrDefault();
        Assert.SkipWhen(device is null, "no Vulkan device");
        return provider.OpenDevice(device!);
    }
}
