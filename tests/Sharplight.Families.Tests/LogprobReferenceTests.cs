using Sharplight.Cpu;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// Logits-level checks for models whose greedy text is too fragile to pin (near-tied top tokens): the log-probabilities of the reference's top tokens,
/// captured from llama.cpp b11177 (CPU, weights re-quantized to F16 from the checked-in Q8_0 file so no activation quantization noise), must be matched within
/// the quantization noise of the Q8_0 file the tests load. Truncated real checkpoints compare with transformers in float32 on the file's own
/// dequantized weights instead (no quantization noise in the reference, so a tighter tolerance).
/// </summary>
public class LogprobReferenceTests
{
    private const double DefaultTolerance = 0.25;
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    /// <summary>A prompt and the reference log-probability of some next-token candidates.</summary>
    public sealed record Case(string Name, string Path, string Prompt, (int Id, double LogProb)[] Expected, double Tolerance = DefaultTolerance);

    private static readonly Case[] Cases =
    [
        // RWKV-6 World 1.6B: llama.cpp F16 (the Q8_0 model flips the near-tied " the" / " France" continuation).
        new("rwkv6 'The'", "gguf/rwkv-6-world-1b6/rwkv-6-world-1.6b-Q8_0.gguf", "The", [(267, -3.768), (38499, -4.545), (31461, -5.077), (265, -5.107)]),
        new("rwkv6 'The capital of France is'", "gguf/rwkv-6-world-1b6/rwkv-6-world-1.6b-Q8_0.gguf", "The capital of France is", [(37138, -0.773), (22590, -2.585), (4596, -2.869), (22187, -3.169)]),
        new("rwkv6 'The capital of France is Paris.\\nThe capital of'", "gguf/rwkv-6-world-1b6/rwkv-6-world-1.6b-Q8_0.gguf", "The capital of France is Paris.\nThe capital of",
            [(44312, -2.049), (22590, -2.134), (50275, -2.527), (36847, -2.748)]),
        // DeepSeek-V3.1 (unsloth Q4_K_M) cut to its first 6 layers (3 dense, 3 MoE with 256 real experts; tools/truncate-gguf.py), the
        // prompt at which the full model's greedy text and llama.cpp's part (a near tie of " Spain" / " France"). Reference: transformers'
        // DeepseekV3 in float32 on the file's dequantized weights (tools/latent-moe-gguf-reference.py ... 8 4: the model's group-limited
        // routing, 8 groups with the best 4 kept, which the importer defaults to for DeepSeek-V3 files without expert_group_count). Ours is
        // within 0.03 on CPU; without the groups (as llama.cpp b11177 routes this file) token 50365 is 0.19 off and the argmax differs.
        new("deepseek-v3.1 6 layers 'The capital of France is Paris. The capital of'", "gguf/DeepSeek-V3.1-trunc/DeepSeek-V3.1-Q4_K_M-L6.gguf",
            "The capital of France is Paris. The capital of", [(81661, -3.608), (50365, -3.633), (104323, -4.217), (119996, -4.387), (74218, -4.778)],
            Tolerance: 0.1),
        // GLM-5.2 (unsloth UD-Q2_K_XL) cut to its first 6 layers (3 dense, 3 MoE, DSA indexer in its dense regime), at the prompt where the
        // full model's greedy text and llama.cpp's part. Reference: transformers' GlmMoeDsa in float32 on the file's dequantized weights
        // (tools/latent-moe-gguf-reference.py). Ours is within 0.04 on CPU; llama.cpp b11177 within 0.06.
        new("glm-5.2 6 layers 'The capital of France is Paris. Distance ... while direct flight time'", "gguf/GLM-5.2-trunc/GLM-5.2-UD-Q2_K_XL-L6.gguf",
            "The capital of France is Paris. Distance from Paris to Marseille is 661 km, while direct flight time",
            [(21158, -6.023), (79386, -6.209), (52628, -6.407), (7588, -6.598), (4740, -6.609)], Tolerance: 0.1),
        // Kimi-K3 (unsloth UD-Q1_0) cut to its first 4 layers (3 Kimi delta-attention layers, one latent-attention layer, real latent MoE),
        // its narrow IQ1 experts widened to IQ1_S (tools/truncate-gguf-remote.py --widen-narrow-iq1) so that llama.cpp b11177 reads it.
        // Reference: llama.cpp b11177 with the prompt decoded token by token (CPU; its batched prefill is within 0.04 of that).
        new("kimi-k3 4 layers 'The capital of France is Paris. The capital of Germany is'", "gguf/Kimi-K3-trunc/Kimi-K3-UD-Q1_0-L4-iq1s.gguf",
            "The capital of France is Paris. The capital of Germany is",
            [(36356, -5.422), (140580, -5.526), (24917, -5.601), (127274, -5.689), (19320, -5.871)], Tolerance: 0.12),
    ];

    public static IEnumerable<TheoryDataRow<string>> CaseNames() => Cases.Select(c => new TheoryDataRow<string>(c.Name));

    private static void Run(string name, bool vulkan)
    {
        var c = Cases.Single(x => x.Name == name);
        string path = System.IO.Path.Combine(Home, "models", c.Path);
        Assert.SkipUnless(File.Exists(path), $"{path} not present");
        IExecutionProvider provider = vulkan ? new VulkanProvider() : new CpuExecutionProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "provider not available");
        var devices = provider.GetDevices();
        var info = vulkan ? devices.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devices[0] : devices[0];
        var slot = new DeviceSlot { Device = provider.OpenDevice(info) };
        // The test owns the device (models do not): disposed after the model, so no device outlives the test.
        using var device = slot.Device;
        using var model = LlmModel.Load(TestHost.Loader, path, new LlmLoadOptions { Devices = [slot], ContextLength = 256, MaxBatchTokens = 64 });
        using var session = model.CreateSession(WellKnown.DecodeGraph);
        var logits = session.Process(model.Tokenizer.Encode(c.Prompt, addSpecial: true, parseSpecial: false));
        double max = logits.Max();
        double logSum = Math.Log(logits.Sum(l => Math.Exp(l - max)));
        foreach (var (id, expected) in c.Expected)
        {
            double actual = logits[id] - max - logSum;
            Assert.True(Math.Abs(actual - expected) < c.Tolerance, $"{c.Name}: token {id} log-probability {actual:F3}, reference {expected:F3}");
        }
    }

    [Theory, MemberData(nameof(CaseNames))]
    public void Logprobs_match_reference_on_cpu(string name) => Run(name, vulkan: false);

    [Theory, MemberData(nameof(CaseNames))]
    public void Logprobs_match_reference_on_vulkan(string name) => Run(name, vulkan: true);
}
