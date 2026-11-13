using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// Logits-level checks for models whose greedy text is too fragile to pin (near-tied top tokens): the log-probabilities of the reference's top tokens,
/// captured from llama.cpp b11177 (CPU, weights re-quantized to F16 from the checked-in Q8_0 file so no activation quantization noise), must be matched within
/// the quantization noise of the Q8_0 file the tests load.
/// </summary>
public class LogprobReferenceTests
{
    private const double Tolerance = 0.25;
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    static LogprobReferenceTests()
    {
        GgufImporter.Register();
    }

    /// <summary>A prompt and the reference log-probability of some next-token candidates.</summary>
    public sealed record Case(string Name, string Path, string Prompt, (int Id, double LogProb)[] Expected);

    private static readonly Case[] Cases =
    [
        // RWKV-6 World 1.6B: llama.cpp F16 (the Q8_0 model flips the near-tied " the" / " France" continuation).
        new("rwkv6 'The'", "gguf/rwkv-6-world-1b6/rwkv-6-world-1.6b-Q8_0.gguf", "The", [(267, -3.768), (38499, -4.545), (31461, -5.077), (265, -5.107)]),
        new("rwkv6 'The capital of France is'", "gguf/rwkv-6-world-1b6/rwkv-6-world-1.6b-Q8_0.gguf", "The capital of France is", [(37138, -0.773), (22590, -2.585), (4596, -2.869), (22187, -3.169)]),
        new("rwkv6 'The capital of France is Paris.\\nThe capital of'", "gguf/rwkv-6-world-1b6/rwkv-6-world-1.6b-Q8_0.gguf", "The capital of France is Paris.\nThe capital of",
            [(44312, -2.049), (22590, -2.134), (50275, -2.527), (36847, -2.748)]),
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
        using var model = LlmModel.Load(path, new LlmLoadOptions { Devices = [slot], ContextLength = 256, MaxBatchTokens = 64 });
        using var session = model.CreateSession(WellKnown.DecodeGraph);
        var logits = session.Process(model.Tokenizer.Encode(c.Prompt, addSpecial: true, parseSpecial: false));
        double max = logits.Max();
        double logSum = Math.Log(logits.Sum(l => Math.Exp(l - max)));
        foreach (var (id, expected) in c.Expected)
        {
            double actual = logits[id] - max - logSum;
            Assert.True(Math.Abs(actual - expected) < Tolerance, $"{c.Name}: token {id} log-probability {actual:F3}, reference {expected:F3}");
        }
    }

    [Theory, MemberData(nameof(CaseNames))]
    public void Logprobs_match_reference_on_cpu(string name) => Run(name, vulkan: false);

    [Theory, MemberData(nameof(CaseNames))]
    public void Logprobs_match_reference_on_vulkan(string name) => Run(name, vulkan: true);
}
