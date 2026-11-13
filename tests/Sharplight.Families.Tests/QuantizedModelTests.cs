using System.Text.Json;
using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// End-to-end checks of the weight formats added by the quantization track (Q2_K, Q3_K, IQ1/IQ2/IQ3, TQ1_0/TQ2_0, MXFP4, NVFP4): a small
/// model requantized to one format (tools/make-quant-test-models.py builds ~/models/gguf/quant-test/qwen3-0.6b-*.gguf) is run on the CPU and
/// on Vulkan, and the next-token distribution after a fixed prompt is compared with the one llama.cpp b11177 produced from the same file
/// (Fixtures/quant_next_token.json). The files are tiny and heavily quantized, so near-tied logits flip easily: the comparison is on
/// probabilities, not on greedy text. Skips when the model files are absent.
/// </summary>
public class QuantizedModelTests
{
    private const double MaxProbabilityDifference = 0.3;
    private const double MaxCpuVulkanDifference = 0.1;

    private static readonly string ModelDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "gguf", "quant-test");

    static QuantizedModelTests()
    {
        GgufImporter.Register();
    }

    private sealed record Reference(string Prompt, int[] Ids, double[] Probs);

    private static Dictionary<string, Reference> LoadReferences()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "quant_next_token.json");
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var references = new Dictionary<string, Reference>();
        foreach (var entry in document.RootElement.EnumerateObject())
        {
            var value = entry.Value;
            references[entry.Name] = new Reference(
                value.GetProperty("prompt").GetString()!,
                [.. value.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32())],
                [.. value.GetProperty("probs").EnumerateArray().Select(e => e.GetDouble())]);
        }

        return references;
    }

    public static IEnumerable<TheoryDataRow<string>> Files() => LoadReferences().Keys.Select(name => new TheoryDataRow<string>(name));

    private static float[] NextTokenLogits(string path, string prompt, bool vulkan)
    {
        DeviceSlot slot;
        if (vulkan)
        {
            var provider = new VulkanProvider();
            if (!provider.CheckAvailability().IsAvailable)
            {
                Assert.Skip("Vulkan is not available");
            }

            var devices = provider.GetDevices();
            var device = devices.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devices[0];
            slot = new DeviceSlot { Device = provider.OpenDevice(device) };
        }
        else
        {
            var provider = new CpuExecutionProvider();
            slot = new DeviceSlot { Device = provider.OpenDevice(provider.GetDevices()[0]) };
        }

        using var model = LlmModel.Load(path, new LlmLoadOptions { Devices = [slot], ContextLength = 256, MaxBatchTokens = 256 });
        try
        {
            using var session = model.CreateSession(WellKnown.DecodeGraph);
            var ids = model.Tokenizer.Encode(prompt, addSpecial: true, parseSpecial: true);
            return session.Process(ids);
        }
        finally
        {
            slot.Device.Dispose();
        }
    }

    private static double[] Probabilities(float[] logits, int[] ids)
    {
        float max = logits.Max();
        double sum = logits.Sum(l => Math.Exp(l - max));
        return [.. ids.Select(id => Math.Exp(logits[id] - max) / sum)];
    }

    [Theory, MemberData(nameof(Files))]
    public void MatchesLlamaCpp(string file)
    {
        string path = Path.Combine(ModelDirectory, file);
        if (!File.Exists(path))
        {
            Assert.Skip($"{path} not present (tools/make-quant-test-models.py creates it)");
            return;
        }

        var reference = LoadReferences()[file];
        var cpu = Probabilities(NextTokenLogits(path, reference.Prompt, vulkan: false), reference.Ids);
        for (int i = 0; i < reference.Ids.Length; i++)
        {
            Assert.True(Math.Abs(cpu[i] - reference.Probs[i]) <= MaxProbabilityDifference, $"{file}: token {reference.Ids[i]} has probability {cpu[i]:F4} on the CPU, llama.cpp {reference.Probs[i]:F4}");
        }

        var vulkan = Probabilities(NextTokenLogits(path, reference.Prompt, vulkan: true), reference.Ids);
        for (int i = 0; i < reference.Ids.Length; i++)
        {
            Assert.True(Math.Abs(vulkan[i] - cpu[i]) <= MaxCpuVulkanDifference, $"{file}: token {reference.Ids[i]} has probability {vulkan[i]:F4} on Vulkan, {cpu[i]:F4} on the CPU");
        }
    }
}
