using Sharplight.Import.HuggingFace;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// Checkpoints whose quantization the importer decodes to dense weights (no matching block type), on Vulkan against PyTorch:
/// FP-Quant (ISTA-DASLab Qwen3-1.7B RTN, MXFP4 and NVFP4 with a Hadamard rotation folded in, BF16), HIGGS (galqiwi Qwen3-1.7B 4-bit,
/// F16), VPTQ (Tiberiw TinyLlama-1.1B, 65536 + 256 residual centroids, BF16) and EXL2 (turboderp TinyLlama-1B 4.0 bpw, F16). The
/// perplexity of a fixed text, teacher-forced token by token, must be within 3 % of the perplexity transformers computes with the same
/// dense weights (tools/dequantized-parity.py), and the greedy continuation of a factual prompt must begin like PyTorch's (a few words: BF16 rounding differences let even the original model's continuations part after five
/// tokens; RTN MXFP4
/// loops on "the capital of France is" in PyTorch too). A wrong rotation, scale layout or permutation multiplies the perplexity many
/// times. Skips when the models are absent.
/// </summary>
public class DequantizedCheckpointParityTests(ITestOutputHelper output)
{
    private const string Text =
        "The history of astronomy is the story of how people learned to read the sky. Early farmers watched the rising of certain "
        + "stars to know when to plant, and sailors steered by the pole star long before anyone understood why it stood still. "
        + "Later, careful measurements of planetary motion led to the laws that describe orbits, and telescopes revealed moons, "
        + "rings and distant galaxies. Today, spacecraft carry instruments far beyond the atmosphere, where the view is clear. "
        + "A small kitchen garden needs little more than sunlight, loose soil and patience. Tomatoes want warmth and a stake to "
        + "climb, beans fix their own nitrogen, and a row of marigolds keeps some pests away. Water in the morning, not at noon, "
        + "and pull weeds while they are young. By late summer the harvest usually outgrows the family, and neighbours get "
        + "baskets of zucchini whether they asked for them or not.";

    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    static DequantizedCheckpointParityTests()
    {
        HuggingFaceImporter.Register();
    }

    [Theory]
    [InlineData("Qwen3-1.7B", 26.642, "Paris. The capital of")]
    [InlineData("quant/fpquant/Qwen3-1.7B-FPQuant-RTN-NVFP4", 57.599, "Paris. The capital of Italy")]
    [InlineData("quant/fpquant/Qwen3-1.7B-FPQuant-RTN-MXFP4", 44.479, "the capital of France is the")]
    [InlineData("quant/higgs/Qwen3-1.7B-HIGGS-4bit", 38.153, "Paris. The capital of Italy")]
    [InlineData("quant/vptq/TinyLlama-1.1B-Chat-v1.0", 16.534, "Paris.")]
    [InlineData("quant/vptq/TinyLlama-1.1B-VPTQ-v8-k65536-256", 19.321, "Paris. The capital of Germany")]
    [InlineData("quant/exl2/TinyLlama-1B-exl2-4.0bpw", 18.656, "Paris. The capital of France")]
    public void PerplexityAndGreedyContinuationMatchPyTorch(string checkpoint, double reference, string continuationStart)
    {
        string path = Path.Combine(Home, "models/hf", checkpoint);
        if (!Directory.Exists(path))
        {
            Assert.Skip("checkpoint not present");
        }

        var (perplexity, continuation) = Evaluate(path);
        output.WriteLine($"{checkpoint}: perplexity {perplexity:F3}, PyTorch {reference:F3}, ratio {perplexity / reference:F4}");
        output.WriteLine($"greedy: The capital of France is{continuation}");
        Assert.InRange(perplexity / reference, 0.97, 1.03);
        // SentencePiece tokenizers drop the leading space when generated tokens are decoded on their own.
        Assert.StartsWith(continuationStart, continuation.TrimStart(), StringComparison.Ordinal);
    }

    private static (double Perplexity, string Continuation) Evaluate(string path)
    {
        var provider = new VulkanProvider();
        if (!provider.CheckAvailability().IsAvailable)
        {
            Assert.Skip("Vulkan is not available");
        }

        var devices = provider.GetDevices();
        var device = devices.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devices[0];
        var slot = new DeviceSlot { Device = provider.OpenDevice(device) };
        try
        {
            using var model = LlmModel.Load(path, new LlmLoadOptions { Devices = [slot], ContextLength = 512, MaxBatchTokens = 64 });
            double perplexity;
            using (var session = model.CreateSession(WellKnown.DecodeGraph))
            {
                var ids = model.Tokenizer.Encode(Text, addSpecial: true, parseSpecial: false);
                double nll = 0;
                float[] logits = [.. session.Process(ids.AsSpan(0, 1))];
                for (int i = 1; i < ids.Length; i++)
                {
                    nll -= LogSoftmax(logits, ids[i]);
                    logits = [.. session.Process(ids.AsSpan(i, 1))];
                }

                perplexity = Math.Exp(nll / (ids.Length - 1));
            }

            using var greedy = model.CreateSession(WellKnown.DecodeGraph);
            var prompt = model.Tokenizer.Encode("The capital of France is", addSpecial: true, parseSpecial: false);
            float[] next = [];
            for (int i = 0; i < prompt.Length; i++)
            {
                next = [.. greedy.Process(prompt.AsSpan(i, 1))];
            }

            var generated = new List<int>();
            for (int step = 0; step < 12; step++)
            {
                int token = Array.IndexOf(next, next.Max());
                generated.Add(token);
                next = [.. greedy.Process([token])];
            }

            return (perplexity, model.Tokenizer.Decode([.. generated]));
        }
        finally
        {
            slot.Device.Dispose();
        }
    }

    private static double LogSoftmax(float[] logits, int token)
    {
        float max = logits.Max();
        double sum = 0;
        foreach (float value in logits)
        {
            sum += Math.Exp(value - max);
        }

        return logits[token] - max - Math.Log(sum);
    }
}
