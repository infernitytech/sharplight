using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// FP8 Hugging Face checkpoints as vLLM ships them (block-scaled Qwen3.8-27B-FP8, per-tensor ModelOpt Nemotron-3-Nano-30B-A3B-FP8)
/// against a GGUF quantization of the same model on Vulkan: a fixed text is teacher-forced (one 64-token prefill chunk, which takes
/// the tensor-core GEMMs, then chunks of 1 and 2 tokens on the matvecs) and the top-1 token after every chunk is compared. Both
/// sides are quantized differently, so near-tied positions may flip; the FP8 model must pick the GGUF model's top-1 token (or a
/// token within <see cref="NearTie"/> of its logit spread) at most positions. Large: runs with <c>SHARPLIGHT_TEST_LARGE=1</c> when the
/// files are present; the models are loaded one after the other.
/// </summary>
public class Fp8CheckpointTests(ITestOutputHelper output)
{
    private const double NearTie = 0.02;
    private const int PrefillChunk = 64;

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

    static Fp8CheckpointTests()
    {
        GgufImporter.Register();
        HuggingFaceImporter.Register();
    }

    public static TheoryData<string, string, double> Models() => new()
    {
        { "hf-vllm/Qwen3.8-27B-FP8", "lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf", 0.9 },
        { "hf-vllm/Nemotron-3-Nano-30B-A3B-FP8", "gguf/NVIDIA-Nemotron-3-Nano-30B-A3B/NVIDIA-Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf", 0.8 },
    };

    [Theory]
    [MemberData(nameof(Models))]
    public void AgreesWithGguf(string checkpoint, string gguf, double minAgreement)
    {
        string fp8Path = Path.Combine(Home, "models", checkpoint);
        string ggufPath = Path.Combine(Home, "models", gguf);
        if (!Directory.Exists(fp8Path) || !File.Exists(ggufPath))
        {
            Assert.Skip($"{fp8Path} or {ggufPath} not present");
        }

        if (Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_LARGE") is null)
        {
            Assert.Skip("large model: set SHARPLIGHT_TEST_LARGE=1 to run it");
        }

        var reference = TeacherForcedTopTokens(ggufPath);
        var fp8 = TeacherForcedTopTokens(fp8Path);
        Assert.Equal(reference.Count, fp8.Count);
        int same = 0;
        int agree = 0;
        for (int i = 0; i < reference.Count; i++)
        {
            var (referenceTop, spread, referenceLogits) = reference[i];
            int top = fp8[i].Top;
            same += top == referenceTop ? 1 : 0;
            agree += (referenceLogits[referenceTop] - referenceLogits[top]) / spread <= NearTie ? 1 : 0;
        }

        double agreement = (double)agree / reference.Count;
        output.WriteLine($"{checkpoint}: same top-1 {same}/{reference.Count}, within {NearTie:P0} of the spread {agree}/{reference.Count} ({agreement:P1})");
        Assert.True(agreement >= minAgreement, $"{checkpoint}: top-1 agreement {agreement:P1} < {minAgreement:P0}");
    }

    /// <summary>Top-1 token, logit spread and logits after every teacher-forced chunk.</summary>
    private static List<(int Top, double Spread, float[] Logits)> TeacherForcedTopTokens(string path)
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
            using var model = LlmModel.Load(path, new LlmLoadOptions { Devices = [slot], ContextLength = 512, MaxBatchTokens = 128 });
            using var session = model.CreateSession(WellKnown.DecodeGraph);
            var ids = model.Tokenizer.Encode(Text, addSpecial: true, parseSpecial: false);
            var result = new List<(int, double, float[])>();
            int start = 0;
            int chunk = PrefillChunk;
            while (start < ids.Length)
            {
                int length = Math.Min(chunk, ids.Length - start);
                float[] logits = [.. session.Process(ids.AsSpan(start, length))];
                result.Add((ArgMax(logits), logits.Max() - logits.Min(), logits));
                start += length;
                chunk = chunk == 1 ? 2 : 1;
            }

            return result;
        }
        finally
        {
            slot.Device.Dispose();
        }
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }

        return best;
    }
}
