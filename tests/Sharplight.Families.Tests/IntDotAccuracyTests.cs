using Sharplight.Gguf;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// End-to-end precision of the integer-dot matvec (<see cref="DeviceFeatures.IntDot"/>: activations quantized to 8 bits per
/// 32 values): a fixed text is teacher-forced through a model on Vulkan with and without the feature, in chunks of 1, 2 and 3
/// tokens in turn (decode and the few-row matvecs; Q8_0 single rows stay on the f32 matvec), and the last-row logits after
/// every chunk are compared. The int-dot run must pick the reference's top-1 token at nearly every position (or a token the
/// reference rates as tied with it: within <see cref="NearTie"/> of the logit spread, where the 8-bit activations' error of a
/// few percent decides) and stay within a logit bound relative to the reference's logit spread. Skips when the model files
/// are absent.
/// </summary>
public class IntDotAccuracyTests(ITestOutputHelper output)
{
    private const double MinTopOneAgreement = 0.95;

    /// <summary>Reference logit margin, as a fraction of the spread, below which a different top-1 token counts as a tie.</summary>
    private const double NearTie = 0.02;

    /// <summary>Largest logit difference allowed, as a fraction of the reference logits' max - min spread at that position.</summary>
    private const double MaxRelativeLogitDifference = 0.1;

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

    static IntDotAccuracyTests()
    {
        GgufImporter.Register();
    }

    public static TheoryData<string, bool> Models() => new()
    {
        { "gguf/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-Q4_K_M.gguf", false },
        { "gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf", false },
        { "gguf/OLMoE-1B-7B-0924-Instruct/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf", false },
        { "gguf/Mistral-7B-Instruct-v0.3/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", true },
        { "gguf/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf", true },
    };

    [Theory]
    [MemberData(nameof(Models))]
    public void MatchesFloatMatvec(string file, bool large)
    {
        string path = Path.Combine(Home, "models", file);
        if (!File.Exists(path))
        {
            Assert.Skip($"{path} not present");
        }

        if (large && Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_LARGE") is null)
        {
            Assert.Skip("large model: set SHARPLIGHT_TEST_LARGE=1 to run it");
        }

        var reference = TeacherForcedLogits(path, disableIntDot: true);
        var intDot = TeacherForcedLogits(path, disableIntDot: false);
        int agree = 0;
        int exact = 0;
        double worst = 0;
        var flips = new List<string>();
        for (int i = 0; i < reference.Count; i++)
        {
            var a = reference[i];
            var b = intDot[i];
            double spread = a.Max() - a.Min();
            double margin = (a[ArgMax(a)] - a[ArgMax(b)]) / spread;
            if (margin == 0)
            {
                exact++;
            }
            else
            {
                flips.Add($"{margin:P2}");
            }

            if (margin <= NearTie)
            {
                agree++;
            }

            double diff = 0;
            for (int v = 0; v < a.Length; v++)
            {
                diff = Math.Max(diff, Math.Abs(a[v] - b[v]));
            }

            worst = Math.Max(worst, diff / spread);
        }

        double agreement = (double)agree / reference.Count;
        output.WriteLine($"{file}: top-1 agreement {agree}/{reference.Count} ({agreement:P1}, exactly the same token {exact}), "
            + $"max logit difference {worst:P2} of the spread; "
            + $"reference margins of the flipped tokens: [{string.Join(", ", flips)}]");
        Assert.True(agreement >= MinTopOneAgreement, $"{file}: top-1 agreement {agreement:P1} < {MinTopOneAgreement:P0}");
        Assert.True(worst <= MaxRelativeLogitDifference, $"{file}: logit difference {worst:P2} of the spread > {MaxRelativeLogitDifference:P0}");
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

    private static List<float[]> TeacherForcedLogits(string path, bool disableIntDot)
    {
        var provider = new VulkanProvider();
        if (!provider.CheckAvailability().IsAvailable)
        {
            Assert.Skip("Vulkan is not available");
        }

        var devices = provider.GetDevices();
        var device = devices.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devices[0];
        if (!device.Features.Contains(DeviceFeatures.IntDot))
        {
            Assert.Skip("the device has no integer dot product");
        }

        var slot = new DeviceSlot { Device = provider.OpenDevice(device) };
        try
        {
            using var model = LlmModel.Load(path, new LlmLoadOptions
            {
                Devices = [slot], ContextLength = 512, MaxBatchTokens = 512,
                DisabledFeatures = disableIntDot ? new HashSet<string> { DeviceFeatures.IntDot } : new HashSet<string>(),
            });
            using var session = model.CreateSession(WellKnown.DecodeGraph);
            var ids = model.Tokenizer.Encode(Text, addSpecial: true, parseSpecial: false);
            var logits = new List<float[]>();
            int chunk = 1;
            for (int start = 0; start < ids.Length; start += chunk, chunk = chunk % 3 + 1)
            {
                int length = Math.Min(chunk, ids.Length - start);
                logits.Add([.. session.Process(ids.AsSpan(start, length))]);
            }

            return logits;
        }
        finally
        {
            slot.Device.Dispose();
        }
    }
}
