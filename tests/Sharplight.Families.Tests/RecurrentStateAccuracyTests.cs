using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// The gated delta rule's state in 16 bits (<see cref="LlmLoadOptions.RecurrentStateType"/> bf16 and f16) against the f32 state on a
/// Qwen3.5-style hybrid model on Vulkan: a fixed text is teacher-forced (one 64-token prefill chunk, then one token per step, the
/// decode path whose state traffic the 16-bit state halves) and every step's logits are compared with the f32 run: top-1 token
/// (or a token within <see cref="NearTie"/> of the f32 logit spread), mean KL divergence of the softmax and the largest logit
/// difference relative to the spread. Large: runs with <c>SHARPLIGHT_TEST_LARGE=1</c> when the file is present.
/// </summary>
public class RecurrentStateAccuracyTests(ITestOutputHelper output)
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
        + "baskets of zucchini whether they asked for them or not. The first railways were built to haul coal from mines to "
        + "rivers and harbours, pulled by horses along wooden rails. Steam locomotives replaced the horses, iron replaced the "
        + "wood, and within a few decades lines connected the major cities of several countries. Timetables forced towns to "
        + "agree on a common time, and the telegraph wires that ran beside the tracks carried the news faster than any train.";

    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    static RecurrentStateAccuracyTests()
    {
        GgufImporter.Register();
        HuggingFaceImporter.Register();
    }

    public static TheoryData<string, string, double> Cases() => new()
    {
        { "hf-vllm/Qwen3.8-27B-FP8", "bf16", 0.95 },
        { "hf-vllm/Qwen3.8-27B-FP8", "f16", 0.95 },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void SixteenBitStateAgreesWithF32(string checkpoint, string stateType, double minAgreement)
    {
        string path = Path.Combine(Home, "models", checkpoint);
        Assert.SkipUnless(Directory.Exists(path) || File.Exists(path), $"{path} not present");
        Assert.SkipWhen(Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_LARGE") is null, "large model: set SHARPLIGHT_TEST_LARGE=1 to run it");

        var reference = TeacherForcedLogits(path, DataType.F32);
        var reduced = TeacherForcedLogits(path, DataType.Get(stateType));
        Assert.Equal(reference.Count, reduced.Count);
        int same = 0;
        int agree = 0;
        double kl = 0;
        double worst = 0;
        for (int i = 0; i < reference.Count; i++)
        {
            float[] r = reference[i];
            float[] x = reduced[i];
            int referenceTop = ArgMax(r);
            int top = ArgMax(x);
            double spread = r.Max() - r.Min();
            same += top == referenceTop ? 1 : 0;
            agree += (r[referenceTop] - r[top]) / spread <= NearTie ? 1 : 0;
            kl += KlDivergence(r, x);
            for (int j = 0; j < r.Length; j++)
            {
                worst = Math.Max(worst, Math.Abs(r[j] - x[j]) / spread);
            }
        }

        double agreement = (double)agree / reference.Count;
        output.WriteLine($"{checkpoint} state {stateType}: same top-1 {same}/{reference.Count}, within {NearTie:P0} of the spread {agree}/{reference.Count} "
            + $"({agreement:P1}), mean KL {kl / reference.Count:E2}, max |logit diff| / spread {worst:E2}");
        Assert.True(agreement >= minAgreement, $"{stateType} state: top-1 agreement {agreement:P1} < {minAgreement:P0}");
    }

    /// <summary>Logits after every teacher-forced step: the 64-token prefill chunk, then one token per step.</summary>
    private static List<float[]> TeacherForcedLogits(string path, DataType stateType)
    {
        var provider = new VulkanProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "Vulkan is not available");
        var devices = provider.GetDevices();
        var device = devices.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devices[0];
        var slot = new DeviceSlot { Device = provider.OpenDevice(device) };
        try
        {
            using var model = LlmModel.Load(path, new LlmLoadOptions
            {
                Devices = [slot], ContextLength = 512, MaxBatchTokens = 128, RecurrentStateType = stateType,
            });
            using var session = model.CreateSession(WellKnown.DecodeGraph);
            var ids = model.Tokenizer.Encode(Text, addSpecial: true, parseSpecial: false);
            var result = new List<float[]>();
            int start = 0;
            while (start < ids.Length)
            {
                int length = start == 0 ? Math.Min(PrefillChunk, ids.Length) : 1;
                result.Add([.. session.Process(ids.AsSpan(start, length))]);
                start += length;
            }

            return result;
        }
        finally
        {
            slot.Device.Dispose();
        }
    }

    /// <summary>KL(softmax(p) || softmax(q)) in nats.</summary>
    private static double KlDivergence(float[] p, float[] q)
    {
        double pMax = p.Max();
        double qMax = q.Max();
        double pSum = p.Sum(v => Math.Exp(v - pMax));
        double qSum = q.Sum(v => Math.Exp(v - qMax));
        double kl = 0;
        for (int i = 0; i < p.Length; i++)
        {
            double lp = p[i] - pMax - Math.Log(pSum);
            double lq = q[i] - qMax - Math.Log(qSum);
            kl += Math.Exp(lp) * (lp - lq);
        }

        return kl;
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
