using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// Recurrent-state rollback (speculative verification on models with recurrent layers), checked at the batch-decoder level on tiny
/// random models covering every recurrent op: a decoder compiled with rollback rows feeds a pending token plus random drafts as one
/// speculative input, keeps a random prefix (<see cref="IBatchDecoder.Truncate"/>), and continues; a decoder without speculation
/// is fed only the kept tokens, one at a time. The logits of every kept position and of the continuation must agree.
/// </summary>
public class RecurrentRollbackTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    private const int DraftTokens = 3;
    private const int Rounds = 12;

    static RecurrentRollbackTests()
    {
        GgufImporter.Register();
        HuggingFaceImporter.Register();
    }

    /// <summary>Tiny models with recurrent layers: path below ~/models and the recurrent ops they use.</summary>
    public static TheoryData<string, string> Models => new()
    {
        { "hf/tiny-random-qwen3.5-mtp", "GatedDeltaRule, CausalConv1d" },
        { "gguf/tiny-random-plamo2-f32.gguf", "SelectiveScan, CausalConv1d" },
        { "hf/tiny-random-MambaForCausalLM", "SelectiveScan (Mamba-1), CausalConv1d" },
        { "hf/tiny-random-granite-4.0-h", "SelectiveScan (Mamba-2), CausalConv1d" },
        { "gguf/tiny-random-arwkv7-f32.gguf", "Wkv7, CausalConv1d (token shift)" },
        { "gguf/tiny-random-rwkv6qwen2-f32.gguf", "Wkv6, CausalConv1d (token shift)" },
        { "hf/tiny-random-lfm2", "CausalConv1d (short conv)" },
    };

    [Theory]
    [MemberData(nameof(Models))]
    public void Cpu_RolledBackState_Matches_NonSpeculativeDecoding(string path, string ops) => Run(path, ops, vulkan: false);

    [Theory]
    [MemberData(nameof(Models))]
    public void Vulkan_RolledBackState_Matches_NonSpeculativeDecoding(string path, string ops) => Run(path, ops, vulkan: true);

    private static void Run(string relativePath, string ops, bool vulkan)
    {
        string path = Path.Combine(Home, "models", relativePath);
        Assert.SkipUnless(File.Exists(path) || Directory.Exists(path), $"{path} not present");
        using var device = OpenDevice(vulkan);
        using var model = LlmModel.Load(path, new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }], ContextLength = 256, MaxBatchTokens = 64, DeviceSampling = false,
            DisabledFeatures = new HashSet<string> { DeviceFeatures.IntDot },
        });
        using var speculative = model.CreateBatchDecoder(2, maxLogitsRows: 2 * (DraftTokens + 1), draftTokens: DraftTokens);
        using var reference = model.CreateBatchDecoder(2, maxLogitsRows: 2);
        Assert.False(speculative.CanTruncate, $"{relativePath} should have recurrent state ({ops})");
        Assert.Equal(DraftTokens + 1, speculative.MaxSpeculativeRows);

        int vocab = speculative.VocabSize;
        var rng = new Random(7);
        int[] prompt = [.. Enumerable.Range(0, 9).Select(_ => rng.Next(1, Math.Min(vocab, 1000)))];
        // Slot 1 runs a second, shorter sequence alongside, so runs of two slots share each step.
        int[] prompt1 = [.. prompt.Take(5).Reverse()];
        var spec = speculative.Step([new SequenceInput(0, prompt), new SequenceInput(1, prompt1)]);
        var refOut = reference.Step([new SequenceInput(0, prompt), new SequenceInput(1, prompt1)]);
        AssertClose(refOut[0].Span, spec[0].Span, "prompt");
        AssertClose(refOut[1].Span, spec[1].Span, "prompt, slot 1");
        int pending = LogitsMath.ArgMax(spec[0].Span);
        int maxKeepSeen = 0;
        for (int round = 0; round < Rounds; round++)
        {
            // Pending token plus drafts: the first draft is right (the greedy continuation) every other round, the rest random.
            int[] input = new int[DraftTokens + 1];
            input[0] = pending;
            for (int j = 1; j < input.Length; j++)
            {
                input[j] = rng.Next(1, Math.Min(vocab, 1000));
            }

            if (round % 2 == 0)
            {
                input[1] = LogitsMath.ArgMax(ReferenceNext(model, prompt, input[0]));
            }

            int before = speculative.Length(0);
            int[] other = [rng.Next(1, Math.Min(vocab, 1000))];
            var rows = speculative.Step([new SequenceInput(0, input, true, input.Length, Speculative: true), new SequenceInput(1, other)]);
            int keep = round == Rounds - 1 ? input.Length : 1 + rng.Next(input.Length);
            maxKeepSeen = Math.Max(maxKeepSeen, keep);
            var logits = rows[0].ToArray();
            var otherLogits = rows[1].ToArray();
            speculative.Truncate(0, before + keep);
            Assert.Equal(before + keep, speculative.Length(0));

            // Reference: the kept tokens one at a time; slot 1 advances once per round on both decoders.
            for (int j = 0; j < keep; j++)
            {
                var r = reference.Step(j == 0 ? [new SequenceInput(0, new[] { input[j] }), new SequenceInput(1, other)] : [new SequenceInput(0, new[] { input[j] })]);
                AssertClose(r[0].Span, logits.AsSpan(j * vocab, vocab), $"round {round}, kept row {j} of {keep}");
                if (j == 0)
                {
                    AssertClose(r[1].Span, otherLogits, $"round {round}, slot 1");
                }
            }

            prompt = [.. prompt, .. input.Take(keep)];

            pending = LogitsMath.ArgMax(logits.AsSpan((keep - 1) * vocab, vocab));
        }

        // The state after the last (fully kept) round continues exactly like the reference.
        var next = speculative.Step([new SequenceInput(0, new[] { pending })]);
        var expected = reference.Step([new SequenceInput(0, new[] { pending })]);
        AssertClose(expected[0].Span, next[0].Span, "continuation");
        Assert.True(maxKeepSeen > 1);
    }

    /// <summary>Greedy logits after <paramref name="context"/> plus <paramref name="token"/>, from a fresh single-sequence session.</summary>
    private static float[] ReferenceNext(LlmModel model, int[] context, int token)
    {
        using var session = model.CreateSession();
        session.Process(context);
        return session.Process([token]).ToArray();
    }

    private static void AssertClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string what)
    {
        float scale = 1e-6f;
        float worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            scale = Math.Max(scale, Math.Abs(expected[i]));
            worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]));
        }

        Assert.True(worst <= 2e-3f * scale, $"{what}: max |diff| {worst} at logit scale {scale}");
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
