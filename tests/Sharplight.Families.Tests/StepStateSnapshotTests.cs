using Sharplight.Cpu;
using Sharplight.Cuda;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// In-step snapshots of recurrent state (<see cref="Graph.RecurrentSnapshots"/>), checked at the batch-decoder level on models covering
/// every recurrent op: one long prompt step keeps the state at a block boundary inside it (<see cref="SequenceInput.StateSnapshotRows"/>),
/// next to a second slot's run. Another slot attaches the blocks up to that boundary, which restores the kept state, and computes only
/// the rest of the prompt: its logits must agree with the first slot's and with a decoder without snapshots, and so must the next
/// token of both slots. Boundaries at 48 and 112 rows: not chunk ends of the chunked delta rule (64 rows), which then starts the run
/// with a shorter chunk. GPU cases run on Vulkan, or on CUDA with <c>SHARPLIGHT_TEST_PROVIDER=cuda</c>.
/// </summary>
public class StepStateSnapshotTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    private const int PromptRows = 150;

    /// <summary>Models with recurrent layers (path below ~/models, recurrent ops), each on the CPU and on the GPU.</summary>
    public static TheoryData<string, string, bool> Models()
    {
        var data = new TheoryData<string, string, bool>();
        (string Path, string Ops)[] models =
        [
            ("hf/tiny-random-qwen3.5-mtp", "GatedDeltaRule, CausalConv1d"),
            ("gguf/tiny-random-plamo2-f32.gguf", "SelectiveScan, CausalConv1d"),
            ("hf/tiny-random-MambaForCausalLM", "SelectiveScan (Mamba-1), CausalConv1d"),
            ("hf/tiny-random-granite-4.0-h", "SelectiveScan (Mamba-2), CausalConv1d"),
            ("gguf/tiny-random-arwkv7-f32.gguf", "Wkv7, CausalConv1d (token shift)"),
            ("gguf/tiny-random-rwkv6qwen2-f32.gguf", "Wkv6, CausalConv1d (token shift)"),
            ("hf/tiny-random-lfm2", "CausalConv1d (short conv)"),
            ("hf/tiny-random-kimi-linear-f32", "GatedDeltaRule (per-channel decay), CausalConv1d"),
            ("hf/tiny-random-minimax", "GatedDeltaRule (additive rule, time-invariant decay)"),
        ];
        foreach (var (path, ops) in models)
        {
            data.Add(path, ops, false);
            data.Add(path, ops, true);
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void KeptState_ContinuesLikeTheWholePrompt(string path, string ops, bool gpu)
    {
        Run(path, ops, gpu, 48, null);
        Run(path, ops, gpu, 112, null);
    }

    /// <summary>The selective scan's and the RWKV recurrences' state in bf16, on the GPU.</summary>
    [Theory]
    [InlineData("gguf/tiny-random-plamo2-f32.gguf", "SelectiveScan, CausalConv1d")]
    [InlineData("hf/tiny-random-granite-4.0-h", "SelectiveScan (Mamba-2), CausalConv1d")]
    [InlineData("hf/granite-4.0-h-350m", "SelectiveScan (Mamba-2), CausalConv1d")]
    [InlineData("gguf/tiny-random-arwkv7-f32.gguf", "Wkv7, CausalConv1d (token shift)")]
    [InlineData("gguf/tiny-random-rwkv6qwen2-f32.gguf", "Wkv6, CausalConv1d (token shift)")]
    public void SixteenBitScanState_KeptState_ContinuesLikeTheWholePrompt(string path, string ops) => Run(path, ops, true, 48, DataType.BF16);

    /// <summary>A real gated delta net hybrid (Qwen3.5-0.8B: the chunked delta rule on the GPU), f32 and bf16 recurrent state.</summary>
    [Theory]
    [InlineData(null)]
    [InlineData("bf16")]
    public void RealHybrid_KeptState_ContinuesLikeTheWholePrompt(string? stateType)
        => Run("hf/Qwen3.5-0.8B", "GatedDeltaRule (chunked), CausalConv1d", true, 112, stateType is null ? null : DataType.Get(stateType));

    private static void Run(string relativePath, string ops, bool gpu, int keep, DataType? stateType)
    {
        string path = Path.Combine(Home, "models", relativePath);
        Assert.SkipUnless(File.Exists(path) || Directory.Exists(path), $"{path} not present");
        using var device = OpenDevice(gpu);
        using var model = LlmModel.Load(TestHost.Loader, path, new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }], ContextLength = 256, MaxBatchTokens = 256, DeviceSampling = false,
            DisabledFeatures = new HashSet<string> { DeviceFeatures.IntDot }, RecurrentStateType = stateType,
        });
        using var decoder = model.CreateBatchDecoder(3, stateSnapshots: 2);
        using var reference = model.CreateBatchDecoder(2, stateSnapshots: 0);
        Assert.True(decoder.SnapshotsInsideSteps, $"{relativePath} ({ops}): no in-step snapshots");
        Assert.False(reference.SnapshotsInsideSteps);

        int vocab = Math.Min(decoder.VocabSize, 1000);
        var rng = new Random(11);
        int[] prompt = [.. Enumerable.Range(0, PromptRows).Select(_ => rng.Next(1, vocab))];
        int[] other = [.. Enumerable.Range(0, 37).Select(_ => rng.Next(1, vocab))];
        var expected = reference.Step([new SequenceInput(0, prompt), new SequenceInput(1, other)]).Select(m => m.ToArray()).ToArray();
        var kept = decoder.Step([new SequenceInput(0, prompt) { StateSnapshotRows = keep }, new SequenceInput(1, other)]).Select(m => m.ToArray()).ToArray();
        AssertClose(expected[0], kept[0], $"{ops}: the prompt keeping a state after {keep} rows");
        AssertClose(expected[1], kept[1], $"{ops}: the run next to it");

        int bs = decoder.KvBlockSize;
        UInt128[] hashes = [.. Enumerable.Range(1, PromptRows / bs).Select(i => (UInt128)(0x5EED0000UL + (ulong)i))];
        decoder.PublishKvBlocks(0, 0, hashes);
        Assert.True(decoder.SaveKvStateSnapshotAt(0, keep, hashes[keep / bs - 1]), "the kept state was not saved");

        // Slot 2 restores it with the blocks up to the boundary and computes only the rest.
        Assert.Equal(keep, decoder.AttachKvBlocks(2, hashes.AsSpan(0, keep / bs)));
        var restored = decoder.Step([new SequenceInput(2, prompt[keep..])]).ToArray()[0].ToArray();
        // A 16-bit state is rounded where it is kept: the restored slot continues from a state rounded at row `keep`, while the chunked
        // delta rule of the reference (no snapshot, so no chunk ends at `keep`) rounds at other rows, so it agrees within the state's
        // precision.
        AssertClose(expected[0], restored, $"{ops}: the rest of the prompt from the state kept at {keep}", stateType is null ? 2e-3f : 1e-2f);

        // The slot's own state after the whole prompt, and the restored one, continue like the reference.
        int next = LogitsMath.ArgMax(expected[0]);
        var refNext = reference.Step([new SequenceInput(0, new[] { next })]).ToArray()[0].ToArray();
        var both = decoder.Step([new SequenceInput(0, new[] { next }), new SequenceInput(2, new[] { next })]).Select(m => m.ToArray()).ToArray();
        AssertClose(refNext, both[0], $"{ops}: next token after the prompt that kept a state");
        AssertClose(refNext, both[1], $"{ops}: next token after the restored state", stateType is null ? 2e-3f : 1e-2f);
    }

    private static void AssertClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string what, float tolerance = 2e-3f)
    {
        Assert.Equal(expected.Length, actual.Length);
        float scale = 1e-6f;
        float worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            scale = Math.Max(scale, Math.Abs(expected[i]));
            worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]));
        }

        Assert.True(worst <= tolerance * scale, $"{what}: max |diff| {worst} at logit scale {scale}");
    }

    private static IComputeDevice OpenDevice(bool gpu)
    {
        if (!gpu)
        {
            var cpu = new CpuExecutionProvider();
            return cpu.OpenDevice(cpu.GetDevices()[0]);
        }

        IExecutionProvider p = Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_PROVIDER")?.ToLowerInvariant() == CudaProvider.ProviderName
            ? new CudaProvider()
            : new VulkanProvider();
        Assert.SkipUnless(p.CheckAvailability().IsAvailable, $"{p.Name} is not available");
        var devs = p.GetDevices();
        var dev = devs.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devs.FirstOrDefault();
        Assert.SkipWhen(dev is null, $"no {p.Name} device");
        return p.OpenDevice(dev!);
    }
}
