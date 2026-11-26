using Sharplight.Cpu;
using Sharplight.Cuda;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Serving.Tests;

/// <summary>
/// Prompt-bearing steps padded to shape buckets (<see cref="EngineOptions.StepShapeBuckets"/>): the pad rows extend the last prompt
/// chunk past its end, so the real rows' logits and the steps after them (which read the KV rows the pads wrote over) must match a
/// decoder that runs every step at its exact shape. Real small models on the CPU and on the GPU (Vulkan, or CUDA with
/// <c>SHARPLIGHT_TEST_PROVIDER=cuda</c>); skipped when a file or the GPU is missing.
/// </summary>
[Collection("RealModel")]
public class StepShapeBucketTests
{
    private static readonly string Models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models");

    /// <summary>Attention-only models (with sliding-window layers: granitemoe-swa, gemma4) on the CPU and on the GPU.</summary>
    public static TheoryData<string, bool> Cases() => new()
    {
        { "gguf/tiny-random-qwen1-f32.gguf", false },
        { "gguf/tiny-random-qwen1-f32.gguf", true },
        { "gguf/tiny-random-granitemoe-swa-f32.gguf", false },
        { "gguf/tiny-random-granitemoe-swa-f32.gguf", true },
        { "gguf/tiny-random-gemma4-moe-f32.gguf", false },
        { "gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf", true },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void PaddedSteps_MatchExactSteps(string name, bool gpu)
    {
        string path = Path.Combine(Models, name);
        Assert.SkipUnless(File.Exists(path), $"{path} not present");
        using var device = OpenDevice(gpu);
        using var model = LlmModel.Load(TestHost.Loader, path, new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 256, MaxBatchTokens = 256 });
        using var padded = (MultiSequenceDecoder)model.CreateBatchDecoder(slots: 4);
        using var exact = (MultiSequenceDecoder)model.CreateBatchDecoder(slots: 4);
        Assert.True(padded.KvBlockSize > 0, "the decoder is not paged by default");
        TestContext.Current.SendDiagnosticMessage($"{name}: window-sized caches of {padded.KvWindow} keys");
        padded.PadsStepShapes = true;
        var rng = new Random(11);
        int vocab = Math.Min(1000, model.Tokenizer.VocabSize);
        int[] Tokens(int n) => [.. Enumerable.Range(0, n).Select(_ => rng.Next(10, vocab))];
        int[] a = Tokens(37), b = Tokens(48), c = Tokens(30), d = Tokens(20), f = Tokens(61);
        // A prompt and the first chunk of another (62 rows, padded to 64), that prompt's second chunk behind a decode row (24 rows, padded
        // to 64; the prompt ends on a block boundary, so a block holds pad rows only), a fork of it (which shares the slot's blocks), a
        // third prompt behind decode rows and its continuation, decode steps that read every cache row the pads went past, and a step of
        // exactly 64 rows (bucketed by its runs and logits rows, no pad rows).
        SequenceInput[][] steps =
        [
            [new(0, a), new(1, b.AsMemory(0, 25), WantLogits: false)],
            [new(0, a[..1]), new(1, b.AsMemory(25))],
            [new(0, a[1..2]), new(1, b[..1]), new(3, c[..1]), new(2, c)],
            [new(0, a[2..3]), new(1, b[1..2]), new(3, c[1..2]), new(2, d)],
            [new(0, a[3..4]), new(1, b[2..3]), new(3, c[2..3]), new(2, c[..1])],
            [new(0, a[4..5]), new(1, b[3..4]), new(3, c[3..4]), new(2, c[1..2])],
            [new(0, a[5..6]), new(1, b[4..5]), new(3, c[4..5]), new(2, f)],
            [new(0, a[6..7]), new(1, b[5..6]), new(3, c[5..6]), new(2, f[..1])],
        ];
        int paddedSteps = 0, bucketedSteps = 0;
        for (int s = 0; s < steps.Length; s++)
        {
            if (s == 2)
            {
                // Where the device cannot copy KV rows neither forks, and slot 3 starts a sequence of its own on both.
                Assert.Equal(exact.ForkKv(3, 1), padded.ForkKv(3, 1));
                Assert.Equal(exact.Length(3), padded.Length(3));
            }

            var expected = exact.Step(steps[s]).Select(m => m.ToArray()).ToArray();
            var actual = padded.Step(steps[s]).Select(m => m.ToArray()).ToArray();
            paddedSteps += padded.LastPadRows > 0 ? 1 : 0;
            bucketedSteps += padded.LastStepBucketed ? 1 : 0;
            Assert.Equal(0, exact.LastPadRows);
            Assert.False(exact.LastStepBucketed);
            Assert.Equal(expected.Length, actual.Length);
            for (int u = 0; u < expected.Length; u++)
            {
                Assert.Equal(expected[u].Length, actual[u].Length);
                double error = 0, norm = 0;
                for (int i = 0; i < expected[u].Length; i++)
                {
                    double e = actual[u][i] - expected[u][i];
                    error += e * e;
                    norm += (double)expected[u][i] * expected[u][i];
                }

                // The padded step runs other row counts (other GEMM tiles); KV rows overwritten by pad rows would be far off.
                double relative = Math.Sqrt(error / Math.Max(norm, 1e-30));
                Assert.True(relative <= (gpu ? 2e-2 : 1e-4), $"step {s}, input {u}: logits {relative:E2} (relative L2) away from the exact step");
            }
        }

        // The prompt-bearing steps run in buckets: 0 to 3 padded, 6 (64 rows) by its runs and logits rows only; decode-only steps never.
        Assert.Equal(4, paddedSteps);
        Assert.Equal(5, bucketedSteps);
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
