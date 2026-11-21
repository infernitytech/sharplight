using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Families.Tests;

/// <summary>
/// Window-sized KV caches of the single-sequence session (<see cref="DeviceFeatures.WindowKv"/>, the one-slot paged form with a ring for the
/// sliding-window layers' caches): tiny random Cohere2 MoE and Gemma 4 MoE (8-key windows) give the same greedy tokens and logits as the
/// one-slot paged session with full-length caches (bit for bit: the same kernels) and the same tokens as the plain session, for prompts
/// shorter and longer than the window, prefilled in chunks of 16 rows so the ring (48 rows) wraps during the prompt and the decode.
/// The GPU cases run on Vulkan, or on CUDA with <c>SHARPLIGHT_TEST_PROVIDER=cuda</c>.
/// </summary>
public class SessionWindowKvTests
{
    private static readonly string Models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models");

    public static TheoryData<string, bool> Cases()
    {
        var data = new TheoryData<string, bool>();
        foreach (string path in new[] { "gguf/tiny-random-cohere2-moe-f32.gguf", "gguf/tiny-random-gemma4-moe-f32.gguf", "hf/tiny-random-cohere2-moe" })
        {
            data.Add(path, false);
            data.Add(path, true);
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void WindowSizedSessionCaches_DecodeLikeFullCaches(string name, bool gpu)
    {
        string path = Path.Combine(Models, name);
        Assert.SkipUnless(Directory.Exists(path) || File.Exists(path), $"{path} not present");
        var rng = new Random(11);
        int[][] prompts;
        (int[] Tokens, float[][] Logits)[] expected;
        (int[] Tokens, float[][] Logits)[] plain;
        using (var device = OpenDevice(gpu))
        using (var full = Load(path, device, window: false))
        {
            int vocab = Math.Min(1000, full.Tokenizer.VocabSize);
            prompts = [.. new[] { 5, 70 }.Select(n => Enumerable.Range(0, n).Select(_ => rng.Next(10, vocab)).ToArray())];
            using var session = full.CreateSession();
            Assert.Null(session.Compiled.Graph.TryInput(InputRole.KvWindowBlockTable));
            Assert.Null(session.Compiled.Graph.TryInput(InputRole.KvBlockTable));
            plain = [.. prompts.Select(p => Generate(session, p))];
            // The same one-slot paged kernels with full-length caches: the window-sized ring must reproduce them bit for bit.
            using var paged = full.CreateSession(SessionKvLayout.FullLength(256, 16));
            expected = [.. prompts.Select(p => Generate(paged, p))];
        }

        using var device2 = OpenDevice(gpu);
        using var model = Load(path, device2, window: true);
        using var windowed = model.CreateSession();
        var table = windowed.Compiled.Graph.TryInput(InputRole.KvWindowBlockTable);
        Assert.NotNull(table);
        var windowRows = windowed.Compiled.Graph.States.Select(v => v.State!).Where(s => !s.Shape[0].IsSymbolic).Select(s => s.Shape[0].Fixed).Distinct();
        Assert.Equal([48L], windowRows);
        Assert.True(model.EstimateFootprint().StateBytes < LoadFootprint(path, gpu), "the session's estimate does not count the window-sized caches");
        for (int i = 0; i < prompts.Length; i++)
        {
            var (tokens, logits) = Generate(windowed, prompts[i]);
            Assert.Equal(expected[i].Tokens, tokens);
            Assert.Equal(plain[i].Tokens, tokens);
            for (int s = 0; s < logits.Length; s++)
            {
                Assert.Equal(expected[i].Logits[s], logits[s]);
                // The plain graph runs other attention kernels (contiguous caches), so its logits agree to rounding only.
                for (int v = 0; v < logits[s].Length; v++)
                {
                    Assert.True(Math.Abs(logits[s][v] - plain[i].Logits[s][v]) <= 1e-2f + 1e-2f * Math.Abs(plain[i].Logits[s][v]),
                        $"prompt {i}, step {s}, logit {v}: {logits[s][v]} against {plain[i].Logits[s][v]}");
                }
            }
        }
    }

    /// <summary>Greedy decoding of 40 tokens after <paramref name="prompt"/> from a fresh context; the logits of every step.</summary>
    private static (int[] Tokens, float[][] Logits) Generate(InferenceSession session, int[] prompt)
    {
        session.Reset();
        var tokens = new int[40];
        var logits = new float[tokens.Length][];
        float[] last = session.Process(prompt);
        for (int s = 0; s < tokens.Length; s++)
        {
            logits[s] = (float[])last.Clone();
            tokens[s] = Array.IndexOf(last, last.Max());
            last = session.Process([tokens[s]]);
        }

        return (tokens, logits);
    }

    /// <summary>The state bytes of the session's estimate with full-length caches.</summary>
    private static long LoadFootprint(string path, bool gpu)
    {
        using var device = OpenDevice(gpu);
        using var full = Load(path, device, window: false);
        return full.EstimateFootprint().StateBytes;
    }

    private static LlmModel Load(string path, IComputeDevice device, bool window)
    {
        var options = new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 256, MaxBatchTokens = 16 };
        if (!window)
        {
            options.DisabledFeatures.Add(DeviceFeatures.WindowKv);
        }

        return LlmModel.Load(TestHost.Loader, path, options);
    }

    private static IComputeDevice OpenDevice(bool gpu) => WindowKvParityTests.OpenDevice(gpu);
}
