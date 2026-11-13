using Sharplight.Llm;
using Sharplight.Llm.Serving;

namespace Sharplight.Vision.Tests;

/// <summary>
/// Continuous batching: a <see cref="MultiSequenceDecoder"/> advancing several sequences per step must produce the
/// logits of independent single-sequence sessions fed the same chunks.
/// </summary>
public class MultiSequenceTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    public static TheoryData<string, string> Cases => new()
    {
        { "qwen35-0.8b", "cpu" }, { "qwen35-0.8b", "vulkan" },
        { "qwen2.5-0.5b", "cpu" }, { "qwen2.5-0.5b", "vulkan" },
        { "gemma3-1b", "vulkan" }, { "gemma3-1b", "cpu" },
        { "mamba2-130m", "cpu" }, { "mamba2-130m", "vulkan" },
        { "nemotron-h-4b", "vulkan" },
        { "granite-4.0-h-350m", "cpu" }, { "granite-4.0-h-350m", "vulkan" },
        { "lfm2.5-1.2b", "cpu" }, { "lfm2.5-1.2b", "vulkan" },
        { "mamba-130m", "cpu" }, { "mamba-130m", "vulkan" },
        { "rwkv7-g1h-1.5b", "cpu" }, { "rwkv7-g1h-1.5b", "vulkan" },
        { "rwkv6-1.6b", "cpu" }, { "rwkv6-1.6b", "vulkan" },
    };

    private static string ModelPath(string m) => m switch
    {
        "qwen35-0.8b" => Env.Gguf08,
        "qwen2.5-0.5b" => Path.Combine(Home, "models/gguf/Qwen2.5-0.5B-Instruct/qwen2.5-0.5b-instruct-fp16.gguf"),
        "gemma3-1b" => Path.Combine(Home, "models/gguf/gemma-3-1b-it/gemma-3-1b-it-BF16.gguf"),
        "mamba2-130m" => Path.Combine(Home, "models/gguf/mamba2-130m-hf/mamba2-130m-hf-q8_0.gguf"),
        "nemotron-h-4b" => Path.Combine(Home, "models/gguf/NVIDIA-Nemotron-3-Nano-4B/NVIDIA-Nemotron-3-Nano-4B-Q8_0.gguf"),
        "granite-4.0-h-350m" => Path.Combine(Home, "models/gguf/granite-4.0-h-350m/granite-4.0-h-350m-Q4_K_M.gguf"),
        "lfm2.5-1.2b" => Path.Combine(Home, "models/gguf/LFM2.5-1.2B-Instruct/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"),
        "mamba-130m" => Path.Combine(Home, "models/gguf/mamba-130m-hf/mamba-130m-hf.Q4_K_M.gguf"),
        "rwkv7-g1h-1.5b" => Path.Combine(Home, "models/gguf/rwkv7-g1h-1.5b/rwkv7-g1h-1.5b-20260710-ctx10240.i1-Q4_K_M.gguf"),
        "rwkv6-1.6b" => Path.Combine(Home, "models/gguf/rwkv-6-world-1b6/rwkv-6-world-1.6b-Q8_0.gguf"),
        _ => throw new ArgumentException(m),
    };

    private static int[] Tokens(int seed, int n)
    {
        var r = new Random(seed);
        return [.. Enumerable.Range(0, n).Select(_ => r.Next(100, 5000))];
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Matches_Independent_Sessions(string model, string device)
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable("MSEQ_CASE") is { } only && only != $"{model}/{device}", "filtered");
        var path = ModelPath(model);
        Assert.SkipUnless(File.Exists(path), $"model not found: {path}");
        var slot = device == "cpu" ? Env.Cpu() : Env.Vulkan();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var dev = slot!.Device;
        using var llm = LlmModel.Load(path, new LlmLoadOptions { Devices = [slot!], ContextLength = 256, MaxBatchTokens = 128 });
        using var dec = llm.CreateBatchDecoder(4, 256);
        Assert.IsType<MultiSequenceDecoder>(dec);

        // Script: per step, (slot, tokens) inputs. Slot 0 is reset and reused for a new sequence at step 5.
        // Shapes cover: prompts of several flash-attention tiles next to decode rows, short mixed rows (per-query
        // attention kernel), pure batched decode (split kernel) and a slot starting over (fresh recurrent state).
        var steps = new List<(int Slot, int[] Tokens)[]>
        {
            new[] { (0, Tokens(1, 40)), (1, Tokens(2, 7)) },
            new[] { (0, Tokens(3, 1)), (1, Tokens(4, 1)), (2, Tokens(5, 70)) },
            new[] { (0, Tokens(6, 1)), (1, Tokens(7, 1)), (2, Tokens(8, 1)), (3, Tokens(9, 3)) },
            new[] { (0, Tokens(10, 1)), (1, Tokens(11, 1)), (2, Tokens(12, 1)), (3, Tokens(13, 1)) },
            new[] { (1, Tokens(14, 2)), (2, Tokens(15, 33)) },
            new[] { (0, Tokens(16, 9)), (3, Tokens(17, 1)) },   // slot 0: new sequence
            new[] { (0, Tokens(18, 1)), (1, Tokens(19, 1)), (2, Tokens(20, 1)), (3, Tokens(21, 1)) },
        };
        const int ResetStep = 5;
        const int ResetSlot = 0;

        // Reference: one session per sequence, fed the same chunks in order.
        var refs = new Dictionary<(int Slot, int Gen), InferenceSession>();
        try
        {
            float worst = 0;
            for (int s = 0; s < steps.Count; s++)
            {
                if (s == ResetStep)
                {
                    dec.Reset(ResetSlot);
                }

                var inputs = steps[s].Select(i => new SequenceInput(i.Slot, i.Tokens)).ToList();
                var got = dec.Step(inputs);
                for (int u = 0; u < inputs.Count; u++)
                {
                    var key = (steps[s][u].Slot, steps[s][u].Slot == ResetSlot && s >= ResetStep ? 1 : 0);
                    if (!refs.TryGetValue(key, out var sess))
                    {
                        refs[key] = sess = llm.CreateSession();
                    }

                    var want = sess.Process(steps[s][u].Tokens);
                    var row = got[u].ToArray();
                    var (maxAbs, _, rms) = Env.Compare(row, want);
                    worst = Math.Max(worst, (float)rms);
                    // CPU: same math per row up to summation order. Vulkan: f16 tensor-core paths depend on the batch shape (more visibly on the tiny Mamba-2 model).
                    Assert.True(rms < (device == "cpu" ? 1e-3 : model.StartsWith("mamba", StringComparison.Ordinal) || model.StartsWith("rwkv", StringComparison.Ordinal) ? 1e-2 : 5e-3), $"step {s} slot {key.Item1}: rms {rms:G3} maxAbs {maxAbs:G3}");
                    Assert.Equal(ArgMax(want), ArgMax(row));
                    Assert.Equal(sess.Position, dec.Length(key.Item1));
                }
            }
            TestContext.Current.TestOutputHelper?.WriteLine($"{model}/{device}: worst relative rms {worst:G3}");
        }
        finally
        {
            foreach (var r in refs.Values)
            {
                r.Dispose();
            }
        }
    }

    private static int ArgMax(float[] v)
    {
        int b = 0;
        for (int i = 1; i < v.Length; i++)
        {
            if (v[i] > v[b])
            {
                b = i;
            }
        }

        return b;
    }
}
