using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// Greedy speculative decoding keeps a draft while it equals the target's own choice, so its output equals decoding without
/// speculation only when a verification row (the pending token plus drafts, several rows of one sequence in one step) computes
/// bit-identical logits to the single-row step that would have produced the same position. This teacher-forces one token
/// sequence through the single-sequence session (what non-speculative generation runs), through a multi-sequence decoder
/// compiled for speculation in steps of one row, and through the same decoder in steps of 1 + drafts rows, and compares every
/// position's logits bit for bit. Skips when the model file is absent.
/// </summary>
public class VerificationRowInvarianceTests(ITestOutputHelper output)
{
    private const int Drafts = 3;

    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    /// <summary>
    /// Model, whether it is large, and the prompt length: Q8_0 runs few-row batches on the integer-dot matvec, Q4_K_M (Q4_K and
    /// Q6_K; unsloth/Qwen3-0.6B-GGUF) single rows too, FP8 few-row batches on the skinny GEMM, and the 27B has recurrent (gated
    /// delta rule) layers; the 700-token prompts are prefilled in two chunks.
    /// </summary>
    public static TheoryData<string, bool, int> Models() => new()
    {
        { "gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf", false, 40 },
        { "gguf/Qwen3-0.6B/Qwen3-0.6B-Q4_K_M.gguf", false, 40 },
        { "gguf/Qwen3-0.6B/Qwen3-0.6B-Q4_K_M.gguf", false, 700 },
        { "lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf", true, 700 },
        { "hf-vllm/Qwen3.8-27B-FP8", true, 700 },
    };

    [Theory]
    [MemberData(nameof(Models))]
    public void VerificationRows_MatchSingleRowSteps(string file, bool large, int promptLength)
    {
        string path = Path.Combine(Home, "models", file);
        if (!File.Exists(path) && !Directory.Exists(path))
        {
            Assert.Skip($"{path} not present");
        }

        if (large && Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_LARGE") is null)
        {
            Assert.Skip("large model: set SHARPLIGHT_TEST_LARGE=1 to run it");
        }

        const int Steps = 6;
        const int Warm = 1;   // one single row between the prompt and the verification steps
        var provider = new VulkanProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "Vulkan is not available");
        var devices = provider.GetDevices();
        var info = devices.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devices[0];
        using var device = provider.OpenDevice(info);
        using var model = LlmModel.Load(TestHost.Loader, path, new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }], ContextLength = promptLength + 64, MaxBatchTokens = 512, DeviceSampling = false,
        });
        var rng = new Random(3);
        int[] tokens = [.. Enumerable.Range(0, promptLength + Warm + Steps * (Drafts + 1)).Select(_ => rng.Next(100, 20000))];

        // Non-speculative generation: the session, prompt in chunks of the batch size, then one row per step.
        var session = new List<float[]>();
        using (var s = model.CreateSession(WellKnown.DecodeGraph))
        {
            session.Add([.. s.Process(tokens.AsSpan(0, promptLength))]);
            for (int i = promptLength; i < tokens.Length; i++)
            {
                session.Add([.. s.Process(tokens.AsSpan(i, 1))]);
            }
        }

        // The speculative decoder: same prompt, then single rows, and then 1 + Drafts rows per step (all drafts kept).
        var single = new List<float[]>();
        var verify = new List<float[]>();
        using (var decoder = model.CreateBatchDecoder(1, maxLogitsRows: Drafts + 1, draftTokens: Drafts))
        {
            single.Add(Prefill(decoder, tokens, promptLength));
            for (int i = promptLength; i < tokens.Length; i++)
            {
                single.Add(decoder.Step([new SequenceInput(0, tokens.AsMemory(i, 1))])[0].ToArray());
            }

            decoder.Reset(0);
            verify.Add(Prefill(decoder, tokens, promptLength));
            for (int i = promptLength; i < promptLength + Warm; i++)
            {
                verify.Add(decoder.Step([new SequenceInput(0, tokens.AsMemory(i, 1))])[0].ToArray());
            }

            for (int i = promptLength + Warm; i < tokens.Length; i += Drafts + 1)
            {
                var rows = decoder.Step([new SequenceInput(0, tokens.AsMemory(i, Drafts + 1), LogitsRows: Drafts + 1, Speculative: true)]);
                // One input: its rows' logits back to back.
                var all = rows[0].ToArray();
                int vocab = all.Length / (Drafts + 1);
                for (int r = 0; r <= Drafts; r++)
                {
                    verify.Add(all[(r * vocab)..((r + 1) * vocab)]);
                }
            }
        }

        // The prompt in chunks of the batch size, as the engine feeds it; the last chunk's logits.
        static float[] Prefill(IBatchDecoder decoder, int[] tokens, int length)
        {
            float[] last = [];
            for (int start = 0; start < length; start += 512)
            {
                int n = Math.Min(512, length - start);
                last = decoder.Step([new SequenceInput(0, tokens.AsMemory(start, n), WantLogits: start + n == length)])[0].ToArray();
            }

            return last;
        }

        string Report(List<float[]> a, List<float[]> b)
        {
            var bad = new List<string>();
            for (int i = 0; i < Math.Min(a.Count, b.Count); i++)
            {
                double diff = 0;
                for (int v = 0; v < a[i].Length; v++)
                {
                    diff = Math.Max(diff, Math.Abs(a[i][v] - b[i][v]));
                }

                if (diff != 0)
                {
                    bad.Add($"{i}:{diff:G3}");
                }
            }

            return bad.Count == 0 ? "identical" : string.Join(" ", bad);
        }

        output.WriteLine($"session vs single-row decoder: {Report(session, single)}");
        output.WriteLine($"single-row vs verification rows: {Report(single, verify)}");
        output.WriteLine($"session vs verification rows: {Report(session, verify)}");
        Assert.Equal("identical", Report(session, verify));
    }
}
