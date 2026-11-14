using System.Text.Json;
using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Llm.Scoring;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// Decision models through <see cref="SpanScorer"/>: the prompts of a tiny model's <c>reference_scores.json</c> (written by its generator in
/// <c>tools/</c>) are given as segments, tokenized like the reference and scored; every option's score matches the reference.
/// </summary>
public class SpanScoringTests
{
    private static readonly string Models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models");

    static SpanScoringTests()
    {
        HuggingFaceImporter.Register();
        GgufImporter.Register();
    }

    /// <summary>(model path under ~/models, reference directory under ~/models, Vulkan, tolerance).</summary>
    public static TheoryData<string, string, bool, float> Cases()
    {
        var data = new TheoryData<string, string, bool, float>();
        (string Model, string Reference)[] models =
        [
            ("gguf/tiny-random-vllm-decision2-f32.gguf", "hf/tiny-random-vllm-decision2"),
            ("hf/tiny-random-vllm-decision2", "hf/tiny-random-vllm-decision2"),
        ];
        foreach (var (model, reference) in models)
        {
            data.Add(model, reference, false, 2e-3f);
            data.Add(model, reference, true, 2e-2f);
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Scores_match_reference(string model, string reference, bool vulkan, float tolerance)
    {
        string path = Path.Combine(Models, model);
        string refPath = Path.Combine(Models, reference, "reference_scores.json");
        Assert.SkipUnless((Directory.Exists(path) || File.Exists(path)) && File.Exists(refPath), $"{path} or {refPath} not present");
        using var device = OpenDevice(vulkan);
        using var scorer = SpanScorer.Load(path, new SpanScorerLoadOptions { Devices = [new DeviceSlot { Device = device }], MaxBatchTokens = 512 });
        using var json = JsonDocument.Parse(File.ReadAllText(refPath));
        foreach (var p in json.RootElement.GetProperty("prompts").EnumerateArray())
        {
            var segments = p.GetProperty("segments").EnumerateArray().Select(s => new PromptSegment(
                s.GetProperty("text").GetString()!,
                s.GetProperty("role").GetString() switch { "question" => PromptSegmentRole.Question, "option" => PromptSegmentRole.Option, _ => PromptSegmentRole.Text },
                s.TryGetProperty("kind", out var k) ? (SpanQuestionKind)k.GetInt32() : SpanQuestionKind.Choice)).ToList();
            var prompt = scorer.Encode(segments);
            Assert.Equal(p.GetProperty("token_ids").EnumerateArray().Select(e => e.GetInt32()), prompt.Tokens);

            var expected = p.GetProperty("scores").EnumerateArray().Select(e => e.GetSingle()).ToArray();
            var result = scorer.Score(prompt);
            Assert.Equal(expected.Length, result.Scores.Count);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(Math.Abs(expected[i] - result.Scores[i]) <= tolerance * Math.Max(1f, Math.Abs(expected[i])),
                    $"option {i}: expected {expected[i]}, got {result.Scores[i]}");
            }

            Assert.Equal(Array.IndexOf(expected, expected.Max()), result.Best(0));
        }
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
