using System.Text.Json;
using Sharplight.Architectures;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vision;

namespace Sharplight.Families.Tests;

/// <summary>
/// DFlash block drafters (llama.cpp <c>dflash</c> GGUF, loaded as the <see cref="AuxiliaryRoles.Draft"/> source of the tiny Llama target of the
/// EAGLE-3 tests; <c>tools/mk_tiny_dflash.py</c>): a plain drafter and a DFlash2 drafter (convolution, selector lattice, sinks, post norms).
/// </summary>
public partial class SpeculativeDecodingTests
{
    /// <summary>Greedy output with the block drafter is token-identical to decoding without it, several sequences in flight.</summary>
    [Theory]
    [InlineData("tiny-random-dflash-f32.gguf", false)]
    [InlineData("tiny-random-dflash-f32.gguf", true)]
    [InlineData("tiny-random-dflash2-f32.gguf", false)]
    [InlineData("tiny-random-dflash2-f32.gguf", true)]
    public async Task DflashDrafter_GreedyOutput_Matches_NonSpeculativeDecoding(string drafterFile, bool vulkan)
    {
        var (target, drafter) = DflashPaths(drafterFile);
        Assert.SkipUnless(File.Exists(drafter), $"{drafter} not present");
        await Matches_NonSpeculativeDecoding(target, vulkan, "mtp", vulkan, SamplingParams.GreedyDefault, m =>
        {
            Assert.Equal(DflashDraftHead.TapsOutput, m.Definition.Properties[WellKnown.DraftHiddenSource]);
            Assert.Equal(8, m.Definition.Property<int>(WellKnown.DraftBlockSize));
        }, [(AuxiliaryRoles.Draft, drafter)]);
    }

    /// <summary>
    /// The block drafter's proposals are the reference's (a float64 numpy forward of llama.cpp's dflash graph over transformers' layer inputs of
    /// the target, greedy verification, 24 generated tokens): two prompts drafted in one batch match the reference's single sequences, for every
    /// proposal whose smallest top-2 gap leaves room for f32 rounding; the first proposal always.
    /// </summary>
    [Theory]
    [InlineData("tiny-random-dflash-f32.gguf", false)]
    [InlineData("tiny-random-dflash-f32.gguf", true)]
    [InlineData("tiny-random-dflash2-f32.gguf", false)]
    [InlineData("tiny-random-dflash2-f32.gguf", true)]
    public async Task DflashDrafter_Proposes_ReferenceDrafts(string drafterFile, bool vulkan)
    {
        var (target, drafter) = DflashPaths(drafterFile);
        string referencePath = drafter + ".reference.json";
        Assert.SkipUnless(File.Exists(drafter) && File.Exists(referencePath), $"{drafter} or its reference not present");
        using var reference = JsonDocument.Parse(File.ReadAllText(referencePath));
        using var device = OpenDevice(vulkan);
        using var model = LlmModel.Load(target, new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }], ContextLength = 512, MaxBatchTokens = 64, DeviceSampling = vulkan,
            Auxiliary = [(AuxiliaryRoles.Draft, drafter)],
        });
        var cases = reference.RootElement.EnumerateArray().Select(e =>
        {
            var engine = e.GetProperty("engine");
            int[] prompt = [.. e.GetProperty("prompt").EnumerateArray().Select(t => t.GetInt32())];
            int[] output = [.. engine.GetProperty("output").EnumerateArray().Select(t => t.GetInt32())];
            return (Prompt: prompt, Sequence: (int[])[.. prompt, .. output],
                Proposals: engine.GetProperty("proposals").EnumerateArray().Select(p => p.EnumerateArray().Select(t => t.GetInt32()).ToArray()).ToArray(),
                Gaps: engine.GetProperty("gaps").EnumerateArray().Select(g => g.GetDouble()).ToArray(),
                Lengths: engine.GetProperty("context_lengths").EnumerateArray().Select(l => l.GetInt32()).ToArray());
        }).ToList();

        var decoder = model.CreateBatchDecoder(Slots, maxLogitsRows: Slots * (Drafts + 1), draftTokens: Drafts);
        var recorder = new RecordingSpeculator(model.CreateMtpSpeculator(Slots, Drafts));
        var outputs = await Generate(model, decoder, recorder, [.. cases.Select(c => c.Prompt)], SamplingParams.GreedyDefault, TestContext.Current.CancellationToken);
        int compared = 0;
        foreach (var c in cases)
        {
            Assert.Equal(c.Sequence[c.Prompt.Length..], outputs[cases.IndexOf(c)][..(c.Sequence.Length - c.Prompt.Length)]);
            for (int j = 0; j < c.Proposals.Length; j++)
            {
                var context = c.Sequence[..c.Lengths[j]];
                var proposal = recorder.Proposals.FirstOrDefault(p => p.Context.SequenceEqual(context));
                if (j == 0)
                {
                    Assert.NotNull(proposal.Drafts);
                }

                if (proposal.Drafts is null || (j > 0 && c.Gaps[j] < 5e-3))
                {
                    continue;
                }

                Assert.Equal(c.Proposals[j], proposal.Drafts);
                compared++;
            }
        }

        TestContext.Current.SendDiagnosticMessage($"{drafterFile}: {compared} proposals compared");
        Assert.True(compared >= cases.Count * 4, $"only {compared} proposals compared");
    }

    /// <summary>
    /// A released target and its DFlash drafter (opt-in: <c>SHARPLIGHT_TEST_DFLASH_TARGET</c> and <c>SHARPLIGHT_TEST_DFLASH_DRAFTER</c>,
    /// optionally <c>SHARPLIGHT_TEST_DFLASH_DRAFTS</c>): greedy output of a raw text prompt with speculation equals the output without,
    /// and the acceptance rate is reported (to compare with llama.cpp's <c>llama-speculative-simple</c> on the same prompt).
    /// </summary>
    [Fact]
    public async Task DflashDrafter_RealCheckpoint_GreedyOutput_Matches_NonSpeculativeDecoding()
    {
        string? target = Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_DFLASH_TARGET");
        string? drafter = Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_DFLASH_DRAFTER");
        Assert.SkipUnless(target is not null && drafter is not null && File.Exists(target) && File.Exists(drafter),
            "SHARPLIGHT_TEST_DFLASH_TARGET / SHARPLIGHT_TEST_DFLASH_DRAFTER not set");
        int drafts = int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_DFLASH_DRAFTS"), out int d) ? d : 7;
        using var device = OpenDevice(vulkan: true);
        using var model = LlmModel.Load(target, new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }], ContextLength = 1024, MaxBatchTokens = 128, DeviceSampling = true,
            Auxiliary = [(AuxiliaryRoles.Draft, drafter)],
        });
        const string Text = "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and type hints.\n\n```python\n";
        int[] prompt = [.. model.Tokenizer.Encode(Text, addSpecial: true, parseSpecial: false)];
        var ct = TestContext.Current.CancellationToken;
        var expected = await Generate(model, model.CreateBatchDecoder(1), null, [prompt], SamplingParams.GreedyDefault, ct);
        var decoder = model.CreateBatchDecoder(1, maxLogitsRows: drafts + 1, draftTokens: drafts);
        var (actual, metrics) = await GenerateWithMetrics(model, decoder, model.CreateMtpSpeculator(1, drafts), [prompt], SamplingParams.GreedyDefault, ct);
        TestContext.Current.SendDiagnosticMessage(
            $"dflash: {metrics.AcceptedDraftTokens} of {metrics.DraftTokens} drafts accepted ({metrics.DraftAcceptanceRate:P1}); prompt {string.Join(',', prompt)}; output {string.Join(',', actual[0])}");
        Assert.Equal(expected[0], actual[0]);
        Assert.True(metrics.AcceptedDraftTokens > 0, "no draft was accepted");
    }

    private static (string Target, string Drafter) DflashPaths(string drafterFile)
    {
        string gguf = Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(ModelPath))!, "gguf");
        return (Path.Combine(gguf, "tiny-random-llama-eagle3-target-f32.gguf"), Path.Combine(gguf, drafterFile));
    }
}
