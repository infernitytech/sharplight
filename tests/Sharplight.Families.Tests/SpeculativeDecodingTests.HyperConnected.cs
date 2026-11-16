using System.Text.Json;
using Sharplight.Architectures;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Families.Tests;

public partial class SpeculativeDecodingTests
{
    private static readonly string HyperConnectedModelPath = Path.Combine(Path.GetDirectoryName(ModelPath)!, "tiny-random-qwen3.8-flash-next");

    /// <summary>
    /// The hyper-connected MTP head of Qwen3.8-Flash-Next (qwen4_exp: reads the trunk's residual streams before the final mixer and mixes its
    /// own streams before the LM head): one draft layer, the decode graph's streams as its hidden source, and greedy output with it is
    /// token-identical.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task HyperConnectedMtpHead_GreedyOutput_Matches_NonSpeculativeDecoding(bool vulkan)
        => await Matches_NonSpeculativeDecoding(HyperConnectedModelPath, vulkan, "mtp", vulkan, SamplingParams.GreedyDefault, m =>
        {
            Assert.Equal(1, m.Definition.Properties[WellKnown.DraftLayers]);
            Assert.Equal(HyperConnections.StreamsOutput, m.Definition.Properties[WellKnown.DraftHiddenSource]);
        });

    /// <summary>
    /// The hyper-connected MTP head proposes the reference drafts: transformers' residual streams of the target and transformers' own decoder
    /// layer and stream mixer loaded with the checkpoint's <c>mtp.*</c> tensors, assembled as llama.cpp's qwen4exp MTP graph
    /// (<c>tools/mk_qwen4exp_mtp_reference.py</c>); the first draft from the prompt rows, two more chained on the head's own residual.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task HyperConnectedMtpHead_Proposes_ReferenceDrafts(bool vulkan)
    {
        string referencePath = Path.Combine(HyperConnectedModelPath, "qwen4exp_mtp_reference.json");
        Assert.SkipUnless(File.Exists(referencePath), $"{referencePath} not present");
        using var reference = JsonDocument.Parse(File.ReadAllText(referencePath));
        var cases = reference.RootElement.EnumerateArray().Select(e => (
            Prompt: e.GetProperty("prompt").EnumerateArray().Select(t => t.GetInt32()).ToArray(),
            Pending: e.GetProperty("pending").GetInt32(),
            Drafts: e.GetProperty("drafts").EnumerateArray().Select(t => t.GetInt32()).ToArray())).ToList();
        using var device = OpenDevice(vulkan);
        using var model = LlmModel.Load(TestHost.Loader, HyperConnectedModelPath, new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }], ContextLength = 512, MaxBatchTokens = 64, DeviceSampling = false,
        });
        var decoder = model.CreateBatchDecoder(Slots, maxLogitsRows: Slots * (Drafts + 1), draftTokens: Drafts);
        var recorder = new RecordingSpeculator(model.CreateMtpSpeculator(Slots, Drafts));
        var outputs = await Generate(model, decoder, recorder, [.. cases.Select(c => c.Prompt)], SamplingParams.GreedyDefault, TestContext.Current.CancellationToken);
        for (int i = 0; i < cases.Count; i++)
        {
            var (prompt, pending, drafts) = cases[i];
            Assert.Equal(pending, outputs[i][0]);
            int[] context = [.. prompt, pending];
            var proposal = recorder.Proposals.First(p => p.Context.SequenceEqual(context));
            Assert.Equal(drafts, proposal.Drafts.Take(drafts.Length));
        }
    }
}
