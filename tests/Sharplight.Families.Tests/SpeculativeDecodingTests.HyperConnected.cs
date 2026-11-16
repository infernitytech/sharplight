using System.Text.Json;
using Sharplight.Architectures;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vision;

namespace Sharplight.Families.Tests;

public partial class SpeculativeDecodingTests
{
    private static readonly string HyperConnectedModelPath = Path.Combine(Path.GetDirectoryName(ModelPath)!, "tiny-random-qwen3.8-flash-next");

    /// <summary>
    /// The tiny Qwen3.8-Flash-Next as a variant: "hf" (the checkpoint), "gguf" (llama.cpp's conversion with the NextN block) or "sidecar" (the
    /// conversion without it, <c>--no-mtp</c>, plus the <c>--mtp</c> draft file as the <see cref="AuxiliaryRoles.Draft"/> source).
    /// </summary>
    private static (string Model, Dictionary<string, string> Auxiliary) HyperConnectedVariant(string variant)
    {
        string gguf = Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(ModelPath))!, "gguf");
        return variant switch
        {
            "hf" => (HyperConnectedModelPath, []),
            "gguf" => (Path.Combine(gguf, "tiny-random-qwen3.8-flash-next-f32.gguf"), []),
            "sidecar" => (Path.Combine(gguf, "tiny-random-qwen3.8-flash-next-nomtp-f32.gguf"),
                new Dictionary<string, string> { [AuxiliaryRoles.Draft] = Path.Combine(gguf, "mtp-tiny-random-qwen3.8-flash-next-f32.gguf") }),
            _ => throw new ArgumentOutOfRangeException(nameof(variant)),
        };
    }

    /// <summary>
    /// The hyper-connected MTP head of Qwen3.8-Flash-Next (qwen4_exp: reads the trunk's residual streams before the final mixer and mixes its
    /// own streams before the LM head): one draft layer, the decode graph's streams as its hidden source, and greedy output with it is
    /// token-identical. The tiny n-gram table has 1-wide rows, which the Vulkan gather leaves to the CPU (fallback device).
    /// </summary>
    [Theory]
    [InlineData("hf", false)]
    [InlineData("hf", true)]
    [InlineData("gguf", false)]
    [InlineData("sidecar", true)]
    public async Task HyperConnectedMtpHead_GreedyOutput_Matches_NonSpeculativeDecoding(string variant, bool vulkan)
    {
        var (model, auxiliary) = HyperConnectedVariant(variant);
        await Matches_NonSpeculativeDecoding(model, vulkan, "mtp", vulkan, SamplingParams.GreedyDefault, m =>
        {
            Assert.Equal(1, m.Definition.Properties[WellKnown.DraftLayers]);
            Assert.Equal(HyperConnections.StreamsOutput, m.Definition.Properties[WellKnown.DraftHiddenSource]);
        }, auxiliary, cpuFallback: true);
    }

    /// <summary>
    /// The hyper-connected MTP head proposes the reference drafts: transformers' residual streams of the target and transformers' own decoder
    /// layer and stream mixer loaded with the checkpoint's <c>mtp.*</c> tensors, assembled as llama.cpp's qwen4exp MTP graph
    /// (<c>tools/mk_qwen4exp_mtp_reference.py</c>); the first draft from the prompt rows, two more chained on the head's own residual.
    /// </summary>
    [Theory]
    [InlineData("hf", false)]
    [InlineData("hf", true)]
    [InlineData("gguf", false)]
    [InlineData("gguf", true)]
    [InlineData("sidecar", false)]
    public async Task HyperConnectedMtpHead_Proposes_ReferenceDrafts(string variant, bool vulkan)
    {
        var (modelPath, auxiliary) = HyperConnectedVariant(variant);
        string referencePath = Path.Combine(HyperConnectedModelPath, "qwen4exp_mtp_reference.json");
        Assert.SkipUnless(File.Exists(referencePath) && Path.Exists(modelPath), $"{referencePath} or {modelPath} not present");
        using var reference = JsonDocument.Parse(File.ReadAllText(referencePath));
        var cases = reference.RootElement.EnumerateArray().Select(e => (
            Prompt: e.GetProperty("prompt").EnumerateArray().Select(t => t.GetInt32()).ToArray(),
            Pending: e.GetProperty("pending").GetInt32(),
            Drafts: e.GetProperty("drafts").EnumerateArray().Select(t => t.GetInt32()).ToArray())).ToList();
        using var device = OpenDevice(vulkan);
        using var fallback = vulkan ? OpenDevice(false) : null;
        using var model = LlmModel.Load(TestHost.Loader, modelPath, new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }], ContextLength = 512, MaxBatchTokens = 64, DeviceSampling = false, Auxiliary = auxiliary,
            Fallback = fallback is null ? null : new DeviceSlot { Device = fallback },
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
