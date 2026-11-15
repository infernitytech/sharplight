using System.Text.Json;
using Sharplight.Architectures;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vision;

namespace Sharplight.Families.Tests;

/// <summary>Gemma 4 assistants (shared-KV drafters) on a tiny Gemma 4 target (<c>tools/mk_tiny_gemma4_assistant.py</c>).</summary>
public partial class SpeculativeDecodingTests
{
    /// <summary>
    /// A Gemma 4 assistant (a separate checkpoint, loaded as the <see cref="AuxiliaryRoles.Draft"/> source of a tiny Gemma 4 target with KV-shared
    /// layers and a sliding window shorter than the sequences; it attends the target's keys and values and drafts every token at the pending
    /// token's position): greedy output with it is token-identical, Hugging Face (centroid head) and GGUF (full head).
    /// </summary>
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public async Task Gemma4Assistant_GreedyOutput_Matches_NonSpeculativeDecoding(bool gguf, bool vulkan)
    {
        var (target, assistant) = Gemma4AssistantPaths(gguf);
        Assert.SkipUnless(Path.Exists(assistant), $"{assistant} not present");
        await Matches_NonSpeculativeDecoding(target, vulkan, "mtp", vulkan, SamplingParams.GreedyDefault, m =>
        {
            Assert.Equal(SharedKvDraftHead.FeaturesOutput, m.Definition.Properties[WellKnown.DraftHiddenSource]);
            Assert.Equal(true, m.Definition.Properties[WellKnown.DraftAnchored]);
        }, [(AuxiliaryRoles.Draft, assistant)]);
    }

    /// <summary>
    /// The assistant's proposals after each prompt (two prompts drafted together) are transformers' (SinglePositionMultiTokenCandidateGenerator after
    /// the prompt: the first draft from the target's embedding of the pending token and its last hidden state, the next two from the previous
    /// draft and the assistant's projected output, all at the pending token's position against the prompt's keys and values). The Hugging Face
    /// assistant ranks through its centroid head; the GGUF has the full head (llama.cpp's converter drops the centroids).
    /// </summary>
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public async Task Gemma4Assistant_Proposes_ReferenceDrafts(bool gguf, bool vulkan)
    {
        var (target, assistant) = Gemma4AssistantPaths(gguf);
        string referencePath = Path.Combine(Path.GetDirectoryName(ModelPath)!, "tiny-random-gemma4-assistant", "assistant_reference.json");
        Assert.SkipUnless(Path.Exists(assistant) && File.Exists(referencePath), $"{assistant} or its reference not present");
        using var reference = JsonDocument.Parse(File.ReadAllText(referencePath));
        var cases = reference.RootElement.EnumerateArray().Select(e => (
            Prompt: e.GetProperty("prompt").EnumerateArray().Select(t => t.GetInt32()).ToArray(),
            Pending: e.GetProperty("pending").GetInt32(),
            Drafts: e.GetProperty(gguf ? "drafts_full_head" : "drafts").EnumerateArray().Select(t => t.GetInt32()).ToArray())).ToList();
        using var device = OpenDevice(vulkan);
        using var model = LlmModel.Load(target, new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }], ContextLength = 512, MaxBatchTokens = 64, DeviceSampling = vulkan,
            Auxiliary = [(AuxiliaryRoles.Draft, assistant)],
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
            Assert.Equal(drafts, proposal.Drafts);
        }
    }

    private static (string Target, string Assistant) Gemma4AssistantPaths(bool gguf)
    {
        string models = Path.GetDirectoryName(Path.GetDirectoryName(ModelPath))!;
        return gguf
            ? (Path.Combine(models, "gguf", "tiny-random-gemma4-assistant-target-f32.gguf"), Path.Combine(models, "gguf", "tiny-random-gemma4-assistant-f32.gguf"))
            : (Path.Combine(models, "hf", "tiny-random-gemma4-assistant-target"), Path.Combine(models, "hf", "tiny-random-gemma4-assistant"));
    }
}
