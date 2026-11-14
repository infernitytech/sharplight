using Sharplight.Cpu;
using Sharplight.Graph;
using Sharplight.HuggingFace;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Speculative;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// End-to-end speculative decoding through the <see cref="InferenceEngine"/> on a model with recurrent layers (tiny random
/// Qwen3.5 with an MTP layer): greedy output with the MTP head or n-gram drafts is token-identical to decoding without
/// speculation, with several sequences in flight and drafts actually verified and rolled back.
/// </summary>
public class SpeculativeDecodingTests
{
    private static readonly string ModelPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "hf", "tiny-random-qwen3.5-mtp");
    private const int Slots = 3;
    private const int Drafts = 3;

    static SpeculativeDecodingTests() => HuggingFaceImporter.Register();

    [Theory]
    [InlineData(false, "mtp", false)]
    [InlineData(false, "ngram", false)]
    [InlineData(true, "mtp", false)]
    [InlineData(true, "ngram", false)]
    [InlineData(true, "mtp", true)]
    [InlineData(true, "ngram", true)]
    public async Task GreedyOutput_Matches_NonSpeculativeDecoding(bool vulkan, string speculator, bool deviceSampling)
        => await Matches_NonSpeculativeDecoding(vulkan, speculator, deviceSampling, SamplingParams.GreedyDefault);

    /// <summary>
    /// Seeded sampling verified on the device: every verification row draws from the sequence's random stream at the position it
    /// would have without drafts, so the output is not just equally distributed but identical to decoding without speculation.
    /// </summary>
    [Theory]
    [InlineData("mtp")]
    [InlineData("ngram")]
    public async Task SeededSampling_OnDevice_Matches_NonSpeculativeDecoding(string speculator)
        => await Matches_NonSpeculativeDecoding(true, speculator, true, new SamplingParams { Temperature = 0.5f, TopK = 4, Seed = 5 });

    private static async Task Matches_NonSpeculativeDecoding(bool vulkan, string speculator, bool deviceSampling, SamplingParams sampling)
    {
        Assert.SkipUnless(Directory.Exists(ModelPath), $"{ModelPath} not present");
        using var device = OpenDevice(vulkan);
        using var model = LlmModel.Load(ModelPath, new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }], ContextLength = 512, MaxBatchTokens = 64, DeviceSampling = deviceSampling,
        });
        Assert.True(model.HasDraftHead);
        var rng = new Random(11);
        // Repetitive prompts give the n-gram speculator something to find; the random model's own output repeats too.
        int[][] prompts = [.. Enumerable.Range(0, 4).Select(i =>
        {
            int[] motif = [.. Enumerable.Range(0, 3 + i).Select(_ => rng.Next(10, 5000))];
            return Enumerable.Repeat(motif, 4).SelectMany(m => m).ToArray();
        })];
        var ct = TestContext.Current.CancellationToken;

        var expected = await Generate(model, model.CreateBatchDecoder(Slots), null, prompts, sampling, ct);
        var decoder = model.CreateBatchDecoder(Slots, maxLogitsRows: Slots * (Drafts + 1), draftTokens: Drafts);
        ISpeculator spec = speculator == "mtp" ? model.CreateMtpSpeculator(Slots, Drafts) : new NgramSpeculator(Drafts, minNgram: 1);
        var (actual, metrics) = await GenerateWithMetrics(model, decoder, spec, prompts, sampling, ct);
        for (int i = 0; i < prompts.Length; i++)
        {
            Assert.Equal(expected[i], actual[i]);
        }

        Assert.True(metrics.DraftTokens > 0, "no drafts were verified");
        // The random MTP head rarely guesses right; the n-gram drafts of the repetitive prompts must be accepted (rows past the first verified).
        Assert.True(speculator != "ngram" || metrics.AcceptedDraftTokens > 0, $"no draft was accepted ({metrics.DraftTokens} verified)");
        TestContext.Current.SendDiagnosticMessage($"{speculator}: {metrics.AcceptedDraftTokens} of {metrics.DraftTokens} drafts accepted");
    }

    /// <summary>
    /// The draft head shares the embedding / LM head buffers of the decoder (<see cref="LlmModel.SharedWeights"/>): compiling it
    /// adds exactly its own weights to the device, and disposing it gives them back.
    /// </summary>
    [Fact]
    public void DraftHead_Reuses_DecoderWeights()
    {
        Assert.SkipUnless(Directory.Exists(ModelPath), $"{ModelPath} not present");
        using var device = OpenDevice(vulkan: true);
        Assert.True(device.SharesWeights);
        using var model = LlmModel.Load(ModelPath, new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }], ContextLength = 512, MaxBatchTokens = 64, DeviceSampling = false,
        });
        static HashSet<IWeightSource> Sources(ComputeGraph g)
            => g.Nodes.SelectMany(n => n.Inputs).Where(v => v.Weight is not null).Select(v => v.Weight!.Source).ToHashSet<IWeightSource>(ReferenceEqualityComparer.Instance);
        var decodeSources = Sources(model.Definition.Graph(WellKnown.DecodeGraph));
        var draftWeights = model.Definition.Graph(WellKnown.DraftGraph).Nodes.SelectMany(n => n.Inputs).Where(v => v.Weight is not null)
            .Select(v => v.Weight!).DistinctBy(w => w.Source, ReferenceEqualityComparer.Instance).ToList();
        long draftOnly = draftWeights.Where(w => !decodeSources.Contains(w.Source)).Sum(w => w.ByteSize);
        long reused = draftWeights.Where(w => decodeSources.Contains(w.Source)).Sum(w => w.ByteSize);
        Assert.True(reused > draftOnly, $"the head should mostly reuse the decoder's weights ({reused} reused, {draftOnly} own)");

        using var decoder = model.CreateBatchDecoder(Slots, maxLogitsRows: Slots * (Drafts + 1), draftTokens: Drafts);
        long resident = model.SharedWeights.ResidentBytes;
        using (var spec = model.CreateMtpSpeculator(Slots, Drafts))
        {
            Assert.Equal(resident + draftOnly, model.SharedWeights.ResidentBytes);
        }

        Assert.Equal(resident, model.SharedWeights.ResidentBytes);
    }

    /// <summary>
    /// <see cref="LlmModel.EstimateFootprint"/> with speculation counts exactly the record states of the rollback form and the
    /// draft head's own weights (not the ones it shares with the decoder).
    /// </summary>
    [Fact]
    public void Footprint_Counts_RollbackRecords_And_DraftHead()
    {
        Assert.SkipUnless(Directory.Exists(ModelPath), $"{ModelPath} not present");
        using var device = OpenDevice(vulkan: false);
        using var model = LlmModel.Load(ModelPath, new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 512, MaxBatchTokens = 64 });
        var plain = model.EstimateFootprint(Slots, 512, maxLogitsRows: Slots * (Drafts + 1));
        var spec = model.EstimateFootprint(Slots, 512, draftTokens: Drafts);
        var withHead = model.EstimateFootprint(Slots, 512, draftTokens: Drafts, draftHead: true);

        var ms = MultiSequence.Transform(model.Definition.Graph(WellKnown.DecodeGraph), Slots, 0, 0, Drafts + 1);
        long records = ms.Graph.Nodes.SelectMany(n => n.Inputs).Select(v => v.State).OfType<StateDefinition>().Distinct()
            .Where(s => s.Name.EndsWith(RecurrentRollback.RecordStateSuffix, StringComparison.Ordinal)).Sum(s => s.Type.ByteSize(s.Shape.Elements()));
        Assert.True(records > 0);
        Assert.Equal(plain.StateBytes + records, spec.StateBytes);
        Assert.Equal(plain.WeightBytes, spec.WeightBytes);

        var decodeSources = model.Definition.Graph(WellKnown.DecodeGraph).Nodes.SelectMany(n => n.Inputs).Where(v => v.Weight is not null)
            .Select(v => v.Weight!.Source).ToHashSet<IWeightSource>(ReferenceEqualityComparer.Instance);
        long headOwn = model.Definition.Graph(WellKnown.DraftGraph).Nodes.SelectMany(n => n.Inputs).Select(v => v.Weight).OfType<WeightInfo>()
            .DistinctBy(w => w.Source, ReferenceEqualityComparer.Instance).Where(w => !decodeSources.Contains(w.Source)).Sum(w => w.ByteSize);
        Assert.Equal(spec.WeightBytes + headOwn, withHead.WeightBytes);
        Assert.True(withHead.StateBytes > spec.StateBytes, "the head's KV cache is counted");
    }

    private static async Task<int[][]> Generate(LlmModel model, IBatchDecoder decoder, ISpeculator? spec, int[][] prompts, SamplingParams sampling, CancellationToken ct)
        => (await GenerateWithMetrics(model, decoder, spec, prompts, sampling, ct)).Tokens;

    private static async Task<(int[][] Tokens, EngineMetricsSnapshot Metrics)> GenerateWithMetrics(LlmModel model, IBatchDecoder decoder, ISpeculator? spec, int[][] prompts, SamplingParams sampling, CancellationToken ct)
    {
        using var engine = new InferenceEngine(decoder, model.Tokenizer, new EngineOptions { Speculator = spec });
        Assert.Equal(spec is not null, engine.Speculates);
        var handles = prompts.Select(p => engine.Submit(new GenerationRequest
        {
            PromptTokens = p, MaxTokens = 40, StopAtEog = false, Logprobs = true, Sampling = sampling,
        }, ct)).ToList();
        var results = await Task.WhenAll(handles.Select(h => h.CompleteAsync(ct)));
        foreach (var r in results)
        {
            Assert.Null(r.Error);
        }

        return ([.. results.Select(r => r.Logprobs.Select(l => l.TokenId).ToArray())], engine.GetMetrics());
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
