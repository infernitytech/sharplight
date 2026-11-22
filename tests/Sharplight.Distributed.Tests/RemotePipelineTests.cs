using System.Diagnostics;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Distributed.Tests;

/// <summary>
/// Pipeline-parallel generation over segment workers on loopback (in this process, CPU devices): the tokens and logits equal a single
/// CPU run; worker failures fail the next step promptly and cleanly.
/// </summary>
[Collection("Workers")]
public class RemotePipelineTests
{
    private const string Prompt = "The capital of France is";

    public static TheoryData<string, bool> Models => new()
    {
        { "smollm", true },
        { "smollm", false },
        { "deepseek", true },
        { "hybrid", true },
    };

    private static string PathOf(string model) => model switch
    {
        "smollm" => TestModels.SmolLm,
        "deepseek" => TestModels.TinyDeepSeek,
        _ => TestModels.TinyHybrid,
    };

    [Theory]
    [MemberData(nameof(Models))]
    public void LocalAndTwoRemoteStagesMatchOneDevice(string model, bool peerForwarding)
    {
        string path = TestModels.Require(PathOf(model));
        float[][] reference;
        int[] expected;
        using (var cpu = TestModels.Cpu())
        using (var single = TestModels.Load(path, [cpu]))
        using (var decoder = single.CreateBatchDecoder(1, 256))
        {
            (expected, reference) = TestModels.Greedy(decoder, single, Prompt, 8);
        }

        using var w1 = TestModels.StartWorker(path);
        using var w2 = TestModels.StartWorker(path);
        var options = TestModels.Options(w1, w2);
        options.PeerForwarding = peerForwarding;
        using var provider = new RemoteExecutionProvider(options);
        var remote = provider.OpenAll();
        try
        {
            using var local = TestModels.Cpu();
            using var split = TestModels.Load(path, [local, .. remote]);
            using var decoder = split.CreateBatchDecoder(1, 256);
            var (tokens, logits) = TestModels.Greedy(decoder, split, Prompt, 8);
            Assert.Equal(expected, tokens);
            Assert.True(TestModels.MaxDifference(reference, logits) < 1e-3f, $"logits differ by {TestModels.MaxDifference(reference, logits)}");
            Assert.All(remote, d => Assert.True(d.Statistics.Executions > 0));
            Assert.Equal(1, w1.Connections);
        }
        finally
        {
            foreach (var device in remote)
            {
                device.Dispose();
            }
        }
    }

    [Fact]
    public void AllStagesRemoteWithSeveralSequences()
    {
        string path = TestModels.Require(TestModels.SmolLm);
        int[] expected;
        using (var cpu = TestModels.Cpu())
        using (var single = TestModels.Load(path, [cpu]))
        using (var decoder = single.CreateBatchDecoder(4, 256))
        {
            expected = TestModels.Greedy(decoder, single, Prompt, 6, slot: 2).Tokens;
        }

        using var w1 = TestModels.StartWorker(path);
        using var w2 = TestModels.StartWorker(path);
        using var provider = new RemoteExecutionProvider(TestModels.Options(w1, w2));
        var remote = provider.OpenAll();
        try
        {
            using var split = TestModels.Load(path, remote);
            using var decoder = split.CreateBatchDecoder(4, 256);
            Assert.Equal(expected, TestModels.Greedy(decoder, split, Prompt, 6, slot: 2).Tokens);
            Assert.Equal(expected, TestModels.Greedy(decoder, split, Prompt, 6, slot: 0).Tokens);
        }
        finally
        {
            foreach (var device in remote)
            {
                device.Dispose();
            }
        }
    }

    [Fact]
    public void SessionGeneratesTheSameText()
    {
        string path = TestModels.Require(TestModels.SmolLm);
        var options = new GenerationOptions { MaxNewTokens = 10, Sampling = SamplingOptions.Greedy };
        string expected;
        using (var cpu = TestModels.Cpu())
        using (var single = TestModels.Load(path, [cpu]))
        using (var session = single.CreateSession())
        {
            expected = string.Concat(new Generator(session, single.Tokenizer).Generate(Prompt, options).Select(t => t.Text));
        }

        using var worker = TestModels.StartWorker(path);
        using var provider = new RemoteExecutionProvider(TestModels.Options(worker));
        var remote = provider.OpenAll();
        try
        {
            using var local = TestModels.Cpu();
            using var split = TestModels.Load(path, [local, .. remote]);
            using var session = split.CreateSession();
            Assert.Equal(expected, string.Concat(new Generator(session, split.Tokenizer).Generate(Prompt, options).Select(t => t.Text)));
        }
        finally
        {
            foreach (var device in remote)
            {
                device.Dispose();
            }
        }
    }

    [Fact]
    public void KilledWorkerFailsTheNextStepPromptly()
    {
        string path = TestModels.Require(TestModels.SmolLm);
        var w1 = TestModels.StartWorker(path);
        using var w2 = TestModels.StartWorker(path);
        using var provider = new RemoteExecutionProvider(TestModels.Options(w1, w2));
        var remote = provider.OpenAll();
        try
        {
            using var local = TestModels.Cpu();
            using var split = TestModels.Load(path, [local, .. remote]);
            using var decoder = split.CreateBatchDecoder(1, 256);
            TestModels.Greedy(decoder, split, Prompt, 3);
            w1.Dispose();
            var watch = Stopwatch.StartNew();
            var error = Assert.Throws<SegmentWorkerException>(() => TestModels.Greedy(decoder, split, Prompt, 3));
            Assert.True(watch.Elapsed < TimeSpan.FromSeconds(10), $"the failure took {watch.Elapsed}");
            Assert.Contains(w1.Endpoint.ToString(), error.Message + error.Worker);
            // The pipeline stays failed: the next step fails at once instead of waiting.
            watch.Restart();
            Assert.Throws<SegmentWorkerException>(() => TestModels.Greedy(decoder, split, Prompt, 1));
            Assert.True(watch.Elapsed < TimeSpan.FromSeconds(1));
        }
        finally
        {
            foreach (var device in remote)
            {
                device.Dispose();
            }
        }

        // The surviving worker released the failed pipeline's segments with the connection.
        SpinWait.SpinUntil(() => w2.Connections == 0, TimeSpan.FromSeconds(5));
        Assert.Equal(0, w2.Connections);
    }

    [Fact]
    public void WorkerWithAnotherModelIsRefused()
    {
        string path = TestModels.Require(TestModels.SmolLm);
        string other = TestModels.Require(TestModels.TinyDeepSeek);
        using var worker = TestModels.StartWorker(other);
        using var provider = new RemoteExecutionProvider(TestModels.Options(worker));
        var remote = provider.OpenAll();
        try
        {
            using var local = TestModels.Cpu();
            var error = Assert.Throws<SegmentWorkerException>(() =>
            {
                using var split = TestModels.Load(path, [local, .. remote]);
                using var decoder = split.CreateBatchDecoder(1, 256);
            });
            Assert.Contains("model", error.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            foreach (var device in remote)
            {
                device.Dispose();
            }
        }
    }

    [Fact]
    public void WorkerWithChangedWeightsIsRefused()
    {
        string path = TestModels.Require(TestModels.SmolLm);
        var definition = TestModels.Loader.Build(path);
        // The same model with one stored weight of the last layer altered (a different revision or a damaged copy). Weights the importer
        // derives (permuted q/k projections) are checked by metadata only unless WeightVerification.Full.
        var victim = definition.Weights["layers.29.mlp.down.weight"];
        var data = victim.ReadAll();
        data[0] ^= 0xFF;
        var weights = new Dictionary<string, WeightInfo>(definition.Weights)
        {
            [victim.Name] = new WeightInfo { Name = victim.Name, Type = victim.Type, Shape = victim.Shape, Source = new MemoryWeightSource(data), InputOrder = victim.InputOrder },
        };
        var altered = new ModelDefinition
        {
            Architecture = definition.Architecture,
            Name = definition.Name,
            Weights = weights,
            Graphs = definition.Graphs,
            States = definition.States,
            Properties = definition.Properties,
            SourceMetadata = definition.SourceMetadata,
        };
        using var worker = TestModels.StartWorker(path, definition: altered);
        using var provider = new RemoteExecutionProvider(TestModels.Options(worker));
        var remote = provider.OpenAll();
        try
        {
            using var local = TestModels.Cpu();
            var error = Assert.Throws<SegmentWorkerException>(() =>
            {
                using var split = TestModels.Load(path, [local, .. remote]);
                using var decoder = split.CreateBatchDecoder(1, 256);
            });
            Assert.Contains("differ", error.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            foreach (var device in remote)
            {
                device.Dispose();
            }
        }
    }

    [Fact]
    public void PlacementWeightsWorkersByFreeMemory()
    {
        string path = TestModels.Require(TestModels.SmolLm);
        // Two workers whose CPU devices are capped at 3 GiB and 1 GiB: the larger takes about three times the layers.
        using var big = TestModels.StartWorker(path, o => o.Devices = [new DeviceSlot { Device = TestModels.Cpu(), MemoryLimit = 3L << 30 }]);
        using var small = TestModels.StartWorker(path, o => o.Devices = [new DeviceSlot { Device = TestModels.Cpu(), MemoryLimit = 1L << 30 }]);
        using var provider = new RemoteExecutionProvider(TestModels.Options(big, small));
        var remote = provider.OpenAll();
        try
        {
            Assert.InRange(remote[0].GetMemoryInfo().Available, 2.5 * (1L << 30), 3L << 30);
            var graph = TestModels.Loader.Build(path).Graph(WellKnown.DecodeGraph);
            var plan = PlacementPlanner.Plan(graph, new PlacementOptions
            {
                Devices = [.. remote.Select(d => new DeviceSlot { Device = d })],
                ReservePerDevice = 0,
                Compile = new CompileOptions { MaxTokens = 32, ContextLength = 256 },
            });
            double ratio = (double)plan.BytesPerSlot[0] / plan.BytesPerSlot[1];
            Assert.InRange(ratio, 2.0, 4.5);
        }
        finally
        {
            foreach (var device in remote)
            {
                device.Dispose();
            }
        }
    }
}
