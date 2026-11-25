using Sharplight.Hosting.Loaders;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Hosting.Tests;

/// <summary>The JSON form of the hosting options.</summary>
public sealed class ModelHostingJsonTests
{
    private const string Json = """
        {
          // comments and trailing commas are fine
          "reserve": "2G",
          "budget": 8589934592,
          "maxLoadedModels": 3,
          "idleTimeoutSeconds": 600,
          "lazy": true,
          "defaults": { "device": "vulkan:1", "contextLength": 8192, "slots": 8, "cpuOffload": true },
          "models": [
            { "name": "qwen", "path": "/models/qwen.slm", "device": 0, "pinned": true, "lazy": false },
            { "name": "bge", "path": "/models/bge.slm", "kind": "Embedding" },
            { "name": "rerank", "path": "/models/rerank.slm", "kind": "reranker", "device": "cpu" },
            { "name": "whisper", "path": "/models/whisper.slm", "kind": "task", "task": "speech-recognition", },
          ],
        }
        """;

    [Fact]
    public void ParsesSizesKindsAndDevices()
    {
        var options = ModelHostingJson.Parse(Json);
        Assert.Equal(2L << 30, options.Reserve);
        Assert.Equal(8L << 30, options.Budget);
        Assert.Equal(3, options.MaxLoadedModels);
        Assert.Equal(TimeSpan.FromMinutes(10), options.IdleTimeout);
        Assert.True(options.Lazy);
        Assert.Equal(8192, options.Defaults.ContextLength);
        Assert.Equal(["qwen", "bge", "rerank", "whisper"], options.Models.Select(model => model.Name));
        Assert.Equal("0", options.Models[0].Device);
        Assert.Equal(ModelKind.Embedding, options.Models[1].Kind);
        Assert.Equal(ModelKind.Reranker, options.Models[2].Kind);
        Assert.Equal(ModelKind.Task, options.Models[3].Kind);
        Assert.Equal("speech-recognition", options.Models[3].Task);
    }

    [Fact]
    public void SettingsFallBackToTheDefaults()
    {
        var options = ModelHostingJson.Parse(Json);
        var qwen = HostedModelSettings.Resolve(options.Models[0], options.Defaults, options.Lazy);
        var bge = HostedModelSettings.Resolve(options.Models[1], options.Defaults, options.Lazy);
        Assert.Equal("0", qwen.Device);
        Assert.False(qwen.Lazy);
        Assert.True(qwen.Pinned);
        Assert.Equal(8, qwen.Slots);
        Assert.Equal("vulkan:1", bge.Device);
        Assert.True(bge.Lazy);
        Assert.True(bge.CpuOffload);
        Assert.Equal(8192, bge.ContextLength);
        Assert.Equal("/models/bge.slm", bge.Description);
    }

    [Fact]
    public void RoundTrips()
    {
        var options = ModelHostingJson.Parse(Json);
        var again = ModelHostingJson.Parse(ModelHostingJson.Serialize(options));
        Assert.Equal(options.Reserve, again.Reserve);
        Assert.Equal(options.Models.Select(model => (model.Name, model.Kind, model.Device)), again.Models.Select(model => (model.Name, model.Kind, model.Device)));
    }

    [Theory]
    [InlineData("{ \"reserve\": \"lots\" }")]
    [InlineData("{ \"models\": [ { \"name\": \"x\", \"kind\": \"juggler\" } ] }")]
    [InlineData("null")]
    public void RejectsInvalidFiles(string json)
    {
        Assert.Throws<FormatException>(() => ModelHostingJson.Parse(json));
    }

    [Theory]
    [InlineData("1024", 1024L)]
    [InlineData("512M", 512L << 20)]
    [InlineData("1.5G", 3L << 29)]
    [InlineData("2GiB", 2L << 30)]
    [InlineData("2gb", 2L << 30)]
    public void ParsesMemorySizes(string text, long bytes)
    {
        Assert.Equal(bytes, MemorySize.Parse(text));
    }

    [Fact]
    public void DeviceSpecificationsSplit()
    {
        Assert.Equal(["auto"], DeviceSpecifications.Split(null));
        Assert.Equal(["0", "vulkan:1"], DeviceSpecifications.Split(" 0, vulkan:1 "));
        Assert.True(DeviceSpecifications.IsCpu("CPU"));
        Assert.False(DeviceSpecifications.IsCpu("cpu,0"));
    }

    [Fact]
    public void NumericsAndStepFloorSettingsRoundTrip()
    {
        var defaults = HostedModelSettings.Resolve(new HostedModelOptions { Name = "m", Path = "/m.slm" }, new HostedModelDefaults(), false);
        Assert.True(defaults.BlockScaledFp8);
        Assert.Null(defaults.StepFloor);

        var options = ModelHostingJson.Parse("""
            {
              "defaults": { "blockScaledFp8": false, "stepFloor": "always" },
              "models": [
                { "name": "a", "path": "/a.slm" },
                { "name": "b", "path": "/b.slm", "blockScaledFp8": true },
              ],
            }
            """);
        Assert.False(options.Defaults.BlockScaledFp8);
        Assert.Equal(StepFloorPolicy.Always, options.Defaults.StepFloor);
        var a = HostedModelSettings.Resolve(options.Models[0], options.Defaults, false);
        var b = HostedModelSettings.Resolve(options.Models[1], options.Defaults, false);
        Assert.False(a.BlockScaledFp8);
        Assert.True(b.BlockScaledFp8);
        Assert.Equal(StepFloorPolicy.Always, a.StepFloor);

        var again = ModelHostingJson.Parse(ModelHostingJson.Serialize(options));
        Assert.False(again.Defaults.BlockScaledFp8);
        Assert.Equal(StepFloorPolicy.Always, again.Defaults.StepFloor);
        Assert.True(again.Models[1].BlockScaledFp8);
        Assert.Null(again.Models[0].BlockScaledFp8);
        Assert.Equal(StepFloorPolicy.Full, ModelHostingJson.Parse("""{ "defaults": { "stepFloor": "Full" } }""").Defaults.StepFloor);
    }

    [Fact]
    public void StepRowsPerSequenceRoundTripsAndReachesTheEngine()
    {
        var unset = HostedModelSettings.Resolve(new HostedModelOptions { Name = "m", Path = "/m.slm" }, new HostedModelDefaults(), false);
        Assert.Null(unset.StepRowsPerSequence);
        Assert.Equal(48, EngineOf(unset).BatchTokensPerSequence);
        Assert.Equal(48, new EngineOptions().BatchTokensPerSequence);

        var options = ModelHostingJson.Parse("""{ "defaults": { "stepRowsPerSequence": 32 }, "models": [ { "name": "a", "path": "/a.slm" } ] }""");
        Assert.Equal(32, options.Defaults.StepRowsPerSequence);
        var a = HostedModelSettings.Resolve(options.Models[0], options.Defaults, false);
        Assert.Equal(32, a.StepRowsPerSequence);
        Assert.Equal(32, EngineOf(a).BatchTokensPerSequence);
        Assert.Equal(32, ModelHostingJson.Parse(ModelHostingJson.Serialize(options)).Defaults.StepRowsPerSequence);
    }

    [Fact]
    public void KvAdmissionRoundTripsAndReachesTheEngine()
    {
        var unset = HostedModelSettings.Resolve(new HostedModelOptions { Name = "m", Path = "/m.slm" }, new HostedModelDefaults(), false);
        Assert.Null(unset.KvAdmission);
        Assert.Equal(KvAdmissionPolicy.Queue, EngineOf(unset).KvAdmission);

        var options = ModelHostingJson.Parse("""{ "defaults": { "kvAdmission": "preempt" }, "models": [ { "name": "a", "path": "/a.slm" } ] }""");
        Assert.Equal(KvAdmissionPolicy.Preempt, options.Defaults.KvAdmission);
        var a = HostedModelSettings.Resolve(options.Models[0], options.Defaults, false);
        Assert.Equal(KvAdmissionPolicy.Preempt, a.KvAdmission);
        Assert.Equal(KvAdmissionPolicy.Preempt, EngineOf(a).KvAdmission);
        Assert.Equal(KvAdmissionPolicy.Preempt, ModelHostingJson.Parse(ModelHostingJson.Serialize(options)).Defaults.KvAdmission);
    }

    private static EngineOptions EngineOf(HostedModelSettings settings)
    {
        var context = new HostedModelLoadContext(settings, new ModelHostingOptions(), null!, null!);
        return ChatModelLoaderFactory.CreateChatOptions(context).Engine;
    }
}
