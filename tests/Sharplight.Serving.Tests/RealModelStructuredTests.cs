using System.Diagnostics;
using System.Text.Json;
using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Structured;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Serving.Tests;

/// <summary>Structured outputs on a real model and vocabulary (skipped when the file is absent): validity and mask cost.</summary>
[Collection("RealModel")]
public class RealModelStructuredTests
{
    private static readonly string ModelPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf");

    private const string Schema = """
        {"type":"object","properties":{
           "name":{"type":"string"},"population":{"type":"integer","minimum":0},
           "landmarks":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":3},
           "capital":{"type":"boolean"}},
         "required":["name","population","landmarks","capital"],"additionalProperties":false}
        """;

    [Fact]
    public async Task Qwen3_JsonSchemaAndForcedToolCall()
    {
        Assert.SkipUnless(File.Exists(ModelPath), "model not found: " + ModelPath);
        ProviderRegistry.Register(() => new VulkanProvider());
        CpuExecutionProvider.Register();
        GgufImporter.Register();
        var avail = ProviderRegistry.Discover().Where(x => x.Availability.IsAvailable).ToList();
        var provider = avail.Where(x => x.Provider.Name != "cpu").Concat(avail).First().Provider;
        using var device = provider.OpenDevice(provider.GetDevices()[0], new DeviceOptions());
        using var model = LlmModel.Load(ModelPath, new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 2048, MaxBatchTokens = 256 });
        var log = TestContext.Current.TestOutputHelper!;

        // Mask cost on a 151k vocabulary: trie build once, then a walk per new grammar state (cached afterwards).
        var sw = Stopwatch.StartNew();
        var constraint = new GrammarConstraint(Grammar.FromJsonSchema(Schema), model.Tokenizer);
        log.WriteLine($"trie+compile {sw.Elapsed.TotalMilliseconds:F1} ms");
        var matcher = constraint.CreateMatcher();
        var mask = new TokenMask(model.Tokenizer.VocabSize);
        sw.Restart();
        matcher.FillAllowed(mask);
        log.WriteLine($"first mask {sw.Elapsed.TotalMilliseconds:F2} ms, {mask.Count()} tokens allowed");
        foreach (int token in model.Tokenizer.Encode("{\"name\": \"", addSpecial: false))
        {
            Assert.True(matcher.Accept(token));
        }

        sw.Restart();
        matcher.FillAllowed(mask);
        double inString = sw.Elapsed.TotalMilliseconds;
        sw.Restart();
        matcher.FillAllowed(mask);
        log.WriteLine($"string-body mask {inString:F2} ms (cached {sw.Elapsed.TotalMilliseconds:F3} ms), {mask.Count()} allowed");

        // The same position in another context (an array item) and in another request reuses the shared stack-top entries.
        foreach (int token in model.Tokenizer.Encode("Paris\", \"population\": 5, \"landmarks\": [\"", addSpecial: false))
        {
            Assert.True(matcher.Accept(token));
        }

        sw.Restart();
        matcher.FillAllowed(mask);
        double inArray = sw.Elapsed.TotalMilliseconds;
        var second = constraint.CreateMatcher();
        foreach (int token in model.Tokenizer.Encode("{\"name\": \"", addSpecial: false))
        {
            Assert.True(second.Accept(token));
        }

        sw.Restart();
        second.FillAllowed(mask);
        log.WriteLine($"string in array {inArray:F2} ms, string in a new request {sw.Elapsed.TotalMilliseconds:F2} ms");

        using var engine = new InferenceEngine(new SessionBatchDecoder(model.CreateSession()), model.Tokenizer);
        var svc = new ModelService(engine, model.ChatTemplate, new ModelServiceOptions { ModelName = "qwen3" });
        sw.Restart();
        var r = await svc.StartChat(new ChatRequest
        {
            Messages = [new ChatTurn("user", "Describe Paris as JSON.")], Sampling = SamplingParams.GreedyDefault, MaxTokens = 200,
            EnableThinking = false, ResponseFormat = OutputFormat.JsonSchema(Schema),
        }).CompleteAsync();
        log.WriteLine($"json answer in {sw.Elapsed.TotalMilliseconds:F0} ms ({r.Usage.CompletionTokens} tokens): {r.Choices[0].Content}");
        using (var doc = JsonDocument.Parse(r.Choices[0].Content))
        {
            Assert.Equal(4, doc.RootElement.EnumerateObject().Count());
            Assert.True(doc.RootElement.GetProperty("population").GetInt64() >= 0);
        }

        var tool = new ToolSpec("get_weather", "Get the current weather for a city",
            """{"type":"object","properties":{"city":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]}},"required":["city","unit"]}""");
        var t = await svc.StartChat(new ChatRequest
        {
            Messages = [new ChatTurn("user", "Tell me a joke.")], Tools = [tool], ToolChoice = ToolChoiceMode.Required,
            Sampling = SamplingParams.GreedyDefault, MaxTokens = 100, EnableThinking = false,
        }).CompleteAsync();
        var call = Assert.Single(t.Choices[0].ToolCalls);
        log.WriteLine($"forced call: {call.Name} {call.ArgumentsJson}");
        using var args = JsonDocument.Parse(call.ArgumentsJson);
        Assert.Contains(args.RootElement.GetProperty("unit").GetString(), new[] { "celsius", "fahrenheit" });
    }
}
