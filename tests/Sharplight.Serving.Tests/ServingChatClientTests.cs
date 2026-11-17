using Microsoft.Extensions.AI;
using Sharplight.Extensions.AI;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;

namespace Sharplight.Serving.Tests;

/// <summary><see cref="ServingChatClient"/>: an IChatClient over the batched engine with tenant and cache partition.</summary>
public sealed class ServingChatClientTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static (InferenceEngine Engine, ModelService Service, List<string?> Tenants) Create()
    {
        var tenants = new List<string?>();
        var decoder = new FakeBatchDecoder(2);
        decoder.Script("hello");
        var engine = new InferenceEngine(decoder, new FakeTokenizer(), new EngineOptions
        {
            RequestFinished = finished =>
            {
                lock (tenants)
                {
                    tenants.Add(finished.Tenant);
                }
            },
        });
        return (engine, new ModelService(engine, ApiHost.Template, new ModelServiceOptions { ModelName = "fake", DefaultMaxTokens = 64 }), tenants);
    }

    [Fact]
    public async Task AnswersThroughTheEngineWithUsage()
    {
        var (engine, service, tenants) = Create();
        using var _ = engine;
        using var client = new ServingChatClient(service, new ServingChatClientOptions { Tenant = "space-1" });
        var response = await client.GetResponseAsync("hi", new ChatOptions { Temperature = 0 }, Ct);
        Assert.Equal("hello", response.Text);
        Assert.Equal(Microsoft.Extensions.AI.ChatFinishReason.Stop, response.FinishReason);
        Assert.Equal("fake", response.ModelId);
        Assert.Equal(5, response.Usage!.OutputTokenCount);
        Assert.True(await T.WaitUntil(() => tenants.Count == 1));
        Assert.Equal("space-1", tenants[0]);

        var text = "";
        await foreach (var update in client.GetStreamingResponseAsync("hi", new ChatOptions { AdditionalProperties = new() { [ServingChatClient.TenantProperty] = "other" } }, Ct))
        {
            text += update.Text;
        }

        Assert.Equal("hello", text);
        Assert.True(await T.WaitUntil(() => tenants.Count == 2));
        Assert.Equal("other", tenants[1]);
    }

    [Fact]
    public async Task ThePrefixCacheIsPartitioned()
    {
        var (engine, service, _) = Create();
        using var _ = engine;
        using var a = new ServingChatClient(service, new ServingChatClientOptions { CachePartition = "a" });
        using var b = new ServingChatClient(service, new ServingChatClientOptions { CachePartition = "b" });
        const string Prompt = "a fairly long question that repeats";
        await a.GetResponseAsync(Prompt, cancellationToken: Ct);
        var other = await b.GetResponseAsync(Prompt, cancellationToken: Ct);
        var same = await a.GetResponseAsync(Prompt, cancellationToken: Ct);
        Assert.Equal(0, other.Usage!.CachedInputTokenCount);
        Assert.True(same.Usage!.CachedInputTokenCount > 0);
    }

    [Fact]
    public void MapsMessagesToolsAndOptions()
    {
        var (engine, service, _) = Create();
        using var _ = engine;
        using var client = new ServingChatClient(service, new ServingChatClientOptions { DefaultSystemPrompt = "Be brief.", CachePartition = "p", DefaultMaxTokens = 9 });
        var tool = AIFunctionFactory.Create((string city) => "sunny", "weather", "Weather of a city");
        var request = client.CreateRequest(
        [
            new ChatMessage(ChatRole.User, "Weather in Paris?"),
            new ChatMessage(ChatRole.Assistant, [new FunctionCallContent("call_1", "weather", new Dictionary<string, object?> { ["city"] = "Paris" })]),
            new ChatMessage(ChatRole.Tool, [new FunctionResultContent("call_1", "sunny")]),
        ], new ChatOptions { Tools = [tool], ToolMode = ChatToolMode.RequireSpecific("weather"), TopP = 0.5f, StopSequences = ["END"] });

        Assert.Equal(["system", "user", "assistant", "tool"], request.Messages.Select(message => message.Role));
        Assert.Equal("Be brief.", request.Messages[0].Content);
        var call = Assert.Single(request.Messages[2].ToolCalls!);
        Assert.Equal(("call_1", "weather"), (call.Id, call.Name));
        Assert.Contains("Paris", call.ArgumentsJson, StringComparison.Ordinal);
        Assert.Equal(("call_1", "sunny"), (request.Messages[3].ToolCallId, request.Messages[3].Content));
        Assert.Equal("weather", Assert.Single(request.Tools).Name);
        Assert.Equal(ToolChoiceMode.Function, request.ToolChoice);
        Assert.Equal("weather", request.ToolChoiceFunction);
        Assert.Equal(0.5f, request.Sampling.TopP);
        Assert.Equal(["END"], request.StopStrings);
        Assert.Equal(9, request.MaxTokens);
        Assert.Equal("p", request.CachePartition);
    }
}
