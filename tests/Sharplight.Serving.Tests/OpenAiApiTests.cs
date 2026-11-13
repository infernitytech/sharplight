using System.Net;
using System.Text.Json;

namespace Sharplight.Serving.Tests;

public class OpenAiApiTests
{
    private const string Hi = """{"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":0""";

    [Fact]
    public async Task Models_List_And_Get()
    {
        await using var h = await ApiHost.Create();
        var (st, body) = (HttpStatusCode.OK, JsonDocument.Parse(await h.Client.GetStringAsync("/v1/models")).RootElement);
        Assert.Equal("list", body.GetProperty("object").GetString());
        var m = body.GetProperty("data")[0];
        Assert.Equal("fake-model", m.GetProperty("id").GetString());
        Assert.Equal("model", m.GetProperty("object").GetString());
        Assert.True(m.TryGetProperty("created", out _) && m.TryGetProperty("owned_by", out _));
        var one = JsonDocument.Parse(await h.Client.GetStringAsync("/v1/models/fake-model")).RootElement;
        Assert.Equal("fake-model", one.GetProperty("id").GetString());
        using var missing = await h.Client.GetAsync("/v1/models/nope");
        Assert.Equal(HttpStatusCode.NotFound, missing.StatusCode);
    }

    [Fact]
    public async Task ChatCompletion_NonStreaming_WireFormat()
    {
        await using var h = await ApiHost.Create();
        h.Decoder.Script("Hello there!");
        var (st, b) = await h.PostJson("/v1/chat/completions", Hi + "}");
        Assert.Equal(HttpStatusCode.OK, st);
        Assert.StartsWith("chatcmpl-", b.GetProperty("id").GetString());
        Assert.Equal("chat.completion", b.GetProperty("object").GetString());
        Assert.True(b.GetProperty("created").GetInt64() > 1_600_000_000);
        Assert.Equal("fake-model", b.GetProperty("model").GetString());
        var ch = b.GetProperty("choices")[0];
        Assert.Equal(0, ch.GetProperty("index").GetInt32());
        Assert.Equal("assistant", ch.GetProperty("message").GetProperty("role").GetString());
        Assert.Equal("Hello there!", ch.GetProperty("message").GetProperty("content").GetString());
        Assert.Equal("stop", ch.GetProperty("finish_reason").GetString());
        var u = b.GetProperty("usage");
        Assert.Equal(12, u.GetProperty("completion_tokens").GetInt32());
        Assert.Equal(u.GetProperty("prompt_tokens").GetInt32() + 12, u.GetProperty("total_tokens").GetInt32());
        Assert.Equal(0, u.GetProperty("prompt_tokens_details").GetProperty("cached_tokens").GetInt32());
    }

    [Fact]
    public async Task ChatCompletion_PromptIsRenderedThroughTheChatTemplate()
    {
        await using var h = await ApiHost.Create();
        h.Decoder.Script("ok");
        await h.PostJson("/v1/chat/completions", """{"messages":[{"role":"system","content":"Be brief"},{"role":"user","content":"hi"}]}""");
        var ctx = System.Text.Encoding.UTF8.GetString([.. h.Decoder.Context(0).Concat(h.Decoder.Context(1)).Where(t => t < 128).Select(t => (byte)t)]);
        Assert.Contains("<|im_start|>system\nBe brief<|im_end|>", ctx);
        Assert.Contains("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", ctx);
    }

    [Fact]
    public async Task ChatCompletion_Streaming_SseChunks_Usage_And_Done()
    {
        await using var h = await ApiHost.Create();
        h.Decoder.Script("Hi!");
        var ev = await h.PostSse("/v1/chat/completions", Hi + ",\"stream\":true,\"stream_options\":{\"include_usage\":true}}");
        Assert.All(ev, e => Assert.Null(e.Event));
        Assert.Equal("[DONE]", ev[^1].Data);
        var chunks = ev[..^1].Select(e => JsonDocument.Parse(e.Data).RootElement).ToList();
        Assert.All(chunks, c => { Assert.Equal("chat.completion.chunk", c.GetProperty("object").GetString()); Assert.StartsWith("chatcmpl-", c.GetProperty("id").GetString()); });
        Assert.Single(chunks.Select(c => c.GetProperty("id").GetString()).Distinct());
        Assert.Equal("assistant", chunks[0].GetProperty("choices")[0].GetProperty("delta").GetProperty("role").GetString());
        var text = string.Concat(chunks.Where(c => c.GetProperty("choices").GetArrayLength() > 0)
            .Select(c => c.GetProperty("choices")[0].GetProperty("delta").TryGetProperty("content", out var t) ? t.GetString() : ""));
        Assert.Equal("Hi!", text);
        var fin = chunks.Where(c => c.GetProperty("choices").GetArrayLength() > 0).Last();
        Assert.Equal("stop", fin.GetProperty("choices")[0].GetProperty("finish_reason").GetString());
        var usage = chunks[^1];
        Assert.Equal(0, usage.GetProperty("choices").GetArrayLength());
        Assert.Equal(3, usage.GetProperty("usage").GetProperty("completion_tokens").GetInt32());
        Assert.All(chunks[..^1], c => Assert.Equal(JsonValueKind.Null, c.GetProperty("usage").ValueKind));
    }

    [Fact]
    public async Task ChatCompletion_Streaming_WithoutUsage_HasNoUsageChunk()
    {
        await using var h = await ApiHost.Create();
        h.Decoder.Script("Hi!");
        var ev = await h.PostSse("/v1/chat/completions", Hi + ",\"stream\":true}");
        var chunks = ev[..^1].Select(e => JsonDocument.Parse(e.Data).RootElement).ToList();
        Assert.All(chunks, c => Assert.False(c.TryGetProperty("usage", out _)));
        Assert.Equal("stop", chunks[^1].GetProperty("choices")[0].GetProperty("finish_reason").GetString());
    }

    [Fact]
    public async Task ChatCompletion_ToolCall_NonStreaming()
    {
        await using var h = await ApiHost.Create();
        h.Decoder.Script("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>");
        var (_, b) = await h.PostJson("/v1/chat/completions", """
            {"messages":[{"role":"user","content":"weather?"}],
             "tools":[{"type":"function","function":{"name":"get_weather","description":"d","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}
            """);
        var ch = b.GetProperty("choices")[0];
        Assert.Equal("tool_calls", ch.GetProperty("finish_reason").GetString());
        var msg = ch.GetProperty("message");
        Assert.Equal(JsonValueKind.Null, msg.GetProperty("content").ValueKind);
        var tc = msg.GetProperty("tool_calls")[0];
        Assert.StartsWith("call_", tc.GetProperty("id").GetString());
        Assert.Equal("function", tc.GetProperty("type").GetString());
        Assert.Equal("get_weather", tc.GetProperty("function").GetProperty("name").GetString());
        Assert.Equal("Paris", JsonDocument.Parse(tc.GetProperty("function").GetProperty("arguments").GetString()!).RootElement.GetProperty("city").GetString());
    }

    [Fact]
    public async Task ChatCompletion_ToolDefinitions_ReachTheTemplate_AndToolResultsRoundTrip()
    {
        await using var h = await ApiHost.Create(slots: 1);
        h.Decoder.Script("The weather is sunny.");
        await h.PostJson("/v1/chat/completions", """
            {"messages":[{"role":"user","content":"weather in Paris?"},
                         {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},
                         {"role":"tool","tool_call_id":"call_1","content":"sunny, 21C"}],
             "tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]}
            """);
        var prompt = System.Text.Encoding.UTF8.GetString([.. h.Decoder.Context(0).Where(t => t < 256).Select(t => (byte)t)]);
        Assert.Contains("TOOLS: [{\"type\": \"function\", \"function\": {\"name\": \"get_weather\"", prompt);
        Assert.Contains("[call get_weather {\"city\": \"Paris\"}]", prompt);
        Assert.Contains("<|im_start|>tool\nsunny, 21C", prompt);
    }

    [Fact]
    public async Task ChatCompletion_ToolCall_Streaming_DeltaShape()
    {
        await using var h = await ApiHost.Create();
        h.Decoder.Script("<tool_call>{\"name\":\"add\",\"arguments\":{\"a\":1,\"b\":2}}</tool_call>");
        var ev = await h.PostSse("/v1/chat/completions", """
            {"stream":true,"messages":[{"role":"user","content":"1+2"}],"tools":[{"type":"function","function":{"name":"add","parameters":{"type":"object"}}}]}
            """);
        var chunks = ev[..^1].Select(e => JsonDocument.Parse(e.Data).RootElement).ToList();
        var tcChunk = chunks.Single(c => c.GetProperty("choices")[0].GetProperty("delta").TryGetProperty("tool_calls", out _));
        var tc = tcChunk.GetProperty("choices")[0].GetProperty("delta").GetProperty("tool_calls")[0];
        Assert.Equal(0, tc.GetProperty("index").GetInt32());
        Assert.Equal("add", tc.GetProperty("function").GetProperty("name").GetString());
        Assert.Equal("tool_calls", chunks[^1].GetProperty("choices")[0].GetProperty("finish_reason").GetString());
    }

    [Fact]
    public async Task ChatCompletion_ReasoningContent_IsSeparated()
    {
        await using var h = await ApiHost.Create();
        h.Decoder.Script("<think>hmm</think>\n\nanswer");
        var (_, b) = await h.PostJson("/v1/chat/completions", Hi + "}");
        var msg = b.GetProperty("choices")[0].GetProperty("message");
        Assert.Equal("answer", msg.GetProperty("content").GetString());
        Assert.Equal("hmm", msg.GetProperty("reasoning_content").GetString());
    }

    [Fact]
    public async Task ChatCompletion_StopAndMaxTokens_FinishReasons()
    {
        await using var h = await ApiHost.Create();
        h.Decoder.Script("one two three four");
        var (_, b) = await h.PostJson("/v1/chat/completions", Hi + ",\"stop\":[\"three\"]}");
        Assert.Equal("one two ", b.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString());
        Assert.Equal("stop", b.GetProperty("choices")[0].GetProperty("finish_reason").GetString());
        var (_, b2) = await h.PostJson("/v1/chat/completions", Hi + ",\"max_tokens\":3}");
        Assert.Equal("one", b2.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString());
        Assert.Equal("length", b2.GetProperty("choices")[0].GetProperty("finish_reason").GetString());
        var (_, b3) = await h.PostJson("/v1/chat/completions", Hi + ",\"max_completion_tokens\":2,\"stop\":\"zzz\"}");
        Assert.Equal("on", b3.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString());
    }

    [Fact]
    public async Task ChatCompletion_N_ReturnsIndependentChoices_UsageSumsCompletion()
    {
        await using var h = await ApiHost.Create(slots: 3);
        h.Decoder.Script("abc");
        var (_, b) = await h.PostJson("/v1/chat/completions", Hi + ",\"n\":3}");
        var choices = b.GetProperty("choices");
        Assert.Equal(3, choices.GetArrayLength());
        Assert.Equal([0, 1, 2], choices.EnumerateArray().Select(c => c.GetProperty("index").GetInt32()));
        Assert.All(choices.EnumerateArray(), c => Assert.Equal("abc", c.GetProperty("message").GetProperty("content").GetString()));
        Assert.Equal(9, b.GetProperty("usage").GetProperty("completion_tokens").GetInt32());
        var ev = await h.PostSse("/v1/chat/completions", Hi + ",\"n\":2,\"stream\":true}");
        var idx = ev[..^1].Select(e => JsonDocument.Parse(e.Data).RootElement.GetProperty("choices")[0].GetProperty("index").GetInt32()).Distinct().Order();
        Assert.Equal([0, 1], idx);
    }

    [Fact]
    public async Task ChatCompletion_Logprobs()
    {
        await using var h = await ApiHost.Create();
        h.Decoder.Script("ab");
        var (_, b) = await h.PostJson("/v1/chat/completions", Hi + ",\"logprobs\":true,\"top_logprobs\":2}");
        var content = b.GetProperty("choices")[0].GetProperty("logprobs").GetProperty("content");
        Assert.Equal(2, content.GetArrayLength());
        Assert.Equal("a", content[0].GetProperty("token").GetString());
        Assert.True(content[0].GetProperty("logprob").GetDouble() <= 0);
        Assert.Equal(2, content[0].GetProperty("top_logprobs").GetArrayLength());
        Assert.Equal(97, content[0].GetProperty("bytes")[0].GetInt32());
    }

    [Fact]
    public async Task ChatCompletion_JsonMode_AddsInstruction_And_ThinkingKwargDisablesThinking()
    {
        await using var h = await ApiHost.Create(slots: 1);
        h.Decoder.Script("{}");
        await h.PostJson("/v1/chat/completions", Hi + ",\"response_format\":{\"type\":\"json_object\"},\"chat_template_kwargs\":{\"enable_thinking\":false}}");
        var prompt = System.Text.Encoding.UTF8.GetString([.. h.Decoder.Context(0).Where(t => t < 256).Select(t => (byte)t)]);
        Assert.Contains("valid JSON object", prompt);
        Assert.Contains("<think>\n\n</think>", prompt);
    }

    [Fact]
    public async Task Completions_NonStreaming_Streaming_Echo()
    {
        await using var h = await ApiHost.Create();
        h.Decoder.Script(" world");
        var (_, b) = await h.PostJson("/v1/completions", """{"model":"m","prompt":"hello\u0001","max_tokens":20,"temperature":0}""");
        Assert.Equal("text_completion", b.GetProperty("object").GetString());
        Assert.StartsWith("cmpl-", b.GetProperty("id").GetString());
        var ch = b.GetProperty("choices")[0];
        Assert.Equal(" world", ch.GetProperty("text").GetString());
        Assert.Equal("stop", ch.GetProperty("finish_reason").GetString());
        Assert.Equal(6, b.GetProperty("usage").GetProperty("completion_tokens").GetInt32());

        var (_, e) = await h.PostJson("/v1/completions", """{"prompt":"hello\u0001","echo":true,"max_tokens":3}""");
        Assert.Equal("hello\u0001 wo", e.GetProperty("choices")[0].GetProperty("text").GetString());
        Assert.Equal("length", e.GetProperty("choices")[0].GetProperty("finish_reason").GetString());

        var ev = await h.PostSse("/v1/completions", """{"prompt":"hello\u0001","stream":true}""");
        Assert.Equal("[DONE]", ev[^1].Data);
        var text = string.Concat(ev[..^1].Select(x => JsonDocument.Parse(x.Data).RootElement.GetProperty("choices")[0].GetProperty("text").GetString()));
        Assert.Equal(" world", text);

        var (_, tk) = await h.PostJson("/v1/completions", """{"prompt":[104,105,1],"max_tokens":2}""");
        Assert.Equal(" w", tk.GetProperty("choices")[0].GetProperty("text").GetString());
    }

    [Fact]
    public async Task Errors_HaveOpenAiShape()
    {
        await using var h = await ApiHost.Create();
        var (st, b) = await h.PostJson("/v1/chat/completions", "{\"messages\":[]}");
        Assert.Equal(HttpStatusCode.BadRequest, st);
        var err = b.GetProperty("error");
        Assert.Equal("invalid_request_error", err.GetProperty("type").GetString());
        Assert.Contains("messages", err.GetProperty("message").GetString());
        Assert.Equal("messages", err.GetProperty("param").GetString());
        Assert.True(err.TryGetProperty("code", out _));

        using var bad = await h.PostAsync("/v1/chat/completions", "{nope");
        Assert.Equal(HttpStatusCode.BadRequest, bad.StatusCode);
        Assert.Contains("\"error\"", await bad.Content.ReadAsStringAsync());

        var (s2, b2) = await h.PostJson("/v1/chat/completions", Hi + ",\"n\":99}");
        Assert.Equal(HttpStatusCode.BadRequest, s2);
        var (s3, b3) = await h.PostJson("/v1/chat/completions", """{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}""");
        Assert.Equal(HttpStatusCode.BadRequest, s3);
        Assert.Contains("Image", b3.GetProperty("error").GetProperty("message").GetString());
    }

    [Fact]
    public async Task ContextLengthExceeded_Is400WithCode()
    {
        await using var h = await ApiHost.Create();
        var big = new string('x', 5000);
        var (st, b) = await h.PostJson("/v1/chat/completions", $"{{\"messages\":[{{\"role\":\"user\",\"content\":\"{big}\"}}]}}");
        Assert.Equal(HttpStatusCode.BadRequest, st);
        Assert.Equal("context_length_exceeded", b.GetProperty("error").GetProperty("code").GetString());
    }

    [Fact]
    public async Task QueueFull_Is429()
    {
        await using var h = await ApiHost.Create(slots: 1, eo: new Sharplight.Llm.Serving.EngineOptions { MaxQueue = 1 });
        h.Decoder.StepDelay = TimeSpan.FromMilliseconds(5);
        h.Decoder.Script("a", loop: true);
        var r1 = h.Engine.Submit(new Sharplight.Llm.Serving.GenerationRequest { PromptTokens = T.Prompt("busy"), MaxTokens = 100000 });
        await T.WaitUntil(() => h.Engine.GetMetrics().ActiveSequences == 1);
        var r2 = h.Engine.Submit(new Sharplight.Llm.Serving.GenerationRequest { PromptTokens = T.Prompt("queued"), MaxTokens = 1 });
        using var resp = await h.PostAsync("/v1/chat/completions", Hi + "}");
        Assert.Equal((HttpStatusCode)429, resp.StatusCode);
        Assert.Equal("rate_limit_exceeded", JsonDocument.Parse(await resp.Content.ReadAsStringAsync()).RootElement.GetProperty("error").GetProperty("type").GetString());
        Assert.True(resp.Headers.Contains("Retry-After"));
        r1.Cancel();
        r2.Cancel();
    }

    [Fact]
    public async Task Auth_RequiresKey_BearerOrXApiKey_HealthIsOpen()
    {
        await using var h = await ApiHost.Create(o => o.ApiKeys.Add("secret"));
        h.Decoder.Script("ok");
        var (st, b) = await h.PostJson("/v1/chat/completions", Hi + "}");
        Assert.Equal(HttpStatusCode.Unauthorized, st);
        Assert.Equal("authentication_error", b.GetProperty("error").GetProperty("type").GetString());
        var (s2, _) = await h.PostJson("/v1/chat/completions", Hi + "}", ("Authorization", "Bearer wrong"));
        Assert.Equal(HttpStatusCode.Unauthorized, s2);
        var (s3, _) = await h.PostJson("/v1/chat/completions", Hi + "}", ("Authorization", "Bearer secret"));
        Assert.Equal(HttpStatusCode.OK, s3);
        var (s4, _) = await h.PostJson("/v1/chat/completions", Hi + "}", ("x-api-key", "secret"));
        Assert.Equal(HttpStatusCode.OK, s4);
        using var health = await h.Client.GetAsync("/health");
        Assert.Equal(HttpStatusCode.OK, health.StatusCode);
        // Anthropic style errors on the Anthropic path
        var (s5, b5) = await h.PostJson("/v1/messages", "{}");
        Assert.Equal(HttpStatusCode.Unauthorized, s5);
        Assert.Equal("error", b5.GetProperty("type").GetString());
        Assert.Equal("authentication_error", b5.GetProperty("error").GetProperty("type").GetString());
    }

    [Fact]
    public async Task Cors_HeadersAndPreflight()
    {
        await using var h = await ApiHost.Create(o => o.CorsOrigins.Add("https://app.example"));
        var pre = new HttpRequestMessage(HttpMethod.Options, "/v1/chat/completions");
        pre.Headers.Add("Origin", "https://app.example");
        pre.Headers.Add("Access-Control-Request-Method", "POST");
        using var r = await h.Client.SendAsync(pre);
        Assert.Equal(HttpStatusCode.NoContent, r.StatusCode);
        Assert.Equal("https://app.example", r.Headers.GetValues("Access-Control-Allow-Origin").Single());
        var get = new HttpRequestMessage(HttpMethod.Get, "/v1/models");
        get.Headers.Add("Origin", "https://evil.example");
        using var r2 = await h.Client.SendAsync(get);
        Assert.False(r2.Headers.Contains("Access-Control-Allow-Origin"));
    }

    [Fact]
    public async Task RequestSizeLimit_Is413()
    {
        await using var h = await ApiHost.Create(o => o.MaxRequestBodyBytes = 200);
        var (st, b) = await h.PostJson("/v1/chat/completions", $"{{\"messages\":[{{\"role\":\"user\",\"content\":\"{new string('a', 500)}\"}}]}}");
        Assert.Equal((HttpStatusCode)413, st);
        Assert.Equal("request_too_large", b.GetProperty("error").GetProperty("type").GetString());
    }

    [Fact]
    public async Task ClientDisconnect_CancelsGeneration()
    {
        await using var h = await ApiHost.Create(slots: 1);
        h.Decoder.StepDelay = TimeSpan.FromMilliseconds(3);
        h.Decoder.Script("a", loop: true);
        using var cts = new CancellationTokenSource();
        var req = new HttpRequestMessage(HttpMethod.Post, "/v1/chat/completions") { Content = new StringContent(Hi + ",\"stream\":true,\"max_tokens\":100000}", System.Text.Encoding.UTF8, "application/json") };
        using var resp = await h.Client.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, cts.Token);
        using var stream = await resp.Content.ReadAsStreamAsync();
        var buf = new byte[256];
        Assert.True(await stream.ReadAsync(buf, cts.Token) > 0);
        Assert.Equal(1, h.Engine.GetMetrics().ActiveSequences);
        cts.Cancel();
        stream.Dispose();
        resp.Dispose();
        Assert.True(await T.WaitUntil(() => h.Engine.GetMetrics().ActiveSequences == 0, 5000));
        Assert.Equal(1, h.Engine.GetMetrics().RequestsFinished[Sharplight.Llm.Serving.FinishReason.Cancelled]);
    }

    [Fact]
    public async Task Health_And_Metrics()
    {
        await using var h = await ApiHost.Create();
        h.Decoder.Script("ok");
        await h.PostJson("/v1/chat/completions", Hi + "}");
        var health = JsonDocument.Parse(await h.Client.GetStringAsync("/health")).RootElement;
        Assert.Equal("ok", health.GetProperty("status").GetString());
        Assert.Equal(2, health.GetProperty("slots").GetInt32());
        var metrics = await h.Client.GetStringAsync("/metrics");
        Assert.Contains("# TYPE sharplight_requests_finished_total counter", metrics);
        Assert.Contains("sharplight_requests_finished_total{model=\"fake-model\",reason=\"stop\"} 1", metrics);
        Assert.Contains("sharplight_time_to_first_token_seconds_bucket{model=\"fake-model\",le=\"+Inf\"} 1", metrics);
        Assert.Contains("sharplight_queue_depth{model=\"fake-model\"} 0", metrics);
        Assert.Contains("sharplight_slots{model=\"fake-model\"} 2", metrics);
        Assert.Contains("sharplight_completion_tokens_total{model=\"fake-model\"} 2", metrics);
        // every sample line has "name{labels} value"
        foreach (var line in metrics.Split('\n', StringSplitOptions.RemoveEmptyEntries).Where(l => !l.StartsWith('#')))
        {
            Assert.Matches(@"^[a-z_]+(\{[^}]*\})? -?[0-9.eE+-]+$", line);
        }
    }

    [Fact]
    public async Task ApiSelection_OnlyOpenAi_HidesAnthropic()
    {
        await using var h = await ApiHost.Create(o => o.Anthropic = false);
        using var r = await h.PostAsync("/v1/messages", "{}");
        Assert.Equal(HttpStatusCode.NotFound, r.StatusCode);
        h.Decoder.Script("ok");
        var (st, _) = await h.PostJson("/v1/chat/completions", Hi + "}");
        Assert.Equal(HttpStatusCode.OK, st);
        await using var a = await ApiHost.Create(o => o.OpenAi = false);
        using var r2 = await a.PostAsync("/v1/chat/completions", Hi + "}");
        Assert.Equal(HttpStatusCode.NotFound, r2.StatusCode);
    }
}
