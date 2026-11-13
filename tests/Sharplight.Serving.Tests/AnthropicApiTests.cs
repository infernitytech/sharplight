using System.Net;
using System.Text.Json;

namespace Sharplight.Serving.Tests;

public class AnthropicApiTests
{
    private static readonly (string, string)[] Hdr = [("anthropic-version", "2023-06-01")];
    private const string Hi = """{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"hi"}]""";

    [Fact]
    public async Task Message_NonStreaming_WireFormat()
    {
        await using var h = await ApiHost.Create();
        h.Decoder.Script("Hello!");
        var (st, b) = await h.PostJson("/v1/messages", Hi + ",\"temperature\":0}", Hdr);
        Assert.Equal(HttpStatusCode.OK, st);
        Assert.StartsWith("msg_", b.GetProperty("id").GetString());
        Assert.Equal("message", b.GetProperty("type").GetString());
        Assert.Equal("assistant", b.GetProperty("role").GetString());
        Assert.Equal("fake-model", b.GetProperty("model").GetString());
        var block = b.GetProperty("content")[0];
        Assert.Equal("text", block.GetProperty("type").GetString());
        Assert.Equal("Hello!", block.GetProperty("text").GetString());
        Assert.Equal("end_turn", b.GetProperty("stop_reason").GetString());
        Assert.Equal(JsonValueKind.Null, b.GetProperty("stop_sequence").ValueKind);
        var u = b.GetProperty("usage");
        Assert.True(u.GetProperty("input_tokens").GetInt32() > 0);
        Assert.Equal(6, u.GetProperty("output_tokens").GetInt32());
    }

    [Fact]
    public async Task Message_MaxTokens_StopSequence_Reasons()
    {
        await using var h = await ApiHost.Create();
        h.Decoder.Script("alpha beta gamma");
        var (_, b) = await h.PostJson("/v1/messages", """{"model":"m","max_tokens":3,"messages":[{"role":"user","content":"hi"}]}""");
        Assert.Equal("max_tokens", b.GetProperty("stop_reason").GetString());
        Assert.Equal("alp", b.GetProperty("content")[0].GetProperty("text").GetString());
        var (_, s) = await h.PostJson("/v1/messages", """{"model":"m","max_tokens":50,"stop_sequences":["beta"],"messages":[{"role":"user","content":"hi"}]}""");
        Assert.Equal("stop_sequence", s.GetProperty("stop_reason").GetString());
        Assert.Equal("beta", s.GetProperty("stop_sequence").GetString());
        Assert.Equal("alpha ", s.GetProperty("content")[0].GetProperty("text").GetString());
    }

    [Fact]
    public async Task Message_SystemAndBlocks_AreRendered_ToolUseAndResultRoundTrip()
    {
        await using var h = await ApiHost.Create(slots: 1);
        h.Decoder.Script("Sunny.");
        var (st, _) = await h.PostJson("/v1/messages", """
            {"model":"m","max_tokens":20,"system":[{"type":"text","text":"Be terse."}],
             "messages":[{"role":"user","content":[{"type":"text","text":"weather?"}]},
                         {"role":"assistant","content":[{"type":"text","text":"Checking."},{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}]},
                         {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"sunny"}]},{"type":"text","text":"thanks"}]}],
             "tools":[{"name":"get_weather","description":"d","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}]}
            """, Hdr);
        Assert.Equal(HttpStatusCode.OK, st);
        var prompt = System.Text.Encoding.UTF8.GetString([.. h.Decoder.Context(0).Where(t => t < 256).Select(t => (byte)t)]);
        Assert.Contains("<|im_start|>system\nBe terse.<|im_end|>", prompt);
        Assert.Contains("[call get_weather {\"city\": \"Paris\"}]", prompt);
        Assert.Contains("<|im_start|>tool\nsunny<|im_end|>\n<|im_start|>user\nthanks<|im_end|>", prompt);
        Assert.Contains("TOOLS:", prompt);
        Assert.Contains("\"parameters\"", prompt);
    }

    [Fact]
    public async Task Message_ToolUse_Response()
    {
        await using var h = await ApiHost.Create();
        h.Decoder.Script("<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Oslo\"}}</tool_call>");
        var (_, b) = await h.PostJson("/v1/messages", """
            {"model":"m","max_tokens":100,"messages":[{"role":"user","content":"w?"}],
             "tools":[{"name":"get_weather","input_schema":{"type":"object"}}]}
            """, Hdr);
        Assert.Equal("tool_use", b.GetProperty("stop_reason").GetString());
        var blk = b.GetProperty("content").EnumerateArray().Single(c => c.GetProperty("type").GetString() == "tool_use");
        Assert.StartsWith("toolu_", blk.GetProperty("id").GetString());
        Assert.Equal("get_weather", blk.GetProperty("name").GetString());
        Assert.Equal("Oslo", blk.GetProperty("input").GetProperty("city").GetString());
        Assert.DoesNotContain(b.GetProperty("content").EnumerateArray(), c => c.GetProperty("type").GetString() == "text");
    }

    [Fact]
    public async Task Message_Thinking_IsReturnedAsThinkingBlock()
    {
        await using var h = await ApiHost.Create();
        h.Decoder.Script("<think>pondering</think>\n\nanswer");
        var (_, b) = await h.PostJson("/v1/messages", Hi + ",\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":1024}}", Hdr);
        var c = b.GetProperty("content");
        Assert.Equal("thinking", c[0].GetProperty("type").GetString());
        Assert.Equal("pondering", c[0].GetProperty("thinking").GetString());
        Assert.True(c[0].TryGetProperty("signature", out _));
        Assert.Equal("text", c[1].GetProperty("type").GetString());
        Assert.Equal("answer", c[1].GetProperty("text").GetString());
    }

    [Fact]
    public async Task Message_Streaming_OfficialEventSequence()
    {
        await using var h = await ApiHost.Create();
        h.Decoder.Script("Hi!");
        var ev = await h.PostSse("/v1/messages", Hi + ",\"stream\":true}", Hdr);
        Assert.Equal(["message_start", "ping", "content_block_start", "content_block_delta", "content_block_delta", "content_block_delta", "content_block_stop", "message_delta", "message_stop"],
            ev.Select(e => e.Event).ToArray());
        // each data payload repeats its event name in "type"
        Assert.All(ev, e => Assert.Equal(e.Event, JsonDocument.Parse(e.Data).RootElement.GetProperty("type").GetString()));
        var start = JsonDocument.Parse(ev[0].Data).RootElement.GetProperty("message");
        Assert.StartsWith("msg_", start.GetProperty("id").GetString());
        Assert.Equal(0, start.GetProperty("content").GetArrayLength());
        Assert.Equal(JsonValueKind.Null, start.GetProperty("stop_reason").ValueKind);
        Assert.True(start.GetProperty("usage").GetProperty("input_tokens").GetInt32() > 0);
        var cbs = JsonDocument.Parse(ev[2].Data).RootElement;
        Assert.Equal(0, cbs.GetProperty("index").GetInt32());
        Assert.Equal("text", cbs.GetProperty("content_block").GetProperty("type").GetString());
        var text = string.Concat(ev.Where(e => e.Event == "content_block_delta").Select(e => JsonDocument.Parse(e.Data).RootElement.GetProperty("delta").GetProperty("text").GetString()));
        Assert.Equal("Hi!", text);
        var md = JsonDocument.Parse(ev[^2].Data).RootElement;
        Assert.Equal("end_turn", md.GetProperty("delta").GetProperty("stop_reason").GetString());
        Assert.Equal(3, md.GetProperty("usage").GetProperty("output_tokens").GetInt32());
    }

    [Fact]
    public async Task Message_Streaming_ThinkingTextAndToolUseBlocks()
    {
        await using var h = await ApiHost.Create();
        h.Decoder.Script("<think>t</think>\n\nSure.\n<tool_call>{\"name\":\"add\",\"arguments\":{\"a\":1}}</tool_call>");
        var ev = await h.PostSse("/v1/messages", """
            {"model":"m","max_tokens":200,"stream":true,"messages":[{"role":"user","content":"x"}],"tools":[{"name":"add","input_schema":{"type":"object"}}]}
            """, Hdr);
        var starts = ev.Where(e => e.Event == "content_block_start").Select(e => JsonDocument.Parse(e.Data).RootElement).ToList();
        Assert.Equal(["thinking", "text", "tool_use"], starts.Select(s => s.GetProperty("content_block").GetProperty("type").GetString()));
        Assert.Equal([0, 1, 2], starts.Select(s => s.GetProperty("index").GetInt32()));
        Assert.Equal(3, ev.Count(e => e.Event == "content_block_stop"));
        var deltas = ev.Where(e => e.Event == "content_block_delta").Select(e => JsonDocument.Parse(e.Data).RootElement.GetProperty("delta")).ToList();
        Assert.Contains(deltas, d => d.GetProperty("type").GetString() == "thinking_delta" && d.GetProperty("thinking").GetString() == "t");
        var json = deltas.Single(d => d.GetProperty("type").GetString() == "input_json_delta").GetProperty("partial_json").GetString()!;
        Assert.Equal(1, JsonDocument.Parse(json).RootElement.GetProperty("a").GetInt32());
        Assert.Equal("tool_use", JsonDocument.Parse(ev[^2].Data).RootElement.GetProperty("delta").GetProperty("stop_reason").GetString());
        Assert.Equal("message_stop", ev[^1].Event);
    }

    [Fact]
    public async Task Message_Streaming_EmptyOutput_StillHasOneContentBlock()
    {
        await using var h = await ApiHost.Create();
        h.Decoder.Script("");
        var ev = await h.PostSse("/v1/messages", Hi + ",\"stream\":true}", Hdr);
        Assert.Equal(1, ev.Count(e => e.Event == "content_block_start"));
        Assert.Equal("message_stop", ev[^1].Event);
    }

    [Fact]
    public async Task CountTokens_MatchesPrompt()
    {
        await using var h = await ApiHost.Create();
        var (st, b) = await h.PostJson("/v1/messages/count_tokens", """{"model":"m","messages":[{"role":"user","content":"hello"}]}""", Hdr);
        Assert.Equal(HttpStatusCode.OK, st);
        int n = b.GetProperty("input_tokens").GetInt32();
        var (_, b2) = await h.PostJson("/v1/messages/count_tokens", """{"model":"m","system":"sys","messages":[{"role":"user","content":"hello"}]}""", Hdr);
        Assert.True(b2.GetProperty("input_tokens").GetInt32() > n);
        // equals the prompt tokens a real request consumes
        h.Decoder.Script("x");
        var (_, m) = await h.PostJson("/v1/messages", """{"model":"m","max_tokens":5,"messages":[{"role":"user","content":"hello"}]}""", Hdr);
        Assert.Equal(n, m.GetProperty("usage").GetProperty("input_tokens").GetInt32());
    }

    [Theory]
    [InlineData("""{"model":"m","messages":[{"role":"user","content":"hi"}]}""", "max_tokens")]
    [InlineData("""{"model":"m","max_tokens":0,"messages":[{"role":"user","content":"hi"}]}""", "max_tokens")]
    [InlineData("""{"model":"m","max_tokens":5}""", "messages")]
    [InlineData("""{"model":"m","max_tokens":5,"messages":[{"role":"system","content":"x"}]}""", "role")]
    public async Task Errors_HaveAnthropicShape(string body, string mentions)
    {
        await using var h = await ApiHost.Create();
        var (st, b) = await h.PostJson("/v1/messages", body, Hdr);
        Assert.Equal(HttpStatusCode.BadRequest, st);
        Assert.Equal("error", b.GetProperty("type").GetString());
        Assert.Equal("invalid_request_error", b.GetProperty("error").GetProperty("type").GetString());
        Assert.Contains(mentions, b.GetProperty("error").GetProperty("message").GetString());
    }

    [Fact]
    public async Task ModelsEndpoint_UsesAnthropicShape_WhenAnthropicHeaderPresent()
    {
        await using var h = await ApiHost.Create();
        var req = new HttpRequestMessage(HttpMethod.Get, "/v1/models");
        req.Headers.Add("anthropic-version", "2023-06-01");
        using var r = await h.Client.SendAsync(req);
        var b = JsonDocument.Parse(await r.Content.ReadAsStringAsync()).RootElement;
        Assert.Equal("model", b.GetProperty("data")[0].GetProperty("type").GetString());
        Assert.False(b.GetProperty("has_more").GetBoolean());
    }

    [Fact]
    public async Task ContextOverflow_Is400InvalidRequest()
    {
        await using var h = await ApiHost.Create();
        var (st, b) = await h.PostJson("/v1/messages", $"{{\"model\":\"m\",\"max_tokens\":5,\"messages\":[{{\"role\":\"user\",\"content\":\"{new string('x', 5000)}\"}}]}}", Hdr);
        Assert.Equal(HttpStatusCode.BadRequest, st);
        Assert.Equal("invalid_request_error", b.GetProperty("error").GetProperty("type").GetString());
    }
}
