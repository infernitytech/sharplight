using System.Net;
using System.Text.Json;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Structured;

namespace Sharplight.Serving.Tests;

/// <summary>
/// Structured outputs end to end: OpenAI <c>response_format</c>, vLLM guided decoding, forced and strict tool calls and the
/// Anthropic equivalents. The fake model follows its script while the grammar allows it and otherwise falls back to the
/// lowest allowed token, so a script that violates the format shows that the mask, not the model, guarantees the result.
/// </summary>
public class StructuredOutputTests
{
    private const string CitySchema = """{"type":"object","properties":{"city":{"type":"string","maxLength":8}},"required":["city"],"additionalProperties":false}""";

    private const string SchemaFormat = ""","response_format":{"type":"json_schema","json_schema":{"name":"c","schema":""" + CitySchema + "}}";

    /// <param name="reasoning">Use the test template with its think block (else one without, so fallbacks never open a block).</param>
    private static async Task<ApiHost> Host(bool reasoning = false)
    {
        var h = await ApiHost.Create(
            template: reasoning ? ApiHost.Template : ApiHost.Template.Replace("<think>\n\n</think>\n\n", ""),
            mo: new ModelServiceOptions { ModelName = "fake-model", DefaultMaxTokens = 300, ToolFormats = [ToolCallFormats.Hermes] });
        h.Decoder.Bias = FallbackPreference();
        return h;
    }

    /// <summary>What the fake model says when its script is masked: close structures early, avoid blanks and control bytes.</summary>
    private static float[] FallbackPreference()
    {
        var bias = new float[new FakeTokenizer().VocabSize];
        for (int t = 0; t < bias.Length; t++)
        {
            bias[t] = t switch
            {
                '{' or '}' or ']' => 5,
                '"' => 4,
                >= 'a' and <= 'z' or >= '0' and <= '9' => 3,
                ':' or '[' => 2,
                FakeTokenizer.Eos => 1,
                '<' or '/' or '>' or '_' => -1,
                ' ' or '\t' or '\n' or '\r' => -3,
                _ => -5,
            };
        }

        return bias;
    }

    private static string Chat(string extra) => $$"""{"model":"fake-model","messages":[{"role":"user","content":"hi"}],"max_tokens":300,"temperature":0{{extra}}}""";

    private static string Content(JsonElement body) => body.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString()!;

    private static void AssertCity(string json)
    {
        using var doc = JsonDocument.Parse(json);
        Assert.Equal(JsonValueKind.String, doc.RootElement.GetProperty("city").ValueKind);
        Assert.Single(doc.RootElement.EnumerateObject());
    }

    [Fact]
    public async Task JsonSchema_ValidOutputPassesThroughUnchanged()
    {
        await using var h = await Host();
        h.Decoder.Script("""{"city": "Paris"}""");
        var (status, body) = await h.PostJson("/v1/chat/completions", Chat(SchemaFormat));
        Assert.Equal(HttpStatusCode.OK, status);
        Assert.Equal("""{"city": "Paris"}""", Content(body));
    }

    [Fact]
    public async Task JsonSchema_InvalidOutputIsForcedIntoTheSchema()
    {
        await using var h = await Host();
        h.Decoder.Script("""Sure! Here it is: {"town": 42}""");
        var (status, body) = await h.PostJson("/v1/chat/completions", Chat(SchemaFormat));
        Assert.Equal(HttpStatusCode.OK, status);
        AssertCity(Content(body));
        Assert.Equal("stop", body.GetProperty("choices")[0].GetProperty("finish_reason").GetString());
    }

    [Fact]
    public async Task JsonObject_Streaming()
    {
        await using var h = await Host();
        h.Decoder.Script("""Answer: {"a": [1, true]}""");
        var events = await h.PostSse("/v1/chat/completions", Chat(""","stream":true,"response_format":{"type":"json_object"}"""));
        string text = string.Concat(events.Where(e => e.Data != "[DONE]")
            .Select(e => JsonDocument.Parse(e.Data).RootElement.GetProperty("choices")[0].GetProperty("delta"))
            .Where(d => d.TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String)
            .Select(d => d.GetProperty("content").GetString()));
        using var doc = JsonDocument.Parse(text);
        Assert.Equal(JsonValueKind.Object, doc.RootElement.ValueKind);
    }

    [Fact]
    public async Task ReasoningStaysFree_AnswerIsConstrained()
    {
        await using var h = await Host(reasoning: true);
        h.Decoder.Script("""<think>maybe {"city": 1}?</think>{"city":"Rome"}""");
        var (_, body) = await h.PostJson("/v1/chat/completions", Chat(SchemaFormat));
        var msg = body.GetProperty("choices")[0].GetProperty("message");
        Assert.Equal("""{"city":"Rome"}""", msg.GetProperty("content").GetString());
        Assert.Contains("maybe", msg.GetProperty("reasoning_content").GetString());
    }

    [Theory]
    [InlineData(""","guided_regex":"[0-9]{3}-[a-z]{2}" """, @"^[0-9]{3}-[a-z]{2}$")]
    [InlineData(""","guided_choice":["yes","no"]""", "^(yes|no)$")]
    [InlineData(""","structured_outputs":{"grammar":"root ::= \"A\" [0-9]{1,3} \".\""}""", @"^A[0-9]{1,3}\.$")]
    [InlineData(""","structured_outputs":{"regex":"x+"}""", "^x+$")]
    public async Task GuidedDecoding_VllmExtensions(string extra, string expected)
    {
        await using var h = await Host();
        h.Decoder.Script("123-ab and more");
        var (status, body) = await h.PostJson("/v1/chat/completions", Chat(extra));
        Assert.Equal(HttpStatusCode.OK, status);
        Assert.Matches(expected, Content(body));
    }

    [Fact]
    public async Task Completions_GuidedRegex()
    {
        await using var h = await Host();
        h.Decoder.Script("12345");
        var (status, body) = await h.PostJson("/v1/completions", """{"model":"fake-model","prompt":"x\u0001","max_tokens":20,"temperature":0,"guided_regex":"[0-9]{3}"}""");
        Assert.Equal(HttpStatusCode.OK, status);
        Assert.Equal("123", body.GetProperty("choices")[0].GetProperty("text").GetString());
    }

    [Theory]
    [InlineData(""","guided_regex":"(" """)]
    [InlineData(""","response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"$ref":"http://x/y"}}}""")]
    [InlineData(""","response_format":{"type":"xml"}""")]
    public async Task InvalidFormats_AreBadRequests(string extra)
    {
        await using var h = await Host();
        var (status, body) = await h.PostJson("/v1/chat/completions", Chat(extra));
        Assert.Equal(HttpStatusCode.BadRequest, status);
        Assert.Equal("invalid_request_error", body.GetProperty("error").GetProperty("type").GetString());
    }

    private const string WeatherTool = """{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string","maxLength":8}},"required":["city"]}}}""";

    [Fact]
    public async Task RequiredToolChoice_ForcesAValidCall()
    {
        await using var h = await Host();
        h.Decoder.Script("I would rather chat.");
        var (status, body) = await h.PostJson("/v1/chat/completions", Chat($$""","tools":[{{WeatherTool}}],"tool_choice":"required" """));
        Assert.Equal(HttpStatusCode.OK, status);
        var choice = body.GetProperty("choices")[0];
        Assert.Equal("tool_calls", choice.GetProperty("finish_reason").GetString());
        var fn = choice.GetProperty("message").GetProperty("tool_calls")[0].GetProperty("function");
        Assert.Equal("get_weather", fn.GetProperty("name").GetString());
        AssertCity(fn.GetProperty("arguments").GetString()!);
    }

    [Fact]
    public async Task StrictTool_TextAnswerStaysFree_CallIsConstrained()
    {
        await using var h = await Host();
        string strictTool = WeatherTool.Replace("\"name\":\"get_weather\",", "\"name\":\"get_weather\",\"strict\":true,");
        h.Decoder.Script("No call needed.");
        var (_, free) = await h.PostJson("/v1/chat/completions", Chat($$""","tools":[{{strictTool}}]"""));
        Assert.Equal("No call needed.", Content(free));

        h.Decoder.Script("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"town\": 7}}\n</tool_call>");
        var (_, called) = await h.PostJson("/v1/chat/completions", Chat($$""","tools":[{{strictTool}}]"""));
        var fn = called.GetProperty("choices")[0].GetProperty("message").GetProperty("tool_calls")[0].GetProperty("function");
        AssertCity(fn.GetProperty("arguments").GetString()!);
    }

    [Fact]
    public async Task Anthropic_ToolChoiceAnyAndOutputFormat()
    {
        await using var h = await Host();
        h.Decoder.Script("plain words");
        var (_, body) = await h.PostJson("/v1/messages", """
            {"model":"fake-model","max_tokens":300,"temperature":0,"messages":[{"role":"user","content":"hi"}],
             "tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{"city":{"type":"string","maxLength":8}},"required":["city"]}}],
             "tool_choice":{"type":"any"}}
            """);
        var use = body.GetProperty("content").EnumerateArray().Single(c => c.GetProperty("type").GetString() == "tool_use");
        Assert.Equal("get_weather", use.GetProperty("name").GetString());
        Assert.Equal(JsonValueKind.String, use.GetProperty("input").GetProperty("city").ValueKind);

        var (_, formatted) = await h.PostJson("/v1/messages",
            """{"model":"fake-model","max_tokens":300,"temperature":0,"messages":[{"role":"user","content":"hi"}],"output_format":{"type":"json_schema","schema":""" + CitySchema + "}}");
        var text = formatted.GetProperty("content").EnumerateArray().Single(c => c.GetProperty("type").GetString() == "text");
        AssertCity(text.GetProperty("text").GetString()!);
    }

    [Fact]
    public void LongStrings_KeepTheStateSpaceFlat()
    {
        // Right recursion is executed as a tail call: a 5000-character string revisits the same few states.
        var rt = new GrammarRuntime(Grammar.JsonObject);
        var state = rt.Advance(rt.Initial, "{\"text\": \""u8);
        for (int i = 0; i < 5000; i++)
        {
            state = rt.Advance(state, (byte)('a' + (i % 26)));
        }

        state = rt.Advance(state, "\"}"u8);
        Assert.True(state.Accepting);
        Assert.True(rt.StateCount < 60, $"{rt.StateCount} states");
    }
}
