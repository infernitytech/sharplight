using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Server;

/// <summary>OpenAI-compatible endpoints: <c>/v1/chat/completions</c>, <c>/v1/completions</c>, <c>/v1/models</c>.</summary>
public static class OpenAiApi
{
    private static readonly long Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

    public static IEndpointRouteBuilder MapOpenAi(this IEndpointRouteBuilder routes)
    {
        var ctx = routes.ServiceProvider.GetRequiredService<SharplightServerContext>();
        routes.MapModels(ctx);
        routes.MapPost("/v1/chat/completions", http => ChatCompletions(http, ctx));
        routes.MapPost("/v1/completions", http => Completions(http, ctx));
        return routes;
    }

    internal static JsonNode ModelObject(ModelInfo m) => new JsonObject
    {
        ["id"] = m.Name, ["object"] = "model", ["created"] = Created, ["owned_by"] = "sharplight",
        // Extension: loaded or not, and what kind (models are loaded on first use).
        ["sharplight"] = new JsonObject { ["kind"] = m.Kind.ToString().ToLowerInvariant(), ["state"] = m.State.ToString().ToLowerInvariant(), ["resident_bytes"] = m.ResidentBytes },
    };

    internal static JsonNode ModelList(SharplightServerContext ctx)
        => new JsonObject { ["object"] = "list", ["data"] = new JsonArray(ctx.Host.List().Select(m => ModelObject(m)).ToArray()) };

    // ---- request parsing --------------------------------------------------------------------------------------------

    internal static SamplingParams ParseSampling(JsonElement b, SamplingParams d) => d with
    {
        Temperature = b.Float("temperature") ?? d.Temperature,
        TopP = b.Float("top_p") ?? d.TopP,
        TopK = b.Int("top_k") ?? d.TopK,
        MinP = b.Float("min_p") ?? d.MinP,
        RepetitionPenalty = b.Float("repeat_penalty") ?? b.Float("repetition_penalty") ?? d.RepetitionPenalty,
        PresencePenalty = b.Float("presence_penalty") ?? d.PresencePenalty,
        FrequencyPenalty = b.Float("frequency_penalty") ?? d.FrequencyPenalty,
        Seed = b.Int("seed") ?? d.Seed,
        LogitBias = ParseLogitBias(b) ?? d.LogitBias,
    };

    private static Dictionary<int, float>? ParseLogitBias(JsonElement b)
    {
        if (b.Obj("logit_bias") is not { } lb)
        {
            return null;
        }

        var d = new Dictionary<int, float>();
        foreach (var p in lb.EnumerateObject())
        {
            if (!int.TryParse(p.Name, out var id) || p.Value.ValueKind != JsonValueKind.Number)
            {
                throw ApiException.BadRequest("'logit_bias' must map token ids to numbers.", "logit_bias");
            }

            d[id] = Math.Clamp(p.Value.GetSingle(), -100f, 100f);
        }
        return d;
    }

    private static List<string> ParseStop(JsonElement b)
    {
        var res = new List<string>();
        if (b.Get("stop") is not { } s)
        {
            return res;
        }

        if (s.ValueKind == JsonValueKind.String)
        {
            res.Add(s.GetString()!);
        }
        else if (s.ValueKind == JsonValueKind.Array)
        {
            foreach (var e in s.EnumerateArray())
            {
                res.Add(e.ValueKind == JsonValueKind.String ? e.GetString()! : throw ApiException.BadRequest("'stop' must be a string or an array of strings.", "stop"));
            }
        }
        else
        {
            throw ApiException.BadRequest("'stop' must be a string or an array of strings.", "stop");
        }

        return res;
    }

    private static int ParseN(JsonElement b, SharplightServerOptions o)
    {
        int n = b.Int("n") ?? 1;
        if (n < 1 || n > o.MaxChoices)
        {
            throw ApiException.BadRequest($"'n' must be between 1 and {o.MaxChoices}.", "n");
        }

        return n;
    }

    internal static ImageData? ParseImageUrl(string url)
    {
        if (url.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
        {
            int comma = url.IndexOf(',');
            if (comma < 0)
            {
                throw ApiException.BadRequest("Malformed data URL.");
            }

            string meta = url[5..comma];
            try { return new ImageData(Convert.FromBase64String(url[(comma + 1)..]), meta.Split(';')[0], null); }
            catch (FormatException) { throw ApiException.BadRequest("Invalid base64 image data."); }
        }
        return new ImageData(null, null, url);
    }

    private static ChatTurn ParseMessage(JsonElement m, int i)
    {
        string role = m.Str("role") ?? throw ApiException.BadRequest($"messages[{i}].role is required.", $"messages[{i}].role");
        if (role == "developer")
        {
            role = "system";
        }

        if (role is not ("system" or "user" or "assistant" or "tool" or "function"))
        {
            throw ApiException.BadRequest($"Invalid role '{role}' in messages[{i}].", $"messages[{i}].role");
        }

        string? text = null;
        List<ChatPart>? parts = null;
        if (m.Get("content") is { } c)
        {
            if (c.ValueKind == JsonValueKind.String)
            {
                text = c.GetString();
            }
            else if (c.ValueKind == JsonValueKind.Array)
            {
                parts = [];
                foreach (var p in c.EnumerateArray())
                {
                    switch (p.Str("type"))
                    {
                        case "text" or "input_text": parts.Add(ChatPart.FromText(p.Str("text") ?? "")); break;
                        case "image_url":
                            var iu = p.Get("image_url");
                            string? url = iu is { ValueKind: JsonValueKind.String } ? iu.Value.GetString() : iu?.Str("url");
                            parts.Add(ChatPart.FromImage(ParseImageUrl(url ?? throw ApiException.BadRequest("image_url.url is required.")) !));
                            break;
                        case var t: throw ApiException.BadRequest($"Unsupported content part type '{t}'.", $"messages[{i}].content");
                    }
                }
            }
            else
            {
                throw ApiException.BadRequest($"messages[{i}].content must be a string or an array.", $"messages[{i}].content");
            }
        }
        List<ToolCall>? calls = null;
        foreach (var tc in m.Arr("tool_calls"))
        {
            var fn = tc.Obj("function") ?? throw ApiException.BadRequest("tool_calls[].function is required.");
            var args = fn.Get("arguments");
            string argText = args is { ValueKind: JsonValueKind.String } ? args.Value.GetString()! : args?.GetRawText() ?? "{}";
            (calls ??= []).Add(new ToolCall(tc.Str("id") ?? "call_" + Guid.NewGuid().ToString("N")[..24], fn.Str("name") ?? "", argText));
        }
        return new ChatTurn(role == "function" ? "tool" : role, text, parts, m.Str("reasoning_content") ?? m.Str("reasoning"), calls, m.Str("tool_call_id"), m.Str("name"));
    }

    internal static ToolSpec ParseTool(JsonElement t)
    {
        var fn = t.Obj("function") ?? t;
        return new ToolSpec(fn.Str("name") ?? throw ApiException.BadRequest("tools[].function.name is required.", "tools"),
            fn.Str("description"), fn.Obj("parameters")?.GetRawText());
    }

    private static ChatRequest ParseChat(JsonElement b, SharplightServerContext ctx)
    {
        var msgs = new List<ChatTurn>();
        int idx = 0;
        foreach (var m in b.Arr("messages"))
        {
            msgs.Add(ParseMessage(m, idx++));
        }

        if (msgs.Count == 0)
        {
            throw ApiException.BadRequest("'messages' must be a non-empty array.", "messages");
        }

        var tools = b.Arr("tools").Select(ParseTool).ToList();
        var mode = ToolChoiceMode.Auto; string? fname = null;
        if (b.Get("tool_choice") is { } tc)
        {
            if (tc.ValueKind == JsonValueKind.String)
            {
                mode = tc.GetString() switch { "none" => ToolChoiceMode.None, "required" => ToolChoiceMode.Required, "auto" => ToolChoiceMode.Auto, var s => throw ApiException.BadRequest($"Invalid tool_choice '{s}'.", "tool_choice") };
            }
            else { mode = ToolChoiceMode.Function; fname = tc.Obj("function")?.Str("name") ?? tc.Str("name"); }
        }
        Dictionary<string, object?>? kwargs = null;
        if (b.Obj("chat_template_kwargs") is { } kw)
        {
            kwargs = [];
            foreach (var kp in kw.EnumerateObject())
            {
                kwargs[kp.Name] = JsonClr.Convert(kp.Value);
            }
        }
        bool? thinking = kwargs is not null && kwargs.TryGetValue("enable_thinking", out var et) && et is bool eb ? eb : null;
        if (b.Str("reasoning_effort") == "none")
        {
            thinking = false;
        }

        string? rf = b.Obj("response_format")?.Str("type");
        int top = b.Int("top_logprobs") ?? 0;
        return new ChatRequest
        {
            Messages = msgs, Tools = tools, ToolChoice = mode, ToolChoiceFunction = fname,
            Sampling = ParseSampling(b, ctx.Options.DefaultSampling),
            MaxTokens = b.Int("max_completion_tokens") ?? b.Int("max_tokens") ?? 0,
            StopStrings = ParseStop(b), N = ParseN(b, ctx.Options),
            Priority = b.Int("priority") ?? 0, Timeout = ctx.Options.RequestTimeout,
            EnableThinking = thinking, JsonMode = rf is "json_object" or "json_schema",
            Logprobs = b.Bool("logprobs") ?? false, TopLogprobs = Math.Clamp(top, 0, 20),
            TemplateContext = kwargs,
        };
    }

    // ---- chat completions -------------------------------------------------------------------------------------------

    private static string FinishName(ChatFinishReason r) => r switch
    {
        ChatFinishReason.Length => "length", ChatFinishReason.ToolCalls => "tool_calls", _ => "stop",
    };

    private static JsonObject UsageJson(GenerationUsage u) => new()
    {
        ["prompt_tokens"] = u.PromptTokens, ["completion_tokens"] = u.CompletionTokens, ["total_tokens"] = u.TotalTokens,
        ["prompt_tokens_details"] = new JsonObject { ["cached_tokens"] = u.CachedTokens },
    };

    private static JsonObject ToolCallJson(ToolCall c, int? index = null)
    {
        var o = new JsonObject();
        if (index is int i)
        {
            o["index"] = i;
        }

        o["id"] = c.Id; o["type"] = "function";
        o["function"] = new JsonObject { ["name"] = c.Name, ["arguments"] = c.ArgumentsJson };
        return o;
    }

    private static JsonObject ChatLogprobs(IEnumerable<TokenLogprob> lps)
    {
        var arr = new JsonArray();
        foreach (var lp in lps)
        {
            arr.Add(new JsonObject
            {
                ["token"] = lp.Token, ["logprob"] = lp.Logprob, ["bytes"] = Bytes(lp.Token),
                ["top_logprobs"] = new JsonArray(lp.TopAlternatives.Select(a => (JsonNode)new JsonObject { ["token"] = a.Token, ["logprob"] = a.Logprob, ["bytes"] = Bytes(a.Token) }).ToArray()),
            });
        }

        return new JsonObject { ["content"] = arr };
    }

    private static JsonArray Bytes(string s) => new(Encoding.UTF8.GetBytes(s).Select(x => (JsonNode)(int)x).ToArray());

    private static async Task ChatCompletions(HttpContext http, SharplightServerContext ctx)
    {
        using var doc = await HttpJson.ReadBodyAsync(http, ctx.Options.MaxRequestBodyBytes);
        var b = doc.RootElement;
        var req = ParseChat(b, ctx);
        bool stream = b.Bool("stream") ?? false;
        bool includeUsage = b.Obj("stream_options")?.Bool("include_usage") ?? false;
        using var lease = await ctx.AcquireAsync(http, b.Str("model"));   // held until the response is complete
        var run = SharplightServerContext.ChatOf(lease).StartChat(req, http.RequestAborted);
        string id = "chatcmpl-" + Guid.NewGuid().ToString("N")[..24];
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        string model = lease.Name;

        if (!stream)
        {
            var res = await run.CompleteAsync(http.RequestAborted);
            if (FailureOf(res.Choices) is { } fail) { await ApiErrors.WriteAsync(http, fail.Status, fail.Type, fail.Message); return; }
            var choices = new JsonArray();
            foreach (var c in res.Choices)
            {
                var msg = new JsonObject { ["role"] = "assistant", ["content"] = c.Content.Length == 0 && c.ToolCalls.Count > 0 ? null : c.Content };
                if (c.Reasoning is not null)
                {
                    msg["reasoning_content"] = c.Reasoning;
                }

                if (c.ToolCalls.Count > 0)
                {
                    msg["tool_calls"] = new JsonArray(c.ToolCalls.Select(t => (JsonNode)ToolCallJson(t)).ToArray());
                }

                choices.Add(new JsonObject
                {
                    ["index"] = c.Index, ["message"] = msg, ["logprobs"] = req.Logprobs ? ChatLogprobs(c.Logprobs) : null, ["finish_reason"] = FinishName(c.Finish),
                });
            }
            await HttpJson.WriteAsync(http, new JsonObject
            {
                ["id"] = id, ["object"] = "chat.completion", ["created"] = created, ["model"] = model, ["choices"] = choices, ["usage"] = UsageJson(res.Usage),
            });
            return;
        }

        var sse = new SseWriter(http);
        JsonObject Chunk(int index, JsonObject delta, string? finish = null, JsonNode? logprobs = null)
        {
            var o = new JsonObject
            {
                ["id"] = id, ["object"] = "chat.completion.chunk", ["created"] = created, ["model"] = model,
                ["choices"] = new JsonArray(new JsonObject { ["index"] = index, ["delta"] = delta, ["logprobs"] = logprobs, ["finish_reason"] = finish }),
            };
            if (includeUsage)
            {
                o["usage"] = null;
            }

            return o;
        }
        for (int i = 0; i < req.N; i++)
        {
            await sse.DataAsync(Chunk(i, new JsonObject { ["role"] = "assistant", ["content"] = "" }));
        }

        var toolIndex = new Dictionary<int, int>();
        int finished = 0, completion = 0; GenerationUsage first = default;
        await foreach (var e in run.ReadAllAsync(http.RequestAborted))
        {
            if (e.Content is not null)
            {
                await sse.DataAsync(Chunk(e.Index, new JsonObject { ["content"] = e.Content }));
            }
            else if (e.Reasoning is not null)
            {
                await sse.DataAsync(Chunk(e.Index, new JsonObject { ["reasoning_content"] = e.Reasoning }));
            }
            else if (e.ToolCall is not null)
            {
                int k = toolIndex.GetValueOrDefault(e.Index);
                toolIndex[e.Index] = k + 1;
                await sse.DataAsync(Chunk(e.Index, new JsonObject { ["tool_calls"] = new JsonArray(ToolCallJson(e.ToolCall, k)) }));
            }
            else if (e.Logprobs is not null && req.Logprobs)
            {
                await sse.DataAsync(Chunk(e.Index, new JsonObject(), logprobs: ChatLogprobs(e.Logprobs)));
            }
            else if (e.Finish is { } f)
            {
                if (f is ChatFinishReason.Error or ChatFinishReason.Timeout)
                {
                    await sse.DataAsync(ApiErrors.Body(false, f == ChatFinishReason.Timeout ? "timeout" : "server_error", e.Error ?? "generation failed"));
                    return;
                }
                await sse.DataAsync(Chunk(e.Index, new JsonObject(), FinishName(f)));
                completion += e.Usage?.CompletionTokens ?? 0;
                if (finished++ == 0)
                {
                    first = e.Usage ?? default;
                }
            }
        }
        if (includeUsage)
        {
            await sse.DataAsync(new JsonObject
            {
                ["id"] = id, ["object"] = "chat.completion.chunk", ["created"] = created, ["model"] = model, ["choices"] = new JsonArray(),
                ["usage"] = UsageJson(new GenerationUsage(first.PromptTokens, completion, first.CachedTokens)),
            });
        }

        await sse.RawAsync("data: [DONE]\n\n");
    }

    private static (int Status, string Type, string Message)? FailureOf(IReadOnlyList<ChatChoice> choices)
    {
        foreach (var c in choices)
        {
            if (c.Finish == ChatFinishReason.Timeout)
            {
                return (504, "timeout", c.Error ?? "The request timed out.");
            }

            if (c.Finish == ChatFinishReason.Error)
            {
                return (500, "server_error", c.Error ?? "Generation failed.");
            }

            if (c.Finish == ChatFinishReason.Cancelled)
            {
                return (503, "server_error", "The request was cancelled because the server is shutting down.");
            }
        }
        return null;
    }

    // ---- legacy completions -----------------------------------------------------------------------------------------

    private static JsonObject TextLogprobs(IEnumerable<TokenLogprob> lps)
    {
        var tokens = new JsonArray(); var lp = new JsonArray(); var top = new JsonArray(); var offs = new JsonArray();
        int off = 0;
        foreach (var t in lps)
        {
            tokens.Add(t.Token); lp.Add(t.Logprob); offs.Add(off); off += t.Token.Length;
            var d = new JsonObject();
            foreach (var a in t.TopAlternatives)
            {
                d[a.Token] = a.Logprob;
            }

            top.Add(d);
        }
        return new JsonObject { ["tokens"] = tokens, ["token_logprobs"] = lp, ["top_logprobs"] = top, ["text_offset"] = offs };
    }

    private static async Task Completions(HttpContext http, SharplightServerContext ctx)
    {
        using var doc = await HttpJson.ReadBodyAsync(http, ctx.Options.MaxRequestBodyBytes);
        var b = doc.RootElement;
        string? prompt = null; int[]? tokens = null;
        var p = b.Get("prompt") ?? throw ApiException.BadRequest("'prompt' is required.", "prompt");
        if (p.ValueKind == JsonValueKind.String)
        {
            prompt = p.GetString();
        }
        else if (p.ValueKind == JsonValueKind.Array)
        {
            var items = p.EnumerateArray().ToList();
            if (items.Count > 0 && items.All(x => x.ValueKind == JsonValueKind.Number))
            {
                tokens = [.. items.Select(x => x.GetInt32())];
            }
            else if (items.Count == 1 && items[0].ValueKind == JsonValueKind.String)
            {
                prompt = items[0].GetString();
            }
            else
            {
                throw ApiException.BadRequest("'prompt' must be a string, a list of token ids or a single-element list of strings (batched prompts are not supported).", "prompt");
            }
        }
        else
        {
            throw ApiException.BadRequest("'prompt' must be a string or an array.", "prompt");
        }

        int? lpCount = b.Int("logprobs");
        var req = new CompletionRequest
        {
            Prompt = prompt, PromptTokens = tokens, Sampling = ParseSampling(b, ctx.Options.DefaultSampling),
            MaxTokens = b.Int("max_tokens") ?? 16, StopStrings = ParseStop(b), N = ParseN(b, ctx.Options),
            Priority = b.Int("priority") ?? 0, Timeout = ctx.Options.RequestTimeout,
            Logprobs = lpCount is > 0, TopLogprobs = Math.Clamp(lpCount ?? 0, 0, 20),
        };
        bool stream = b.Bool("stream") ?? false, echo = (b.Bool("echo") ?? false) && prompt is not null;
        bool includeUsage = b.Obj("stream_options")?.Bool("include_usage") ?? false;
        using var lease = await ctx.AcquireAsync(http, b.Str("model"));
        var run = SharplightServerContext.ChatOf(lease).StartCompletion(req, http.RequestAborted);
        string id = "cmpl-" + Guid.NewGuid().ToString("N")[..24];
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        string model = lease.Name;

        if (!stream)
        {
            var res = await run.CompleteAsync(http.RequestAborted);
            if (FailureOf(res.Choices) is { } fail) { await ApiErrors.WriteAsync(http, fail.Status, fail.Type, fail.Message); return; }
            var choices = new JsonArray(res.Choices.Select(c => (JsonNode)new JsonObject
            {
                ["text"] = (echo ? prompt : "") + c.Content, ["index"] = c.Index,
                ["logprobs"] = req.Logprobs ? TextLogprobs(c.Logprobs) : null, ["finish_reason"] = FinishName(c.Finish),
            }).ToArray());
            await HttpJson.WriteAsync(http, new JsonObject
            {
                ["id"] = id, ["object"] = "text_completion", ["created"] = created, ["model"] = model, ["choices"] = choices, ["usage"] = UsageJson(res.Usage),
            });
            return;
        }

        var sse = new SseWriter(http);
        JsonObject Chunk(int index, string text, string? finish = null, JsonNode? logprobs = null)
        {
            var o = new JsonObject
            {
                ["id"] = id, ["object"] = "text_completion", ["created"] = created, ["model"] = model,
                ["choices"] = new JsonArray(new JsonObject { ["text"] = text, ["index"] = index, ["logprobs"] = logprobs, ["finish_reason"] = finish }),
            };
            if (includeUsage)
            {
                o["usage"] = null;
            }

            return o;
        }
        if (echo)
        {
            for (int i = 0; i < req.N; i++)
            {
                await sse.DataAsync(Chunk(i, prompt!));
            }
        }

        int completion = 0; GenerationUsage first = default; bool any = false;
        await foreach (var e in run.ReadAllAsync(http.RequestAborted))
        {
            if (e.Content is not null)
            {
                await sse.DataAsync(Chunk(e.Index, e.Content));
            }
            else if (e.Logprobs is not null && req.Logprobs)
            {
                await sse.DataAsync(Chunk(e.Index, "", logprobs: TextLogprobs(e.Logprobs)));
            }
            else if (e.Finish is { } f)
            {
                if (f is ChatFinishReason.Error or ChatFinishReason.Timeout)
                {
                    await sse.DataAsync(ApiErrors.Body(false, f == ChatFinishReason.Timeout ? "timeout" : "server_error", e.Error ?? "generation failed"));
                    return;
                }
                await sse.DataAsync(Chunk(e.Index, "", FinishName(f)));
                completion += e.Usage?.CompletionTokens ?? 0;
                if (!any) { first = e.Usage ?? default; any = true; }
            }
        }
        if (includeUsage)
        {
            await sse.DataAsync(new JsonObject
            {
                ["id"] = id, ["object"] = "text_completion", ["created"] = created, ["model"] = model, ["choices"] = new JsonArray(),
                ["usage"] = UsageJson(new GenerationUsage(first.PromptTokens, completion, first.CachedTokens)),
            });
        }

        await sse.RawAsync("data: [DONE]\n\n");
    }
}
