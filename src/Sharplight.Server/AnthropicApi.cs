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

/// <summary>Anthropic-compatible endpoints: <c>/v1/messages</c> (also streaming) and <c>/v1/messages/count_tokens</c>.</summary>
public static class AnthropicApi
{
    public static IEndpointRouteBuilder MapAnthropic(this IEndpointRouteBuilder routes)
    {
        var ctx = routes.ServiceProvider.GetRequiredService<SharplightServerContext>();
        routes.MapModels(ctx);
        routes.MapPost("/v1/messages", http => Messages(http, ctx));
        routes.MapPost("/v1/messages/count_tokens", http => CountTokens(http, ctx));
        return routes;
    }

    internal static JsonNode ModelObject(ModelInfo m)
        => new JsonObject { ["type"] = "model", ["id"] = m.Name, ["display_name"] = m.Name, ["created_at"] = "2025-01-01T00:00:00Z" };

    internal static JsonNode ModelList(SharplightServerContext ctx)
    {
        var l = ctx.Host.List();
        return new JsonObject { ["data"] = new JsonArray(l.Select(m => ModelObject(m)).ToArray()), ["has_more"] = false, ["first_id"] = l.FirstOrDefault()?.Name, ["last_id"] = l.LastOrDefault()?.Name };
    }

    // ---- request parsing --------------------------------------------------------------------------------------------

    private static string TextOf(JsonElement? e)
    {
        if (e is not { } v)
        {
            return "";
        }

        if (v.ValueKind == JsonValueKind.String)
        {
            return v.GetString()!;
        }

        if (v.ValueKind != JsonValueKind.Array)
        {
            return "";
        }

        return string.Concat(v.EnumerateArray().Where(x => x.Str("type") == "text").Select(x => x.Str("text")));
    }

    private static ImageData ParseImage(JsonElement block)
    {
        var src = block.Obj("source") ?? throw ApiException.BadRequest("image block needs 'source'.");
        return src.Str("type") switch
        {
            "base64" => new ImageData(DecodeBase64(src.Str("data")), src.Str("media_type"), null),
            "url" => new ImageData(null, null, src.Str("url")),
            var t => throw ApiException.BadRequest($"Unsupported image source type '{t}'."),
        };
    }

    private static byte[] DecodeBase64(string? s)
    {
        try { return Convert.FromBase64String(s ?? ""); }
        catch (FormatException) { throw ApiException.BadRequest("Invalid base64 image data."); }
    }

    private static void AddMessage(List<ChatTurn> turns, JsonElement m, int i)
    {
        string role = m.Str("role") ?? throw ApiException.BadRequest($"messages.{i}.role: Field required");
        if (role is not ("user" or "assistant"))
        {
            throw ApiException.BadRequest($"messages.{i}.role: Input should be 'user' or 'assistant'");
        }

        var content = m.Get("content") ?? throw ApiException.BadRequest($"messages.{i}.content: Field required");
        if (content.ValueKind == JsonValueKind.String) { turns.Add(new ChatTurn(role, content.GetString())); return; }
        if (content.ValueKind != JsonValueKind.Array)
        {
            throw ApiException.BadRequest($"messages.{i}.content: Input should be a string or an array");
        }

        var parts = new List<ChatPart>();
        var calls = new List<ToolCall>();
        var reasoning = new System.Text.StringBuilder();
        var toolResults = new List<ChatTurn>();
        foreach (var blk in content.EnumerateArray())
        {
            switch (blk.Str("type"))
            {
                case "text": parts.Add(ChatPart.FromText(blk.Str("text") ?? "")); break;
                case "image": parts.Add(ChatPart.FromImage(ParseImage(blk))); break;
                case "thinking": reasoning.Append(blk.Str("thinking")); break;
                case "redacted_thinking": break;
                case "tool_use":
                    calls.Add(new ToolCall(blk.Str("id") ?? "toolu_" + Guid.NewGuid().ToString("N")[..24], blk.Str("name") ?? "", blk.Get("input")?.GetRawText() ?? "{}"));
                    break;
                case "tool_result":
                    string text = TextOf(blk.Get("content"));
                    if (blk.Bool("is_error") == true)
                    {
                        text = "Error: " + text;
                    }

                    toolResults.Add(new ChatTurn("tool", text, ToolCallId: blk.Str("tool_use_id")));
                    break;
                case var t: throw ApiException.BadRequest($"messages.{i}.content: unsupported block type '{t}'");
            }
        }
        // Tool results (always leading in a user turn) become tool messages, the remaining blocks one user/assistant message.
        turns.AddRange(toolResults);
        if (parts.Count > 0 || calls.Count > 0 || reasoning.Length > 0)
        {
            turns.Add(new ChatTurn(role, null, parts.Count > 0 ? parts : null, reasoning.Length > 0 ? reasoning.ToString() : null, calls.Count > 0 ? calls : null));
        }
    }

    private static ChatRequest ParseRequest(JsonElement b, SharplightServerContext ctx, bool requireMaxTokens)
    {
        int maxTokens = b.Int("max_tokens") ?? 0;
        if (requireMaxTokens)
        {
            if (b.Get("max_tokens") is null)
            {
                throw ApiException.BadRequest("max_tokens: Field required", "max_tokens");
            }

            if (maxTokens < 1)
            {
                throw ApiException.BadRequest("max_tokens: Input should be greater than or equal to 1", "max_tokens");
            }
        }
        var turns = new List<ChatTurn>();
        if (b.Get("system") is { } sys && TextOf(sys) is { Length: > 0 } sysText)
        {
            turns.Add(new ChatTurn("system", sysText));
        }

        int i = 0, count = 0;
        foreach (var m in b.Arr("messages")) { AddMessage(turns, m, i++); count++; }
        if (count == 0)
        {
            throw ApiException.BadRequest("messages: Field required", "messages");
        }

        var tools = b.Arr("tools").Where(t => t.Str("type") is null or "custom")
            .Select(t => new ToolSpec(t.Str("name") ?? throw ApiException.BadRequest("tools.name: Field required"), t.Str("description"), t.Obj("input_schema")?.GetRawText())).ToList();
        var mode = ToolChoiceMode.Auto; string? fname = null;
        if (b.Obj("tool_choice") is { } tc)
        {
            switch (tc.Str("type"))
            {
                case "any": mode = ToolChoiceMode.Required; break;
                case "none": mode = ToolChoiceMode.None; break;
                case "tool": mode = ToolChoiceMode.Function; fname = tc.Str("name"); break;
            }
        }

        bool? thinking = b.Obj("thinking")?.Str("type") switch { "enabled" => true, "disabled" => false, _ => null };
        var sampling = ctx.Options.DefaultSampling with
        {
            Temperature = b.Float("temperature") ?? ctx.Options.DefaultSampling.Temperature,
            TopP = b.Float("top_p") ?? ctx.Options.DefaultSampling.TopP,
            TopK = b.Int("top_k") ?? ctx.Options.DefaultSampling.TopK,
        };
        return new ChatRequest
        {
            Messages = turns, Tools = tools, ToolChoice = mode, ToolChoiceFunction = fname, Sampling = sampling, MaxTokens = maxTokens,
            StopStrings = [.. b.Arr("stop_sequences").Select(s => s.GetString() ?? "")], EnableThinking = thinking, Timeout = ctx.Options.RequestTimeout,
        };
    }

    // ---- responses --------------------------------------------------------------------------------------------------

    private static string ToolUseId(string id) => id.StartsWith("call_", StringComparison.Ordinal) ? "toolu_" + id[5..] : id;

    private static string StopReason(ChatFinishReason r, string? stopSeq) => r switch
    {
        ChatFinishReason.Length => "max_tokens", ChatFinishReason.ToolCalls => "tool_use",
        _ => stopSeq is not null ? "stop_sequence" : "end_turn",
    };

    private static JsonObject UsageJson(GenerationUsage u) => new()
    {
        ["input_tokens"] = u.PromptTokens, ["cache_creation_input_tokens"] = 0, ["cache_read_input_tokens"] = u.CachedTokens, ["output_tokens"] = u.CompletionTokens,
    };

    private static JsonNode ParseInput(string json)
    {
        try { return JsonNode.Parse(json) ?? new JsonObject(); } catch (JsonException) { return new JsonObject(); }
    }

    private static async Task CountTokens(HttpContext http, SharplightServerContext ctx)
    {
        using var doc = await HttpJson.ReadBodyAsync(http, ctx.Options.MaxRequestBodyBytes);
        var req = ParseRequest(doc.RootElement, ctx, requireMaxTokens: false);
        using var lease = await ctx.AcquireAsync(http, doc.RootElement.Str("model"));
        await HttpJson.WriteAsync(http, new JsonObject { ["input_tokens"] = SharplightServerContext.ChatOf(lease).CountTokens(req) });
    }

    private static async Task Messages(HttpContext http, SharplightServerContext ctx)
    {
        using var doc = await HttpJson.ReadBodyAsync(http, ctx.Options.MaxRequestBodyBytes);
        var b = doc.RootElement;
        var req = ParseRequest(b, ctx, requireMaxTokens: true);
        bool stream = b.Bool("stream") ?? false;
        using var lease = await ctx.AcquireAsync(http, b.Str("model"));   // held until the response is complete
        var run = SharplightServerContext.ChatOf(lease).StartChat(req, http.RequestAborted);
        string id = "msg_" + Guid.NewGuid().ToString("N")[..24];
        string model = lease.Name;

        if (!stream)
        {
            var res = await run.CompleteAsync(http.RequestAborted);
            var c = res.Choices[0];
            if (c.Finish == ChatFinishReason.Timeout) { await ApiErrors.WriteAsync(http, 504, "api_error", c.Error ?? "The request timed out."); return; }
            if (c.Finish == ChatFinishReason.Error) { await ApiErrors.WriteAsync(http, 500, "api_error", c.Error ?? "Generation failed."); return; }
            if (c.Finish == ChatFinishReason.Cancelled) { await ApiErrors.WriteAsync(http, 529, "overloaded_error", "The request was cancelled because the server is shutting down."); return; }
            var content = new JsonArray();
            if (c.Reasoning is not null)
            {
                content.Add(new JsonObject { ["type"] = "thinking", ["thinking"] = c.Reasoning, ["signature"] = "" });
            }

            if (c.Content.Length > 0 || c.ToolCalls.Count == 0)
            {
                content.Add(new JsonObject { ["type"] = "text", ["text"] = c.Content });
            }

            foreach (var t in c.ToolCalls)
            {
                content.Add(new JsonObject { ["type"] = "tool_use", ["id"] = ToolUseId(t.Id), ["name"] = t.Name, ["input"] = ParseInput(t.ArgumentsJson) });
            }

            await HttpJson.WriteAsync(http, new JsonObject
            {
                ["id"] = id, ["type"] = "message", ["role"] = "assistant", ["model"] = model, ["content"] = content,
                ["stop_reason"] = StopReason(c.Finish, c.StopSequence), ["stop_sequence"] = c.StopSequence, ["usage"] = UsageJson(res.Usage),
            });
            return;
        }

        var sse = new SseWriter(http);
        await sse.EventAsync("message_start", new JsonObject
        {
            ["type"] = "message_start",
            ["message"] = new JsonObject
            {
                ["id"] = id, ["type"] = "message", ["role"] = "assistant", ["model"] = model, ["content"] = new JsonArray(),
                ["stop_reason"] = null, ["stop_sequence"] = null,
                ["usage"] = new JsonObject { ["input_tokens"] = run.PromptTokens, ["output_tokens"] = 1 },
            },
        });
        await sse.EventAsync("ping", new JsonObject { ["type"] = "ping" });

        int index = -1; string? open = null;   // open block type: text | thinking
        async Task Close()
        {
            if (open is null)
            {
                return;
            }

            if (open == "thinking")
            {
                await sse.EventAsync("content_block_delta", new JsonObject { ["type"] = "content_block_delta", ["index"] = index, ["delta"] = new JsonObject { ["type"] = "signature_delta", ["signature"] = "" } });
            }

            await sse.EventAsync("content_block_stop", new JsonObject { ["type"] = "content_block_stop", ["index"] = index });
            open = null;
        }
        async Task Open(string type)
        {
            await Close();
            index++; open = type;
            await sse.EventAsync("content_block_start", new JsonObject
            {
                ["type"] = "content_block_start", ["index"] = index,
                ["content_block"] = type == "text" ? new JsonObject { ["type"] = "text", ["text"] = "" } : new JsonObject { ["type"] = "thinking", ["thinking"] = "" },
            });
        }
        Task Delta(string kind, string field, string value) => sse.EventAsync("content_block_delta", new JsonObject
        {
            ["type"] = "content_block_delta", ["index"] = index, ["delta"] = new JsonObject { ["type"] = kind, [field] = value },
        });

        await foreach (var e in run.ReadAllAsync(http.RequestAborted))
        {
            if (e.Content is not null)
            {
                if (open != "text")
                {
                    await Open("text");
                }

                await Delta("text_delta", "text", e.Content);
            }
            else if (e.Reasoning is not null)
            {
                if (open != "thinking")
                {
                    await Open("thinking");
                }

                await Delta("thinking_delta", "thinking", e.Reasoning);
            }
            else if (e.ToolCall is { } t)
            {
                await Close();
                index++;
                await sse.EventAsync("content_block_start", new JsonObject
                {
                    ["type"] = "content_block_start", ["index"] = index,
                    ["content_block"] = new JsonObject { ["type"] = "tool_use", ["id"] = ToolUseId(t.Id), ["name"] = t.Name, ["input"] = new JsonObject() },
                });
                await Delta("input_json_delta", "partial_json", t.ArgumentsJson);
                await sse.EventAsync("content_block_stop", new JsonObject { ["type"] = "content_block_stop", ["index"] = index });
            }
            else if (e.Finish is { } f)
            {
                if (f is ChatFinishReason.Error or ChatFinishReason.Timeout or ChatFinishReason.Cancelled)
                {
                    await sse.EventAsync("error", new JsonObject
                    {
                        ["type"] = "error",
                        ["error"] = new JsonObject { ["type"] = f == ChatFinishReason.Cancelled ? "overloaded_error" : "api_error", ["message"] = e.Error ?? "generation " + f.ToString().ToLowerInvariant() },
                    });
                    return;
                }
                if (index < 0 && open is null)
                {
                    await Open("text");
                }

                await Close();
                var u = e.Usage ?? default;
                await sse.EventAsync("message_delta", new JsonObject
                {
                    ["type"] = "message_delta",
                    ["delta"] = new JsonObject { ["stop_reason"] = StopReason(f, e.StopSequence), ["stop_sequence"] = e.StopSequence },
                    ["usage"] = new JsonObject { ["input_tokens"] = u.PromptTokens, ["cache_read_input_tokens"] = u.CachedTokens, ["output_tokens"] = u.CompletionTokens },
                });
                await sse.EventAsync("message_stop", new JsonObject { ["type"] = "message_stop" });
            }
        }
    }
}
