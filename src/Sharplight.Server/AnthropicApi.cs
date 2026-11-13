using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Llm.Structured;

namespace Sharplight.Server;

/// <summary>Anthropic-compatible endpoints: <c>/v1/messages</c> (also streaming) and <c>/v1/messages/count_tokens</c>.</summary>
internal static class AnthropicApi
{
    extension(IEndpointRouteBuilder routes)
    {
        /// <summary>Maps <c>/v1/messages</c> and <c>/v1/messages/count_tokens</c> (and <c>/v1/models</c>).</summary>
        public IEndpointRouteBuilder MapAnthropic()
        {
            var ctx = routes.ServiceProvider.GetRequiredService<SharplightServerContext>();
            routes.MapModels(ctx);
            routes.MapPost("/v1/messages", http => Messages(http, ctx));
            routes.MapPost("/v1/messages/count_tokens", http => CountTokens(http, ctx));
            return routes;
        }
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
            "base64" => new ImageData(Base64Payload.Decode(src.Str("data")), src.Str("media_type"), null),
            "url" => new ImageData(null, null, src.Str("url")),
            var t => throw ApiException.BadRequest($"Unsupported image source type '{t}'."),
        };
    }

    private static void AddMessage(List<ChatTurn> turns, JsonElement m, int i)
    {
        string role = m.Str("role") ?? throw ApiException.BadRequest($"messages.{i}.role: Field required");
        if (role is not ("user" or "assistant"))
        {
            throw ApiException.BadRequest($"messages.{i}.role: Input should be 'user' or 'assistant'");
        }

        var content = m.Get("content") ?? throw ApiException.BadRequest($"messages.{i}.content: Field required");
        if (content.ValueKind == JsonValueKind.String)
        {
            turns.Add(new ChatTurn(role, content.GetString()));
            return;
        }
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
                case "text":
                    parts.Add(ChatPart.FromText(blk.Str("text") ?? ""));
                    break;
                case "image":
                    parts.Add(ChatPart.FromImage(ParseImage(blk)));
                    break;
                case "thinking":
                    reasoning.Append(blk.Str("thinking"));
                    break;
                case "redacted_thinking":
                    break;
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
                case var t:
                    throw ApiException.BadRequest($"messages.{i}.content: unsupported block type '{t}'");
            }
        }
        // Tool results (always leading in a user turn) become tool messages, the remaining blocks one user/assistant message.
        turns.AddRange(toolResults);
        if (parts.Count > 0 || calls.Count > 0 || reasoning.Length > 0)
        {
            turns.Add(new ChatTurn(role, null, parts.Count > 0 ? parts : null, reasoning.Length > 0 ? reasoning.ToString() : null, calls.Count > 0 ? calls : null));
        }
    }

    /// <summary>Structured outputs: <c>output_format: {"type": "json_schema", "schema": {..}}</c>.</summary>
    private static OutputFormat? ParseOutputFormat(JsonElement b)
    {
        if (b.Obj("output_format") is not { } of)
        {
            return null;
        }

        return of.Str("type") switch
        {
            "json_schema" => of.Obj("schema") is { } schema ? OutputFormat.JsonSchema(schema.GetRawText()) : OutputFormat.JsonObject,
            var t => throw ApiException.BadRequest($"output_format.type: unsupported value '{t}'", "output_format"),
        };
    }

    private static ChatRequest ParseRequest(JsonElement b, SharplightServerContext ctx, HttpContext http, bool requireMaxTokens)
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

        int i = 0;
        int count = 0;
        foreach (var m in b.Arr("messages"))
        {
            AddMessage(turns, m, i++);
            count++;
        }
        if (count == 0)
        {
            throw ApiException.BadRequest("messages: Field required", "messages");
        }

        var tools = b.Arr("tools").Where(t => t.Str("type") is null or "custom")
            .Select(t => new ToolSpec(t.Str("name") ?? throw ApiException.BadRequest("tools.name: Field required"), t.Str("description"), t.Obj("input_schema")?.GetRawText(), t.Bool("strict") ?? false)).ToList();
        var mode = ToolChoiceMode.Auto;
        string? fname = null;
        if (b.Obj("tool_choice") is { } tc)
        {
            switch (tc.Str("type"))
            {
                case "any":
                    mode = ToolChoiceMode.Required;
                    break;
                case "none":
                    mode = ToolChoiceMode.None;
                    break;
                case "tool":
                    mode = ToolChoiceMode.Function;
                    fname = tc.Str("name");
                    break;
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
            ResponseFormat = ParseOutputFormat(b), Tenant = Tenants.Of(http, b.Obj("metadata")?.Str("user_id")),
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
        try
        {
            return JsonNode.Parse(json) ?? new JsonObject();
        }
        catch (JsonException)
        {
            return new JsonObject();
        }
    }

    private static async Task CountTokens(HttpContext http, SharplightServerContext ctx)
    {
        using var doc = await HttpJson.ReadBodyAsync(http, ctx.Options.MaxRequestBodyBytes);
        var req = ParseRequest(doc.RootElement, ctx, http, requireMaxTokens: false);
        using var lease = await ctx.AcquireAsync(http, doc.RootElement.Str("model"));
        await HttpJson.WriteAsync(http, new JsonObject { ["input_tokens"] = SharplightServerContext.ChatOf(lease).CountTokens(req) });
    }

    private static async Task Messages(HttpContext http, SharplightServerContext ctx)
    {
        using var doc = await HttpJson.ReadBodyAsync(http, ctx.Options.MaxRequestBodyBytes);
        var b = doc.RootElement;
        var req = ParseRequest(b, ctx, http, requireMaxTokens: true);
        bool stream = b.Bool("stream") ?? false;
        using var lease = await ctx.AcquireAsync(http, b.Str("model"));   // held until the response is complete
        var run = SharplightServerContext.ChatOf(lease).StartChat(req, http.RequestAborted);
        string id = "msg_" + Guid.NewGuid().ToString("N")[..24];

        if (stream)
        {
            await StreamMessage(http, run, id, lease.Name);
            return;
        }

        var res = await run.CompleteAsync(http.RequestAborted);
        var choice = res.Choices[0];
        if (FailureOf(choice) is { } failure)
        {
            await ApiErrors.WriteAsync(http, failure.Status, failure.Type, failure.Message);
            return;
        }

        await HttpJson.WriteAsync(http, new JsonObject
        {
            ["id"] = id,
            ["type"] = "message",
            ["role"] = "assistant",
            ["model"] = lease.Name,
            ["content"] = ContentBlocks(choice),
            ["stop_reason"] = StopReason(choice.Finish, choice.StopSequence),
            ["stop_sequence"] = choice.StopSequence,
            ["usage"] = UsageJson(res.Usage),
        });
    }

    private static (int Status, string Type, string Message)? FailureOf(ChatChoice choice) => choice.Finish switch
    {
        ChatFinishReason.Timeout => (504, "api_error", choice.Error ?? "The request timed out."),
        ChatFinishReason.Error => (500, "api_error", choice.Error ?? "Generation failed."),
        ChatFinishReason.Cancelled => (529, "overloaded_error", "The request was cancelled because the server is shutting down."),
        _ => null,
    };

    private static JsonArray ContentBlocks(ChatChoice choice)
    {
        var content = new JsonArray();
        if (choice.Reasoning is not null)
        {
            content.Add(new JsonObject { ["type"] = "thinking", ["thinking"] = choice.Reasoning, ["signature"] = "" });
        }

        if (choice.Content.Length > 0 || choice.ToolCalls.Count == 0)
        {
            content.Add(new JsonObject { ["type"] = "text", ["text"] = choice.Content });
        }

        foreach (var t in choice.ToolCalls)
        {
            content.Add(new JsonObject { ["type"] = "tool_use", ["id"] = ToolUseId(t.Id), ["name"] = t.Name, ["input"] = ParseInput(t.ArgumentsJson) });
        }

        return content;
    }

    private static async Task StreamMessage(HttpContext http, ChatRun run, string id, string model)
    {
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

        var blocks = new AnthropicBlockWriter(sse);
        await SsePump.PumpAsync(run.ReadAllAsync(http.RequestAborted), sse, e =>
        {
            if (e.Content is not null)
            {
                blocks.Text(e.Content);
            }
            else if (e.Reasoning is not null)
            {
                blocks.Thinking(e.Reasoning);
            }
            else if (e.ToolCall is { } t)
            {
                blocks.ToolUse(ToolUseId(t.Id), t.Name, t.ArgumentsJson);
            }
            else if (e.Finish is { } f)
            {
                if (f is ChatFinishReason.Error or ChatFinishReason.Timeout or ChatFinishReason.Cancelled)
                {
                    sse.Event("error", new JsonObject
                    {
                        ["type"] = "error",
                        ["error"] = new JsonObject { ["type"] = f == ChatFinishReason.Cancelled ? "overloaded_error" : "api_error", ["message"] = e.Error ?? "generation " + f.ToString().ToLowerInvariant() },
                    });
                    return false;
                }

                blocks.CloseAll();
                var u = e.Usage ?? default;
                sse.Event("message_delta", new JsonObject
                {
                    ["type"] = "message_delta",
                    ["delta"] = new JsonObject { ["stop_reason"] = StopReason(f, e.StopSequence), ["stop_sequence"] = e.StopSequence },
                    ["usage"] = new JsonObject { ["input_tokens"] = u.PromptTokens, ["cache_read_input_tokens"] = u.CachedTokens, ["output_tokens"] = u.CompletionTokens },
                });
                sse.Event("message_stop", new JsonObject { ["type"] = "message_stop" });
            }

            return true;
        }, http.RequestAborted);
        await sse.FlushAsync();
    }
}
