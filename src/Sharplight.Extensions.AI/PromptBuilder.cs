using System.Text.Json;
using Microsoft.Extensions.AI;
using Sharplight.Llm;
using Sharplight.Templating;
using Sharplight.Vision;
using AIChatMessage = Microsoft.Extensions.AI.ChatMessage;
using TChatMessage = Sharplight.Templating.ChatMessage;

namespace Sharplight.Extensions.AI;

/// <summary>The pieces of a request after mapping Microsoft.Extensions.AI types to Sharplight types.</summary>
public sealed class PreparedRequest
{
    public required string Prompt { get; init; }
    public required IReadOnlyList<ImageInput> Images { get; init; }
    public required GenerationOptions Options { get; init; }
    public required ToolSchemas Tools { get; init; }
    public required bool HasTools { get; init; }
    /// <summary>The prompt ends inside an open reasoning block.</summary>
    public required bool StartsInReasoning { get; init; }
}

/// <summary>Maps <see cref="AIChatMessage"/>s and <see cref="ChatOptions"/> onto the chat template and <see cref="GenerationOptions"/>.</summary>
public static class PromptBuilder
{
    public static PreparedRequest Prepare(ChatTemplate template, IEnumerable<AIChatMessage> messages, ChatOptions? options,
        SharplightChatClientOptions clientOptions, bool supportsImages, int contextLength = int.MaxValue)
    {
        var images = new List<ImageInput>();
        var msgs = MapMessages(messages, options, clientOptions, supportsImages, images);

        IReadOnlyList<ToolDefinition>? tools = null;
        bool useTools = options?.Tools is { Count: > 0 } && options.ToolMode is not NoneChatToolMode;
        if (useTools)
        {
            tools = MapTools(options!.Tools!);
        }

        var extra = new Dictionary<string, object?>(clientOptions.TemplateVariables)
        {
            ["enable_thinking"] = options?.AdditionalProperties?.TryGetValue("enable_thinking", out var et) == true && et is bool b ? b : clientOptions.EnableThinking,
        };
        if (options?.AdditionalProperties is { } ap)
        {
            foreach (var kv in ap)
            {
                if (kv.Key.StartsWith("template.", StringComparison.Ordinal))
                {
                    extra[kv.Key["template.".Length..]] = kv.Value;
                }
            }
        }

        string prompt = template.Render(msgs, addGenerationPrompt: true, extra, tools);
        return new PreparedRequest
        {
            Prompt = prompt,
            Images = images,
            Options = MapOptions(options, clientOptions, contextLength),
            Tools = new ToolSchemas(useTools ? options!.Tools : null),
            HasTools = tools is { Count: > 0 },
            StartsInReasoning = prompt.TrimEnd().EndsWith("<think>", StringComparison.Ordinal),
        };
    }

    public static GenerationOptions MapOptions(ChatOptions? o, SharplightChatClientOptions client, int contextLength = int.MaxValue)
    {
        var s = client.Sampling;
        if (o is not null)
        {
            if (o.Temperature is { } t)
            {
                s = s with { Temperature = t };
            }

            if (o.TopP is { } p)
            {
                s = s with { TopP = p };
            }

            if (o.TopK is { } k)
            {
                s = s with { TopK = k };
            }

            if (o.Seed is { } seed)
            {
                s = s with { Seed = unchecked((int)seed) };
            }
        }
        return new GenerationOptions
        {
            MaxNewTokens = Math.Max(1, o?.MaxOutputTokens ?? client.DefaultMaxOutputTokens),
            Sampling = s,
            StopStrings = o?.StopSequences is { Count: > 0 } ss ? [.. ss] : [],
        };
    }

    public static List<TChatMessage> MapMessages(IEnumerable<AIChatMessage> messages, ChatOptions? options,
        SharplightChatClientOptions clientOptions, bool supportsImages, List<ImageInput> images)
    {
        var result = new List<TChatMessage>();
        string? system = options?.Instructions;
        if (options?.ResponseFormat is ChatResponseFormatJson json)
        {
            string fmt = json.Schema is { } schema
                ? $"Respond only with valid JSON that conforms to this JSON schema (no prose, no code fences):\n{schema.GetRawText()}"
                : "Respond only with a valid JSON value (no prose, no code fences).";
            system = system is null ? fmt : system + "\n\n" + fmt;
        }

        foreach (var m in messages)
        {
            string role = m.Role == ChatRole.System || m.Role.Value == "developer" ? "system"
                : m.Role == ChatRole.Assistant ? "assistant"
                : m.Role == ChatRole.Tool ? "tool" : "user";

            var parts = new List<ChatContentPart>();
            var text = new System.Text.StringBuilder();
            var reasoning = new System.Text.StringBuilder();
            var calls = new List<ChatToolCall>();
            var toolResults = new List<TChatMessage>();
            bool hasImage = false;

            foreach (var c in m.Contents)
            {
                switch (c)
                {
                    case TextContent tc:
                        text.Append(tc.Text);
                        parts.Add(ChatContentPart.FromText(tc.Text));
                        break;
                    case TextReasoningContent rc:
                        reasoning.Append(rc.Text);
                        break;
                    case DataContent dc when dc.HasTopLevelMediaType("image"):
                        if (!supportsImages)
                        {
                            throw new NotSupportedException("The model has no vision tower; images are not supported.");
                        }

                        images.Add(clientOptions.ImageDecoder(dc));
                        parts.Add(ChatContentPart.Image());
                        hasImage = true;
                        break;
                    case DataContent dc:
                        throw new NotSupportedException($"Unsupported content type '{dc.MediaType}'. Only text and images are supported.");
                    case UriContent uc:
                        throw new NotSupportedException($"UriContent ('{uc.Uri}') is not supported; download the image and pass it as DataContent.");
                    case FunctionCallContent fc:
                        calls.Add(new ChatToolCall(fc.CallId, fc.Name, ToPlain(fc.Arguments)));
                        break;
                    case FunctionResultContent fr:
                        toolResults.Add(new TChatMessage("tool", Content: ResultToString(fr.Result), ToolCallId: fr.CallId));
                        break;
                }
            }

            if (role == "system")
            {
                // System text is hoisted into one leading system message (templates require it first).
                if (text.Length > 0)
                {
                    system = system is null ? text.ToString() : system + "\n\n" + text;
                }

                continue;
            }
            if (role == "tool")
            {
                if (toolResults.Count == 0 && text.Length > 0)
                {
                    result.Add(new TChatMessage("tool", text.ToString()));
                }
            }
            else if (text.Length == 0 && calls.Count == 0 && !hasImage && toolResults.Count > 0)
            {
                // A message carrying only function results: the results below are the whole content.
            }
            else if (hasImage)
            {
                if (role != "user")
                {
                    throw new NotSupportedException("Images are only supported in user messages.");
                }

                result.Add(new TChatMessage(role, Parts: parts));
            }
            else
            {
                result.Add(new TChatMessage(role,
                    Content: text.Length == 0 && calls.Count > 0 ? null : text.ToString(),
                    ReasoningContent: reasoning.Length > 0 ? reasoning.ToString() : null,
                    ToolCalls: calls.Count > 0 ? calls : null));
            }

            result.AddRange(toolResults);
        }

        if (!string.IsNullOrEmpty(system))
        {
            result.Insert(0, new TChatMessage("system", system));
        }

        return result;
    }

    public static List<ToolDefinition> MapTools(IEnumerable<AITool> tools)
    {
        var list = new List<ToolDefinition>();
        foreach (var t in tools)
        {
            if (t is not AIFunctionDeclaration f)
            {
                continue; // hosted tools (web search, ...) can't run locally
            }

            using var ms = new MemoryStream();
            using (var w = new Utf8JsonWriter(ms))
            {
                w.WriteStartObject();
                w.WriteString("name", f.Name);
                if (!string.IsNullOrEmpty(f.Description))
                {
                    w.WriteString("description", f.Description);
                }

                if (f.JsonSchema.ValueKind == JsonValueKind.Object)
                {
                    w.WritePropertyName("parameters");
                    f.JsonSchema.WriteTo(w);
                }
                w.WriteEndObject();
            }
            list.Add(ToolDefinition.FromJson(System.Text.Encoding.UTF8.GetString(ms.ToArray())));
        }
        return list;
    }

    private static string ResultToString(object? result) => result switch
    {
        null => "",
        string s => s,
        JsonElement { ValueKind: JsonValueKind.String } e => e.GetString()!,
        JsonElement e => e.GetRawText(),
        _ => JsonSerializer.Serialize(result, AIJsonUtilities.DefaultOptions.GetTypeInfo(result.GetType())),
    };

    /// <summary>Converts call arguments (often <see cref="JsonElement"/>s) to plain values the template engine understands.</summary>
    private static Dictionary<string, object?>? ToPlain(IDictionary<string, object?>? args)
    {
        if (args is null)
        {
            return null;
        }

        var d = new Dictionary<string, object?>();
        foreach (var kv in args)
        {
            d[kv.Key] = ToPlain(kv.Value);
        }

        return d;
    }

    private static object? ToPlain(object? v) => v switch
    {
        JsonElement e => e.ValueKind switch
        {
            JsonValueKind.String => e.GetString(),
            JsonValueKind.Number => e.TryGetInt64(out var l) ? (object)l : e.GetDouble(),
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.Object => e.EnumerateObject().ToDictionary(p => p.Name, p => ToPlain(p.Value)),
            JsonValueKind.Array => e.EnumerateArray().Select(x => ToPlain(x)).ToList(),
            _ => null,
        },
        _ => v,
    };
}
