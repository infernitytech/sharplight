using System.Text.Json;
using Sharplight.Templating;

namespace Sharplight.Templating.Tests;

internal static class TestData
{
    public static readonly DateTimeOffset FixedNow = new(2026, 1, 15, 10, 30, 45, TimeSpan.Zero);

    public static string Path(params string[] parts) => System.IO.Path.Combine([AppContext.BaseDirectory, .. parts]);

    public static JsonElement LoadJson(string relative)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path(relative)));
        return doc.RootElement.Clone();
    }

    public static string ReadTemplate(string name) => File.ReadAllText(Path("Templates", name + ".jinja"));

    public static IEnumerable<string> TemplateNames() =>
        Directory.GetFiles(Path("Templates"), "*.jinja").Select(f => System.IO.Path.GetFileNameWithoutExtension(f)!).OrderBy(n => n, StringComparer.Ordinal);

    public static Dictionary<string, object?> ToContext(JsonElement obj)
    {
        var d = new Dictionary<string, object?>();
        if (JinjaJson.Parse(obj.GetRawText()) is JinjaDict jd)
            foreach (var kv in jd) d[(string)kv.Key] = kv.Value;
        return d;
    }

    public static JinjaEnvironment ChatEnv()
    {
        var env = ChatTemplate.CreateDefaultEnvironment();
        env.Clock = () => FixedNow;
        return env;
    }

    public static ChatMessage ToMessage(JsonElement m)
    {
        string role = m.GetProperty("role").GetString()!;
        string? content = null;
        List<ChatContentPart>? parts = null;
        if (m.TryGetProperty("content", out var c))
        {
            if (c.ValueKind == JsonValueKind.String) content = c.GetString();
            else if (c.ValueKind == JsonValueKind.Array)
            {
                parts = new();
                foreach (var p in c.EnumerateArray())
                    parts.Add(new ChatContentPart(p.GetProperty("type").GetString()!, p.TryGetProperty("text", out var t) ? t.GetString() : null));
            }
        }
        string? reasoning = m.TryGetProperty("reasoning_content", out var r) ? r.GetString() : null;
        List<ChatToolCall>? calls = null;
        if (m.TryGetProperty("tool_calls", out var tc))
        {
            calls = new();
            foreach (var call in tc.EnumerateArray())
            {
                var fn = call.GetProperty("function");
                var args = ToContext(fn.GetProperty("arguments"));
                calls.Add(new ChatToolCall(call.TryGetProperty("id", out var id) ? id.GetString() : null, fn.GetProperty("name").GetString()!, args));
            }
        }
        return new ChatMessage(role, content, parts, reasoning, calls,
            m.TryGetProperty("name", out var n) ? n.GetString() : null,
            m.TryGetProperty("tool_call_id", out var tid) ? tid.GetString() : null);
    }

    public static (List<ChatMessage> Messages, List<ToolDefinition>? Tools) ToChat(JsonElement caseDef)
    {
        var msgs = caseDef.GetProperty("messages").EnumerateArray().Select(ToMessage).ToList();
        List<ToolDefinition>? tools = null;
        if (caseDef.TryGetProperty("tools", out var t) && t.ValueKind == JsonValueKind.Array)
            tools = t.EnumerateArray().Select(x => ToolDefinition.FromJson(x.GetRawText())).ToList();
        return (msgs, tools);
    }
}
