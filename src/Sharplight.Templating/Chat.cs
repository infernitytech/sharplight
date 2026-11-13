namespace Sharplight.Templating;

/// <summary>One part of a multimodal message: <c>text</c>, <c>image</c> or <c>video</c>.</summary>
public sealed record ChatContentPart(string Type, string? Text = null)
{
    public static ChatContentPart FromText(string text) => new("text", text);
    public static ChatContentPart Image() => new("image");
    public static ChatContentPart Video() => new("video");
}

public sealed record ChatToolCall(string? Id, string Name, IReadOnlyDictionary<string, object?>? Arguments = null);

/// <summary>A chat message. Give either <see cref="Content"/> (plain text) or <see cref="Parts"/> (multimodal).</summary>
public sealed record ChatMessage(
    string Role,
    string? Content = null,
    IReadOnlyList<ChatContentPart>? Parts = null,
    string? ReasoningContent = null,
    IReadOnlyList<ChatToolCall>? ToolCalls = null,
    string? Name = null,
    string? ToolCallId = null)
{
    internal JinjaDict ToValue()
    {
        var d = new JinjaDict { ["role"] = Role };
        if (Parts != null)
        {
            var list = new List<object?>();
            foreach (var p in Parts)
            {
                var pd = new JinjaDict { ["type"] = p.Type };
                if (p.Text != null) pd["text"] = p.Text;
                list.Add(pd);
            }
            d["content"] = list;
        }
        else if (Content != null) d["content"] = Content;
        else d["content"] = ToolCalls is { Count: > 0 } ? null : "";

        if (ReasoningContent != null) d["reasoning_content"] = ReasoningContent;
        if (ToolCalls is { Count: > 0 })
        {
            var calls = new List<object?>();
            foreach (var c in ToolCalls)
            {
                var fn = new JinjaDict { ["name"] = c.Name, ["arguments"] = JinjaValue.From(c.Arguments) ?? new JinjaDict() };
                var cd = new JinjaDict();
                if (c.Id != null) cd["id"] = c.Id;
                cd["type"] = "function";
                cd["function"] = fn;
                calls.Add(cd);
            }
            d["tool_calls"] = calls;
        }
        if (Name != null) d["name"] = Name;
        if (ToolCallId != null) d["tool_call_id"] = ToolCallId;
        return d;
    }
}

/// <summary>A function tool offered to the model. <see cref="Parameters"/> is a JSON-schema object.</summary>
public sealed record ToolDefinition(string Name, string? Description = null, IReadOnlyDictionary<string, object?>? Parameters = null)
{
    internal JinjaDict ToValue()
    {
        var fn = new JinjaDict { ["name"] = Name };
        if (Description != null) fn["description"] = Description;
        if (Parameters != null) fn["parameters"] = JinjaValue.From(Parameters);
        return new JinjaDict { ["type"] = "function", ["function"] = fn };
    }

    /// <summary>
    /// Parses an OpenAI-style tool object (<c>{"type":"function","function":{"name":..,"description":..,"parameters":{..}}}</c>)
    /// or the bare function object. Property order is preserved.
    /// </summary>
    public static ToolDefinition FromJson(string json)
    {
        if (JinjaJson.Parse(json) is not JinjaDict root) throw new ArgumentException("Tool definition must be a JSON object", nameof(json));
        if (root.TryGetValue("function", out var f) && f is JinjaDict fd) root = fd;
        string name = root.TryGetValue("name", out var n) && n is string ns ? ns : throw new ArgumentException("Tool definition has no name", nameof(json));
        string? desc = root.TryGetValue("description", out var d) ? d as string : null;
        IReadOnlyDictionary<string, object?>? pars = null;
        if (root.TryGetValue("parameters", out var p) && p is JinjaDict pd)
            pars = pd.ToDictionary(kv => (string)kv.Key, kv => kv.Value);
        return new ToolDefinition(name, desc, pars);
    }
}

/// <summary>
/// A Hugging Face style chat template (the <c>tokenizer.chat_template</c> GGUF/HF field). Uses the HF Jinja settings:
/// <c>trim_blocks</c> and <c>lstrip_blocks</c> on, <c>tojson</c> with <c>ensure_ascii=False</c>.
/// </summary>
public sealed class ChatTemplate
{
    private readonly JinjaTemplate _template;
    private readonly string? _bos, _eos;

    public ChatTemplate(string source, string? bosToken, string? eosToken, JinjaEnvironment? environment = null)
    {
        Environment = environment ?? CreateDefaultEnvironment();
        _template = Environment.Compile(source);
        _bos = bosToken;
        _eos = eosToken;
    }

    public JinjaEnvironment Environment { get; }

    /// <summary>Environment with the HF settings (trim_blocks=True, lstrip_blocks=True).</summary>
    public static JinjaEnvironment CreateDefaultEnvironment() => new() { TrimBlocks = true, LstripBlocks = true };

    /// <summary>
    /// Renders the prompt. Template variables: <c>messages</c>, <c>add_generation_prompt</c>, <c>bos_token</c>, <c>eos_token</c>,
    /// <c>tools</c> (none when null) and everything in <paramref name="extraContext"/> (e.g. <c>enable_thinking</c>).
    /// </summary>
    public string Render(
        IReadOnlyList<ChatMessage> messages,
        bool addGenerationPrompt,
        IReadOnlyDictionary<string, object?>? extraContext = null,
        IReadOnlyList<ToolDefinition>? tools = null)
    {
        var ctx = new Dictionary<string, object?>
        {
            ["bos_token"] = _bos ?? "",
            ["eos_token"] = _eos ?? "",
            ["add_generation_prompt"] = addGenerationPrompt,
        };
        var msgs = new List<object?>(messages.Count);
        foreach (var m in messages) msgs.Add(m.ToValue());
        ctx["messages"] = msgs;
        if (tools is { Count: > 0 })
        {
            var tl = new List<object?>(tools.Count);
            foreach (var t in tools) tl.Add(t.ToValue());
            ctx["tools"] = tl;
        }
        else ctx["tools"] = null;
        if (extraContext != null)
            foreach (var kv in extraContext) ctx[kv.Key] = kv.Value;
        return _template.Render(ctx);
    }
}
