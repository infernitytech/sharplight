namespace Sharplight.Templating;

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
                if (p.Text != null)
                {
                    pd["text"] = p.Text;
                }

                list.Add(pd);
            }
            d["content"] = list;
        }
        else if (Content != null)
        {
            d["content"] = Content;
        }
        else
        {
            d["content"] = ToolCalls is { Count: > 0 } ? null : "";
        }

        if (ReasoningContent != null)
        {
            d["reasoning_content"] = ReasoningContent;
        }

        if (ToolCalls is { Count: > 0 })
        {
            var calls = new List<object?>();
            foreach (var c in ToolCalls)
            {
                var fn = new JinjaDict { ["name"] = c.Name, ["arguments"] = JinjaValue.From(c.Arguments) ?? new JinjaDict() };
                var cd = new JinjaDict();
                if (c.Id != null)
                {
                    cd["id"] = c.Id;
                }

                cd["type"] = "function";
                cd["function"] = fn;
                calls.Add(cd);
            }
            d["tool_calls"] = calls;
        }
        if (Name != null)
        {
            d["name"] = Name;
        }

        if (ToolCallId != null)
        {
            d["tool_call_id"] = ToolCallId;
        }

        return d;
    }
}
